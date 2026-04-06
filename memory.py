# memory.py — Celeste vector memory
# - GPU-first SentenceTransformers embeddings (fallback to CPU)
# - Explicit Chroma usage (no Chroma embedding wrapper)
# - Only index selected kinds to avoid prompt bloat
# - JSON log fallback + TF-IDF retrieval when vector DB unavailable

import os
import hashlib
import json
import re
import time
import uuid
from collections import defaultdict
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config_types import AgentConfig
from graph_memory import GraphMemory

# Keep tokenizer threads quiet
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Only these kinds are stored in the vector DB
INDEXABLE_KINDS = {"note", "user"}
DEFAULT_ENGRAM_NGRAM_MIN = 2
DEFAULT_ENGRAM_NGRAM_MAX = 4
DEFAULT_ENGRAM_MAX_POSTINGS = 256
DEFAULT_ENGRAM_AUTO_PRUNE = True
DEFAULT_ENGRAM_RETENTION_DAYS = 30
DEFAULT_ENGRAM_KEEP_MIN_USES = 3


class EngramMemory:
    def __init__(
        self,
        path: str,
        *,
        enabled: bool = True,
        ngram_min: int = DEFAULT_ENGRAM_NGRAM_MIN,
        ngram_max: int = DEFAULT_ENGRAM_NGRAM_MAX,
        max_postings: int = DEFAULT_ENGRAM_MAX_POSTINGS,
        auto_prune: bool = DEFAULT_ENGRAM_AUTO_PRUNE,
        retention_days: int = DEFAULT_ENGRAM_RETENTION_DAYS,
        keep_min_uses: int = DEFAULT_ENGRAM_KEEP_MIN_USES,
    ):
        self.path = path
        self.enabled = bool(enabled)
        self.ngram_min = max(1, int(ngram_min))
        self.ngram_max = max(self.ngram_min, int(ngram_max))
        self.max_postings = max(16, int(max_postings))
        self.auto_prune = bool(auto_prune)
        self.retention_days = max(1, int(retention_days))
        self.keep_min_uses = max(1, int(keep_min_uses))
        self.entries: List[Dict[str, Any]] = []
        self.postings: dict[str, list[int]] = {}
        if self.enabled:
            self._load()
            self.auto_prune_stale()

    def _normalize_tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _hash_ngram(self, gram: tuple[str, ...]) -> str:
        raw = "\x1f".join(gram).encode("utf-8", errors="ignore")
        return hashlib.blake2b(raw, digest_size=8).hexdigest()

    def _entry_ngrams(self, text: str) -> set[str]:
        tokens = self._normalize_tokens(text)
        grams: set[str] = set()
        for n in range(self.ngram_min, self.ngram_max + 1):
            if len(tokens) < n:
                continue
            for idx in range(0, len(tokens) - n + 1):
                grams.add(self._hash_ngram(tuple(tokens[idx : idx + n])))
        return grams

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self.entries = []
            self.postings = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
        except Exception:
            self.entries = []
            self.postings = {}
            return
        entries = payload.get("entries", [])
        postings = payload.get("postings", {})
        self.entries = [row for row in entries if isinstance(row, dict) and row.get("text")]
        now = time.time()
        for row in self.entries:
            if not isinstance(row.get("created_at"), (int, float)):
                row["created_at"] = now
            if not isinstance(row.get("last_accessed_at"), (int, float)):
                row["last_accessed_at"] = row["created_at"]
            if not isinstance(row.get("hit_count"), int):
                try:
                    row["hit_count"] = int(row.get("hit_count", 0) or 0)
                except Exception:
                    row["hit_count"] = 0
            if not isinstance(row.get("ngrams"), list):
                row["ngrams"] = sorted(self._entry_ngrams(str(row.get("text", "") or "")))
        self.postings = {
            str(key): [int(v) for v in values if isinstance(v, int) or str(v).isdigit()]
            for key, values in postings.items()
            if isinstance(values, list)
        }
        if self.entries and not self.postings:
            self._rebuild_postings()

    def _save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"entries": self.entries, "postings": self.postings}, f)
        except Exception:
            pass

    def _rebuild_postings(self) -> None:
        self.postings = {}
        for entry_id, entry in enumerate(self.entries):
            entry["id"] = entry_id
            grams = entry.get("ngrams", [])
            if not isinstance(grams, list):
                grams = sorted(self._entry_ngrams(str(entry.get("text", "") or "")))
                entry["ngrams"] = grams
            for gram_hash in grams:
                bucket = self.postings.get(str(gram_hash))
                if bucket is None:
                    self.postings[str(gram_hash)] = [entry_id]
                elif not bucket or bucket[-1] != entry_id:
                    bucket.append(entry_id)
                    if len(bucket) > self.max_postings:
                        del bucket[:-self.max_postings]

    def set_auto_prune(self, enabled: bool) -> dict[str, int | bool]:
        self.auto_prune = bool(enabled)
        if self.auto_prune:
            self.auto_prune_stale()
        return self.stats()

    def stats(self) -> dict[str, int | bool]:
        return {
            "enabled": self.enabled,
            "auto_prune": self.auto_prune,
            "entries": len(self.entries),
            "ngrams": len(self.postings),
            "retention_days": self.retention_days,
            "keep_min_uses": self.keep_min_uses,
        }

    def _prune_entries(self, keep_fn) -> int:
        if not self.enabled:
            return 0
        kept = [entry for entry in self.entries if keep_fn(entry)]
        purged = len(self.entries) - len(kept)
        if purged <= 0:
            return 0
        self.entries = kept
        self._rebuild_postings()
        self._save()
        return purged

    def purge_recent(self, seconds: int | None = None) -> dict[str, int | bool]:
        if not self.enabled:
            return {"purged": 0, **self.stats()}
        if seconds is None or int(seconds) <= 0:
            purged = len(self.entries)
            self.entries = []
            self.postings = {}
            self._save()
            return {"purged": purged, **self.stats()}
        cutoff = time.time() - int(seconds)
        purged = self._prune_entries(
            lambda entry: float(entry.get("created_at", time.time())) < cutoff
        )
        return {"purged": purged, **self.stats()}

    def auto_prune_stale(self) -> dict[str, int | bool]:
        if not self.enabled or not self.auto_prune:
            return {"purged": 0, **self.stats()}
        cutoff = time.time() - (self.retention_days * 86400)
        purged = self._prune_entries(
            lambda entry: (
                float(entry.get("created_at", time.time())) >= cutoff
                or int(entry.get("hit_count", 0) or 0) >= self.keep_min_uses
            )
        )
        return {"purged": purged, **self.stats()}

    def rebuild(self, rows: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        self.entries = []
        self.postings = {}
        for row in rows:
            text = str(row.get("text", "") or "").strip()
            kind = str(row.get("kind", "") or "").strip()
            meta = row.get("meta", {}) if isinstance(row.get("meta", {}), dict) else {}
            if text and kind in INDEXABLE_KINDS:
                self.add(text, kind=kind, metadata=meta, save=False)
        self._save()

    def add(
        self,
        text: str,
        *,
        kind: str = "note",
        metadata: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> None:
        if not self.enabled or kind not in INDEXABLE_KINDS:
            return
        clean = (text or "").strip()
        if not clean:
            return
        entry_id = len(self.entries)
        entry_grams = sorted(self._entry_ngrams(clean))
        now = time.time()
        self.entries.append(
            {
                "id": entry_id,
                "text": clean,
                "kind": kind,
                "meta": metadata or {},
                "ngrams": entry_grams,
                "created_at": now,
                "last_accessed_at": now,
                "hit_count": 0,
            }
        )
        for gram_hash in entry_grams:
            bucket = self.postings.get(gram_hash)
            if bucket is None:
                self.postings[gram_hash] = [entry_id]
            elif not bucket or bucket[-1] != entry_id:
                bucket.append(entry_id)
                if len(bucket) > self.max_postings:
                    del bucket[:-self.max_postings]
        if save:
            self._save()

    def search(self, query: str, *, top_k: int = 5, kinds: Optional[List[str]] = None) -> list[dict[str, Any]]:
        if not self.enabled or not self.entries:
            return []
        allowed_kinds = set(kinds or INDEXABLE_KINDS)
        query_grams = self._entry_ngrams(query)
        if not query_grams:
            return []

        hit_counts: dict[int, float] = defaultdict(float)
        for gram_hash in query_grams:
            for entry_id in self.postings.get(gram_hash, [])[-self.max_postings :]:
                hit_counts[int(entry_id)] += 1.0

        ranked: list[dict[str, Any]] = []
        query_norm = max(1.0, float(len(query_grams)))
        for entry_id, hits in hit_counts.items():
            if entry_id < 0 or entry_id >= len(self.entries):
                continue
            entry = self.entries[entry_id]
            kind = str(entry.get("kind", "") or "")
            if allowed_kinds and kind not in allowed_kinds:
                continue
            denom = (query_norm * max(1.0, float(len(entry.get("ngrams", []))))) ** 0.5
            score = float(hits / denom)
            if score <= 0.0:
                continue
            ranked.append(
                {
                    "text": entry.get("text", ""),
                    "score": score,
                    "meta": {
                        **(entry.get("meta", {}) if isinstance(entry.get("meta", {}), dict) else {}),
                        "kind": kind,
                        "memory_channel": "engram",
                    },
                }
            )
        ranked.sort(key=lambda row: row["score"], reverse=True)
        if ranked:
            now = time.time()
            selected_texts = {str(row.get("text", "") or "") for row in ranked[: max(1, top_k)]}
            changed = False
            for entry in self.entries:
                if str(entry.get("text", "") or "") in selected_texts:
                    entry["hit_count"] = int(entry.get("hit_count", 0) or 0) + 1
                    entry["last_accessed_at"] = now
                    changed = True
            if changed:
                self._save()
        return ranked[: max(1, top_k)]


class MemoryPipeline:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg

        # Paths
        os.makedirs(self.cfg.data_dir, exist_ok=True)
        self.json_path = os.path.join(self.cfg.data_dir, "memory.json")
        self.engram_path = os.path.join(self.cfg.data_dir, "memory_engram.json")
        if getattr(self.cfg, "use_chroma", False):
            os.makedirs(getattr(self.cfg, "persist_dir", self.cfg.data_dir), exist_ok=True)

        # State
        self.use_chroma = False
        self.client = None
        self.collection = None
        self.embedder = None
        self.dim: Optional[int] = None
        self.device = "cpu"
        memory_cfg = getattr(self.cfg, "memory", {}) or {}
        if not isinstance(memory_cfg, dict):
            memory_cfg = {}
        self.engram = EngramMemory(
            self.engram_path,
            enabled=bool(memory_cfg.get("engram_enabled", True)),
            ngram_min=int(memory_cfg.get("engram_ngram_min", DEFAULT_ENGRAM_NGRAM_MIN)),
            ngram_max=int(memory_cfg.get("engram_ngram_max", DEFAULT_ENGRAM_NGRAM_MAX)),
            max_postings=int(memory_cfg.get("engram_max_postings", DEFAULT_ENGRAM_MAX_POSTINGS)),
            auto_prune=bool(memory_cfg.get("engram_auto_prune", DEFAULT_ENGRAM_AUTO_PRUNE)),
            retention_days=int(memory_cfg.get("engram_retention_days", DEFAULT_ENGRAM_RETENTION_DAYS)),
            keep_min_uses=int(memory_cfg.get("engram_keep_min_uses", DEFAULT_ENGRAM_KEEP_MIN_USES)),
        )
        self.graph = GraphMemory(
            os.path.join(self.cfg.data_dir, "memory_graph.sqlite3"),
            enabled=bool(memory_cfg.get("graph_enabled", True)),
        )

        # Ensure JSON store exists
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        if self.engram.enabled and not os.path.exists(self.engram_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
            except Exception:
                rows = []
            if isinstance(rows, list):
                self.engram.rebuild(rows)

        # Vector store + embedder
        if getattr(self.cfg, "use_chroma", False):
            try:
                # Detect CUDA for embeddings (Torch 2.6+cu124 in your venv)
                try:
                    import torch as _torch  # noqa: F401
                    self.device = "cuda" if _torch.cuda.is_available() else "cpu"
                except Exception:
                    self.device = "cpu"

                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(self.cfg.embedding_model, device=self.device)
                self.dim = int(self.embedder.get_sentence_embedding_dimension())

                import chromadb
                self.client = chromadb.PersistentClient(path=self.cfg.persist_dir)
                self.collection = self.client.get_or_create_collection(name="agent_mem")
                self.use_chroma = True
                print(f"[memory] vector store ready (device={self.device}, dim={self.dim})")
            except Exception as e:
                print(f"[memory] Chroma unavailable or embedder load failed ({e}). Falling back to JSON+TFIDF.")

    # ---- Helpers ----
    def _append_json(self, text: str, kind: str, metadata: Dict[str, Any]):
        """Append to the JSON log (always)."""
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
        data.append({"text": text, "kind": kind, "meta": metadata})
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode with SentenceTransformers on GPU if available, then L2-normalize.
        Returns float32 numpy on CPU (what Chroma expects).
        """
        # encode() runs on self.device internally; result is a CPU numpy array
        vec = self.embedder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalize ourselves for explicit control
            show_progress_bar=False,
        )
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return (vec / norms).astype(np.float32)

    # ---- Public API ----
    def add(self, text: str, kind: str = "note", metadata: Optional[Dict[str, Any]] = None):
        """
        Persist an item to the JSON log, and (if enabled) index in Chroma when kind is indexable.
        """
        metadata = metadata or {}

        # Always log to JSON for transparency/audit
        self._append_json(text, kind, metadata)
        self.engram.add(text, kind=kind, metadata=metadata)
        self.graph.observe_text(text, kind=kind, metadata=metadata)

        # Index selected kinds in vector DB
        if self.use_chroma and self.embedder is not None and self.collection is not None:
            if kind in INDEXABLE_KINDS:
                try:
                    _id = str(uuid.uuid4())
                    vec = self._encode([text])
                    self.collection.add(
                        ids=[_id],
                        documents=[text],
                        metadatas=[{"kind": kind, **metadata}],
                        embeddings=vec.tolist(),
                    )
                except Exception as e:
                    print(f"[memory] Chroma add failed ({e}). Continuing with JSON only.")

    def search(self, query: str, top_k: int = 5, kinds: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search vector DB first (filtered by kinds if provided), otherwise fall back to JSON+TFIDF.
        Returns a list of {text, score, meta}.
        """
        kinds = kinds or list(INDEXABLE_KINDS)

        # Vector path
        if self.use_chroma and self.embedder is not None and self.collection is not None:
            try:
                qv = self._encode([query]).tolist()
                where = {"kind": {"$in": kinds}} if kinds else None
                res = self.collection.query(query_embeddings=qv, n_results=top_k, where=where)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]
                out: List[Dict[str, Any]] = []
                for d, m, dist in zip(docs, metas, dists):
                    score = (1.0 - dist) if dist is not None else 0.0
                    out.append({"text": d, "score": float(score), "meta": m or {}})
                return out
            except Exception as e:
                print(f"[memory] Chroma query failed ({e}). Falling back to JSON+TFIDF.")

        # JSON fallback path (TF-IDF cosine)
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []

        # Filter by kind to avoid assistant/meta bloat
        if kinds:
            data = [d for d in data if d.get("kind") in kinds]

        if not data:
            return []

        texts = [d["text"] for d in data]
        vec = TfidfVectorizer(max_features=4096).fit(texts + [query])
        X = vec.transform(texts)
        q = vec.transform([query])
        sims = cosine_similarity(q, X)[0]
        idxs = sims.argsort()[::-1][:top_k]
        out: List[Dict[str, Any]] = []
        for i in idxs:
            out.append({"text": texts[i], "score": float(sims[i]), "meta": data[i].get("meta", {})})
        return out

    def search_engram(self, query: str, top_k: int = 5, kinds: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.engram.search(query, top_k=top_k, kinds=kinds)

    def search_graph(
        self,
        query: str,
        top_k: int = 4,
        *,
        evidence_texts: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self.graph.search(query, top_k=top_k, evidence_texts=evidence_texts)

    def purge_engram(self, seconds: int | None = None) -> dict[str, int | bool]:
        return self.engram.purge_recent(seconds=seconds)

    def set_engram_auto_prune(self, enabled: bool) -> dict[str, int | bool]:
        return self.engram.set_auto_prune(enabled)

    def engram_stats(self) -> dict[str, int | bool]:
        return self.engram.stats()
