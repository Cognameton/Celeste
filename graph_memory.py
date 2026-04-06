import math
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


GRAPH_DEFAULT_ENABLED = True
GRAPH_DEFAULT_RECALL_K = 4


class GraphMemory:
    def __init__(self, path: str, *, enabled: bool = GRAPH_DEFAULT_ENABLED):
        self.path = path
        self.enabled = bool(enabled)
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    evidence TEXT NOT NULL DEFAULT '',
                    source_channel TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    mention_count INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    UNIQUE(subject, relation, object)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_facts_last_seen
                ON facts(last_seen_at DESC)
                """
            )

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9][a-z0-9_./-]*", (text or "").lower())

    def _clean_object(self, value: str) -> str:
        clean = re.sub(r"\s+", " ", (value or "").strip(" \t\r\n\"'`.,;:!?()[]{}"))
        clean = re.split(r"\s+(?:and|but|because|so|though|although)\s+", clean, maxsplit=1)[0].strip()
        return clean[:180]

    def _fact_text(self, row: sqlite3.Row | Dict[str, Any]) -> str:
        subject = str(row["subject"])
        relation = str(row["relation"])
        obj = str(row["object"])
        if relation == "name_is":
            return f"{subject} name is {obj}"
        if relation == "prefers_name":
            return f"{subject} prefers to be called {obj}"
        if relation == "prefers":
            return f"{subject} prefers {obj}"
        if relation == "likes":
            return f"{subject} likes {obj}"
        if relation == "wants":
            return f"{subject} wants {obj}"
        if relation == "needs":
            return f"{subject} needs {obj}"
        if relation == "uses":
            return f"{subject} uses {obj}"
        if relation == "works_on":
            return f"{subject} is working on {obj}"
        if relation == "current_branch":
            return f"{subject} current branch is {obj}"
        if relation == "default_model":
            return f"{subject} default model is {obj}"
        return f"{subject} {relation.replace('_', ' ')} {obj}"

    def _extract_pattern_facts(self, text: str, *, channel: str) -> List[Dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return []
        lowered = raw.lower()
        facts: List[Dict[str, Any]] = []

        patterns = [
            (r"\bmy name is ([^.!?\n]+)", "User", "name_is", 0.98),
            (r"\bcall me ([^.!?\n]+)", "User", "prefers_name", 0.96),
            (r"\bi prefer ([^.!?\n]+)", "User", "prefers", 0.78),
            (r"\bi like ([^.!?\n]+)", "User", "likes", 0.72),
            (r"\bi love ([^.!?\n]+)", "User", "likes", 0.74),
            (r"\bi want ([^.!?\n]+)", "User", "wants", 0.70),
            (r"\bi need ([^.!?\n]+)", "User", "needs", 0.72),
            (r"\bi use ([^.!?\n]+)", "User", "uses", 0.68),
            (r"\bi(?: am|'m) working on ([^.!?\n]+)", "User", "works_on", 0.86),
            (r"\bwe(?: are|'re) working on ([^.!?\n]+)", "Project", "works_on", 0.84),
            (r"\byour name is ([^.!?\n]+)", "Assistant", "name_is", 0.97),
            (r"\b(?:the )?current branch is ([a-zA-Z0-9_./-]+)", "Project", "current_branch", 0.92),
            (r"\bbranch is ([a-zA-Z0-9_./-]+)", "Project", "current_branch", 0.88),
            (r"\bdefault model is ([^.!?\n]+)", "Project", "default_model", 0.88),
            (r"\buse ([^.!?\n]+) as the default model", "Project", "default_model", 0.86),
        ]

        for pattern, subject, relation, confidence in patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if not match:
                continue
            start, end = match.span(1)
            obj = self._clean_object(raw[start:end])
            if not obj:
                continue
            facts.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "evidence": raw[:500],
                    "source_channel": channel,
                    "confidence": confidence,
                }
            )
        return facts

    def observe_text(self, text: str, *, kind: str = "note", metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        channel = str(kind or "note")
        facts = self._extract_pattern_facts(text, channel=channel)
        if not facts and channel == "note":
            facts = self._extract_pattern_facts(str(text or ""), channel="note")
        if not facts:
            return
        for fact in facts:
            self.add_fact(**fact)

    def add_fact(
        self,
        *,
        subject: str,
        relation: str,
        object: str,
        evidence: str = "",
        source_channel: str = "",
        confidence: float = 0.5,
    ) -> None:
        if not self.enabled:
            return
        subject = self._clean_object(subject)
        relation = re.sub(r"[^a-z0-9_]+", "_", (relation or "").strip().lower()).strip("_")
        object = self._clean_object(object)
        if not subject or not relation or not object:
            return
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, mention_count, confidence FROM facts WHERE subject = ? AND relation = ? AND object = ?",
                (subject, relation, object),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO facts (
                        subject, relation, object, evidence, source_channel, confidence,
                        mention_count, created_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (subject, relation, object, evidence[:500], source_channel, float(confidence), now, now),
                )
                return
            merged_confidence = max(float(row["confidence"]), float(confidence))
            conn.execute(
                """
                UPDATE facts
                SET evidence = CASE WHEN ? <> '' THEN ? ELSE evidence END,
                    source_channel = CASE WHEN ? <> '' THEN ? ELSE source_channel END,
                    confidence = ?,
                    mention_count = ?,
                    last_seen_at = ?
                WHERE id = ?
                """,
                (
                    evidence[:500],
                    evidence[:500],
                    source_channel,
                    source_channel,
                    merged_confidence,
                    int(row["mention_count"]) + 1,
                    now,
                    int(row["id"]),
                ),
            )

    def search(
        self,
        query: str,
        *,
        top_k: int = GRAPH_DEFAULT_RECALL_K,
        evidence_texts: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT subject, relation, object, evidence, source_channel, confidence, mention_count, last_seen_at
                FROM facts
                ORDER BY mention_count DESC, last_seen_at DESC
                LIMIT 2048
                """
            ).fetchall()
        if not rows:
            return []

        evidence_texts = [str(text or "").strip() for text in (evidence_texts or []) if str(text or "").strip()]
        expanded_query_parts = [str(query or "").strip()]
        expanded_query_parts.extend(evidence_texts[:12])
        expanded_query = "\n".join(part for part in expanded_query_parts if part)
        if not expanded_query.strip():
            return []

        corpus = [self._fact_text(row) + f"\n{row['evidence']}" for row in rows]
        try:
            vectorizer = TfidfVectorizer(max_features=2048, ngram_range=(1, 2))
            matrix = vectorizer.fit_transform(corpus + [expanded_query])
            sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
        except Exception:
            sims = [0.0 for _ in corpus]

        ranked: List[Dict[str, Any]] = []
        for row, sim in zip(rows, sims):
            text = self._fact_text(row)
            score = float(sim)
            score += min(0.35, 0.05 * math.log1p(max(0, int(row["mention_count"]))))
            score += 0.25 * float(row["confidence"])
            if score <= 0.12:
                continue
            ranked.append(
                {
                    "text": text,
                    "score": score,
                    "meta": {
                        "subject": row["subject"],
                        "relation": row["relation"],
                        "object": row["object"],
                        "source_channel": row["source_channel"],
                        "memory_channel": "graph",
                    },
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[: max(1, int(top_k))]

