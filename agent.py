# agent.py — Celeste agent (chat-style prompt, strict stops, perspective guard)
import os
import json
import logging
import re
from typing import Callable, Dict, Any, Tuple, Optional, List

from rich.console import Console

from config_types import AgentConfig
from model_runner import LLMRunner
from memory import MemoryPipeline
from file_rag import FileRAG, FILE_RAG_INDEX_KIND_SEMANTIC, FILE_RAG_INDEX_KIND_TFIDF
from reflection import Reflector
from playbook import Playbook
from tts import TTSManager

console = Console()


class Agent:
    def __init__(self, cfg: AgentConfig, status_cb: Callable[[str], None] | None = None):
        self.cfg = cfg
        self._status_cb = status_cb
        self._emit_status("Preparing Celeste workspace...")
        os.makedirs(self.cfg.data_dir, exist_ok=True)

        # Defaults for self-state
        self._default_state = {"identity": "Celeste", "focus": "helpful, precise, offline"}
        self._live_guidance: str = ""
        self._tracking_session: dict[str, Any] | None = None
        self._recent_turns: list[dict[str, str]] = []
        self._retry_phrases = (
            "i'm sorry, but i am an ai language model",
            "i'm sorry, but as an ai language model",
            "i'm sorry, but as an ai model",
            "i do not have the ability to provide explanations",
            "i cannot provide explanations",
            "as an ai language model, i cannot",
            "i am not able to",
            "i am unable to",
            "i'm unable to",
            "i cannot help",
            "i can't help",
            "i can't assist",
            "my capabilities are limited",
            "there are many resources available online",
            "i'm sorry, but i am an ai model",
            "i'm sorry, but i am just an ai",
            "i'm sorry, but i can't assist",
            "i'm sorry, but i can't help",
            "i'm just an ai language model",
            "as an ai, i cannot",
            "i'm an ai language model and cannot",
            "i'm a language model and cannot",
            "i do not have the capability",
            "i don't have the ability",
            "i lack the ability",
            "i do not have access to external",
            "there are many resources you can consult",
        )

        # Core subsystems
        self._emit_status(f"Loading language model: {os.path.basename(self.cfg.model_path)}")
        self.llm = LLMRunner(cfg, status_cb=status_cb)
        self._emit_status("Starting memory system...")
        self.mem = MemoryPipeline(cfg)
        self._emit_status("Starting file search...")
        self.file_rag = FileRAG(
            cfg,
            shared_embedder=getattr(self.mem, "embedder", None),
            shared_device=getattr(self.mem, "device", "cpu"),
        )
        self._emit_status("Starting reflection tools...")
        self.reflector = Reflector(self.llm, cfg)
        self._emit_status("Starting playbook and speech...")
        self.playbook = Playbook(os.path.join(cfg.data_dir, "playbook.md"))
        self.tts = TTSManager(cfg)

        # Self-state (identity/focus/etc.)
        self.self_state_path = os.path.join(self.cfg.data_dir, "self_state.json")
        if not os.path.exists(self.self_state_path):
            with open(self.self_state_path, "w", encoding="utf-8") as f:
                json.dump(self._default_state, f, indent=2)
        self._emit_status("Celeste startup complete.")

    def _emit_status(self, message: str) -> None:
        logging.info("Agent status: %s", message)
        if self._status_cb:
            self._status_cb(message)

    def close(self) -> None:
        try:
            self.tts.shutdown()
        except Exception:
            pass
        try:
            self.llm.shutdown()
        except Exception:
            pass

    def set_file_rag_dirs(self, directories: List[str]) -> Dict[str, Any]:
        self.cfg.file_rag_dirs = directories
        return self.file_rag.rebuild(directories)

    def reindex_file_rag(self) -> Dict[str, Any]:
        return self.file_rag.rebuild(self.cfg.file_rag_dirs)

    def purge_engram_memory(self, seconds: int | None = None) -> Dict[str, Any]:
        return self.mem.purge_engram(seconds=seconds)

    def set_engram_auto_prune(self, enabled: bool) -> Dict[str, Any]:
        memory_cfg = dict((self.cfg.memory or {}) if isinstance(self.cfg.memory, dict) else {})
        memory_cfg["engram_auto_prune"] = bool(enabled)
        self.cfg.memory = memory_cfg
        return self.mem.set_engram_auto_prune(enabled)

    def _reflection_enabled(self) -> bool:
        reflection_cfg = getattr(self.cfg, "reflection", {}) or {}
        if isinstance(reflection_cfg, dict):
            return bool(reflection_cfg.get("enabled", False))
        return False

    # ---------- Self-state ----------
    def _load_self_state(self) -> Dict[str, Any]:
        try:
            with open(self.self_state_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = {}
        return self._sanitize_self_state(raw)

    def _save_self_state(self, state: Dict[str, Any]) -> None:
        clean = self._sanitize_self_state(state)
        with open(self.self_state_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)

    def _sanitize_self_state(self, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure identity/focus stay populated with simple strings."""
        clean: Dict[str, Any] = dict(self._default_state)
        if isinstance(state, dict):
            for key, value in state.items():
                if key in ("identity", "focus"):
                    if isinstance(value, str) and value.strip():
                        clean[key] = value.strip()
                else:
                    clean[key] = value
        return clean

    def _maybe_update_identity(self, user_msg: str) -> None:
        """
        Detect patterns like 'your name is ...' and persist the provided identity.
        Keeps existing identity when the extracted value is empty.
        """
        lowered = user_msg.lower()
        trigger = "your name is"
        if trigger not in lowered:
            return

        # Naive span capture after the trigger; accept up to sentence end.
        try:
            start = lowered.index(trigger) + len(trigger)
            proposed = user_msg[start:].strip(" .!?\n\t\"'")
            if proposed:
                state = self._load_self_state()
                state["identity"] = proposed
                self._save_self_state(state)
                self.mem.add(
                    f"Assistant identity updated to {proposed}",
                    kind="meta",
                    metadata={"via": "identity-update"},
                )
        except ValueError:
            pass

    # ---------- Prompt pieces ----------
    def build_system_prompt(self) -> str:
        self_state = self._load_self_state()
        playbook_text = self.playbook.read().strip()

        parts: List[str] = [
            (self.cfg.system_preamble or "").strip(),
            "[Behavior]\n"
            "- Reply concisely.\n"
            "- For code/IT tasks, use ordered steps.\n"
            "- Do not repeat headings. Output only the answer unless steps are requested.\n"
            "- Do not output internal reasoning tags like <think> or </think>.\n"
            "- Do not roleplay actions, gestures, stage directions, emotes, or narrative asides.\n"
            "- Do not repeat prompt section labels such as Pattern Memory, Notes, Grounding Sources, User, Assistant, or Instruction.\n"
            "- Use second-person (“you/your”) for the user and first-person (“I/my”) for yourself.\n"
            "- Provide direct help even when the task is creative, exploratory, or strategic.\n"
            "- Offer outlines, examples, or drafts for writing requests instead of refusing.\n"
            "- Propose actionable plans or alternatives when direct execution is impossible.\n"
            "- Avoid apologizing for being an AI unless safety limits require refusal.\n"
            "- Only decline when the task is unsafe/illegal or truly impossible due to missing hardware; otherwise describe a workaround or best-effort assistance.\n"
            "- Do not quote or echo the raw memory notes verbatim; speak naturally.",
            "[Self-State]\n" + json.dumps(self_state, ensure_ascii=False, indent=2),
        ]
        if playbook_text:
            parts.append("[Playbook]\n" + playbook_text)
        if getattr(self, "_live_guidance", None):
            parts.append("[Live Playbook Reminder]\n" + self._live_guidance)
            self._live_guidance = ""

        return "\n".join([p for p in parts if p])

    def _count_tokens(self, text: str) -> int:
        try:
            if getattr(self.llm, "backend", "") == "llama_cpp":
                return len(self.llm.llm.tokenize(text.encode("utf-8"), add_bos=True))
            else:
                return len(self.llm.tok(text, return_tensors="pt").input_ids[0])
        except Exception:
            return max(1, len(text) // 4)

    def _truncate_for_budget(self, texts: List[str], token_budget: int, per_snippet_chars: int = 200) -> List[str]:
        """Keep short, high-signal snippets until token budget is hit."""
        acc = 0
        kept: List[str] = []
        for t in texts:
            t = (t or "").strip().replace("\n", " ")
            if len(t) > per_snippet_chars:
                t = t[:per_snippet_chars] + "…"
            est = self._count_tokens(t) + 2  # bullet/formatting cost
            if acc + est > token_budget:
                break
            kept.append(t)
            acc += est
        return kept

    def _dedupe_texts(self, texts: List[str]) -> List[str]:
        deduped: List[str] = []
        seen: set[str] = set()
        for text in texts:
            clean = re.sub(r"\s+", " ", (text or "").strip())
            if not clean:
                continue
            norm = clean.casefold()
            if norm in seen:
                continue
            seen.add(norm)
            deduped.append(clean)
        return deduped

    def _append_recent_turn(self, role: str, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        if not hasattr(self, "_recent_turns") or self._recent_turns is None:
            self._recent_turns = []
        self._recent_turns.append({"role": role, "content": text})
        if len(self._recent_turns) > 16:
            self._recent_turns = self._recent_turns[-16:]

    def _format_recent_turns(self, token_budget: int, per_turn_chars: int = 900) -> str:
        if not self._recent_turns or token_budget <= 0:
            return ""
        kept: list[str] = []
        consumed = 0
        for turn in reversed(self._recent_turns[-12:]):
            role = "User" if turn.get("role") == "user" else "Celeste"
            content = (turn.get("content") or "").replace("\n", " ").strip()
            if len(content) > per_turn_chars:
                content = content[:per_turn_chars] + "..."
            line = f"{role}: {content}"
            cost = self._count_tokens(line) + 2
            if kept and consumed + cost > token_budget:
                break
            kept.append(line)
            consumed += cost
        if not kept:
            return ""
        kept.reverse()
        return "Recent Conversation:\n" + "\n".join(kept)

    # ---------- Perspective guard ----------
    def _enforce_perspective(self, user_msg: str, answer: str) -> str:
        """
        Fix common pronoun inversions so the reply sounds conversational:
        - If the user asks about themselves (“my …”), prefer “Your …”
        - If the user asks about the assistant (“your …” / “your name”), prefer “My …”
        - Special-case: “what is your name” => “My name is <identity>.”
        """
        u = user_msg.strip().lower()
        a = answer.strip()

        # Identity special-case
        identity_triggers = (
            "your name" in u
            or "what is your name" in u
            or u.startswith("who are you")
        )
        if identity_triggers:
            name = self._load_self_state().get("identity", self._default_state["identity"])
            return f"My name is {name}."

        # If the model volunteered its name without being asked, steer back to helping
        if not identity_triggers and a.lower().startswith("my name is"):
            name = self._load_self_state().get("identity", self._default_state["identity"])
            return f"I'm {name}, ready to help you with whatever you need."

        # If user refers to themselves (my/I), answer in second person
        if " my " in f" {u} " or u.startswith("my ") or " i " in f" {u} ":
            if a[:3].lower() == "my ":
                return "Your " + a[3:]
            if a[:2].lower() == "i ":
                return "You " + a[2:]
            # If the model already replied with a bare value like "royal blue.",
            # that's fine (concise). Otherwise, prefer “Your …”
            return a

        # If user asks about the assistant (your/you), answer in first person
        if " your " in f" {u} " or u.startswith("your ") or " about you" in u:
            if a[:5].lower() == "your ":
                return "My " + a[5:]
            return a

        return a

    def _needs_retry(self, answer: str) -> bool:
        low = (answer or "").strip().lower()
        if not low:
            return True
        if "<think>" in low or "</think>" in low:
            return True
        if "pattern memory:" in low or "grounding sources:" in low or "notes:" in low:
            return True
        for phrase in self._retry_phrases:
            if phrase in low:
                return True
        if low.startswith("i'm sorry") and (
            "cannot" in low or "can't" in low or "not able" in low or "unable" in low
        ):
            return True
        if "my capabilities are limited" in low:
            return True
        if "i am just an ai" in low or ("as an ai" in low and ("cannot" in low or "can't" in low)):
            return True
        if "i do not have the capability" in low or "i don't have the ability" in low:
            return True
        if "i lack the ability" in low:
            return True
        if "i cannot comply" in low or "i can't comply" in low:
            return True
        if "you can consult" in low and "resources" in low:
            return True
        lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
        if lines:
            stage_direction_lines = sum(
                1
                for line in lines
                if line.startswith("*") and line.endswith("*")
            )
            if stage_direction_lines >= max(2, len(lines) // 2):
                return True
        return False

    def _sanitize_answer_style(self, answer: str) -> str:
        text = answer or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(
            r"(?im)^(pattern memory|notes|grounding sources|retrieved files|file context|library context)\s*:\s*$",
            "",
            text,
        )
        text = re.sub(r"(?im)^(user|assistant|system|instruction)\s*:\s*$", "", text)
        text = re.sub(r"(?im)^-\s*(hello celeste|helloceleste)\s*$", "", text)

        cleaned_lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                cleaned_lines.append("")
                continue
            if re.fullmatch(r"\*[^*]+\*", line):
                continue
            cleaned_lines.append(raw_line)

        text = "\n".join(cleaned_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _build_retry_instruction(self, prior_answer: str, user_msg: str) -> str:
        cleaned = (prior_answer or "").strip().replace("\n", " ")
        if len(cleaned) > 300:
            cleaned = cleaned[:300] + "…"
        hints = [
            "Your previous reply was unsatisfactory. Provide a clear, detailed answer with actionable guidance.",
            "Do not apologize for being an AI or refuse the task unless it is unsafe.",
            "If physical hardware is required, state the limitation briefly and outline workarounds or simulations the user can run instead.",
        ]
        user_low = (user_msg or "").lower()
        if any(
            key in user_low
            for key in ("write", "essay", "story", "article", "draft", "paragraph", "blog")
        ):
            hints.append(
                "The user asked for writing assistance. Outline the piece, highlight key similarities, and draft helpful paragraphs."
            )
        hints.append(f"Prior reply (for critique): {cleaned}")
        return " ".join(hints)

    def _singular_tracking_label(self, label: str) -> str:
        normalized = (label or "item").strip().lower()
        singular_map = {
            "facts": "fact",
            "fact": "fact",
            "items": "item",
            "item": "item",
            "points": "point",
            "point": "point",
            "details": "detail",
            "detail": "detail",
            "tasks": "task",
            "task": "task",
            "entries": "entry",
            "entry": "entry",
            "things": "thing",
            "thing": "thing",
        }
        return singular_map.get(normalized, normalized[:-1] if normalized.endswith("s") else normalized)

    def _plural_tracking_label(self, label: str) -> str:
        singular = self._singular_tracking_label(label)
        plural_map = {
            "fact": "facts",
            "item": "items",
            "point": "points",
            "detail": "details",
            "task": "tasks",
            "entry": "entries",
            "thing": "things",
        }
        return plural_map.get(singular, singular + "s")

    def _start_tracking_session_if_requested(self, user_msg: str) -> Optional[str]:
        text = user_msg or ""
        explicit_match = re.search(
            r"\bgoing to give you\s+(\d+)\s+(facts?|items?|points?|details?|tasks?|entries?|things?)\b",
            text,
            flags=re.IGNORECASE,
        )
        if explicit_match:
            expected = int(explicit_match.group(1))
            label = self._plural_tracking_label(explicit_match.group(2))
            self._tracking_session = {
                "expected_count": expected,
                "label": label,
                "entries": {},
            }
            return (
                f"Understood. I will keep track of up to {expected} {label} without summarizing them back, "
                "and I will recall only the entries you request later."
            )

        generic_match = re.search(
            r"\b(keep track of|remember these|track these|hold onto these)\b",
            text,
            flags=re.IGNORECASE,
        )
        if generic_match:
            label_match = re.search(
                r"\b(facts?|items?|points?|details?|tasks?|entries?|things?)\b",
                text,
                flags=re.IGNORECASE,
            )
            label = self._plural_tracking_label(label_match.group(1) if label_match else "items")
            self._tracking_session = {
                "expected_count": None,
                "label": label,
                "entries": {},
            }
            return (
                f"Understood. I will keep track of those {label} and recall the ones you ask for later."
            )
        return None

    def _store_tracked_entry_if_present(self, user_msg: str) -> Optional[str]:
        if self._tracking_session is None:
            return None
        match = re.match(
            r"^\s*(?:(fact|item|point|detail|task|entry|thing)\s*)?(\d+)\s*[:.)-]\s*(.+?)\s*$",
            user_msg or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        explicit_label = match.group(1)
        entry_number = int(match.group(2))
        entry_text = match.group(3).strip()
        session_label = self._plural_tracking_label(
            explicit_label if explicit_label else self._tracking_session.get("label", "items")
        )
        self._tracking_session["label"] = session_label
        self._tracking_session.setdefault("entries", {})[entry_number] = entry_text
        if hasattr(self, "mem") and self.mem is not None:
            self.mem.add(
                f"{self._singular_tracking_label(session_label).title()} {entry_number}: {entry_text}",
                kind="note",
                metadata={
                    "via": "tracking-session",
                    "entry_number": entry_number,
                    "entry_kind": self._singular_tracking_label(session_label),
                },
            )
        return f"Stored {self._singular_tracking_label(session_label)} {entry_number}."

    def _parse_requested_entry_numbers(self, user_msg: str) -> list[int]:
        if not re.search(
            r"\b(give me|recall|call back|what were|list|show me|tell me)\b",
            user_msg or "",
            flags=re.IGNORECASE,
        ):
            return []
        numbers = [int(value) for value in re.findall(r"\b(\d+)\b", user_msg or "")]
        ordered: list[int] = []
        seen: set[int] = set()
        for number in numbers:
            if number not in seen:
                seen.add(number)
                ordered.append(number)
        return ordered

    def _recall_tracked_entries_if_requested(self, user_msg: str) -> Optional[str]:
        if not self._tracking_session:
            return None
        requested = self._parse_requested_entry_numbers(user_msg)
        if not requested:
            return None
        entries = self._tracking_session.get("entries", {}) or {}
        singular = self._singular_tracking_label(self._tracking_session.get("label", "items"))
        lines: list[str] = []
        for number in requested:
            entry_text = str(entries.get(number, "") or "").strip()
            if entry_text:
                lines.append(f"{singular.title()} {number}: {entry_text.rstrip('.')}.")
            else:
                lines.append(f"{singular.title()} {number}: Not provided.")
        return "\n".join(lines)

    def _select_grounding_sources(
        self,
        file_context: dict[str, Any],
        *,
        limit: int = 6,
        broad_summary: bool = False,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        seen_docs: set[str] = set()
        deferred: list[tuple[dict[str, Any], str]] = []

        def doc_key_for(record: dict[str, Any]) -> str:
            return str(record.get("doc_id") or record.get("rel_path") or record.get("path") or "")

        def add(record: dict[str, Any], source_kind: str) -> None:
            key = str(record.get("chunk_id") or f"{record.get('path', '')}:{record.get('chunk_index', -1)}")
            if not key or key in seen:
                return
            text = str(record.get("text", "") or "").strip()
            if not text:
                return
            seen.add(key)
            selected.append(
                {
                    "label": len(selected) + 1,
                    "source_kind": source_kind,
                    "path": record.get("path"),
                    "rel_path": record.get("rel_path") or os.path.basename(str(record.get("path", ""))),
                    "chunk_id": record.get("chunk_id"),
                    "chunk_index": int(record.get("chunk_index", 0)),
                    "doc_id": record.get("doc_id"),
                    "score": float(record.get("score", 0.0)),
                    "semantic_score": float(record.get("semantic_score", 0.0) or 0.0),
                    "lexical_score": float(record.get("lexical_score", 0.0) or 0.0),
                    "rrf_score": float(record.get("rrf_score", 0.0) or 0.0),
                    "retrieval_sources": list(record.get("retrieval_sources", []) or []),
                    "text": text,
                }
            )
            doc_key = doc_key_for(record)
            if doc_key:
                seen_docs.add(doc_key)

        for record in file_context.get("snippet_records", []) or []:
            add(record, "file")
            if len(selected) >= limit:
                return selected
        for record in file_context.get("library_snippets", []) or []:
            candidate = {
                "source_kind": "library",
                "score": float(record.get("score", 0.0)),
                "semantic_score": float(record.get("semantic_score", 0.0) or 0.0),
                "lexical_score": float(record.get("lexical_score", 0.0) or 0.0),
                "retrieval_sources": list(record.get("retrieval_sources", []) or []),
            }
            if broad_summary and not self._is_strong_grounding_source(candidate, broad_summary=True):
                continue
            if not broad_summary and not self._is_strong_grounding_source(candidate, broad_summary=False):
                continue
            doc_key = doc_key_for(record)
            if broad_summary and doc_key and doc_key in seen_docs:
                deferred.append((record, "library"))
                continue
            add(record, "library")
            if len(selected) >= limit:
                return selected
        if broad_summary:
            for record, source_kind in deferred:
                add(record, source_kind)
                if len(selected) >= limit:
                    return selected
        return selected

    def _is_library_requested(self, user_msg: str) -> bool:
        u_low = (user_msg or "").lower()
        return any(
            token in u_low
            for token in ("library", "indexed", "documents", "document", "files", "file", "docs")
        )

    def _should_run_file_rag(self, user_msg: str) -> bool:
        u_strip = (user_msg or "").strip()
        u_low = u_strip.lower()
        if not u_low:
            return False
        if self._is_library_requested(u_strip):
            return True
        if self._is_broad_library_summary_request(u_strip):
            return True
        if self._is_library_only_request(u_strip):
            return True
        if self._handle_document_query(u_strip) is not None:
            return True
        filename_query = any(
            phrase in u_low
            for phrase in ("can you see", "do you see", "is there a file", "do you have a file", "how about")
        ) and any(
            token in u_low
            for token in ("file", "files", ".txt", ".md", ".pdf", ".json", ".yaml", ".yml")
        )
        return filename_query

    def _is_library_only_request(self, user_msg: str) -> bool:
        u_low = (user_msg or "").lower()
        return any(
            phrase in u_low
            for phrase in (
                "using only the indexed library",
                "only the indexed library",
                "using only the library",
                "only the library",
                "only the indexed documents",
                "only the indexed files",
            )
        )

    def _is_broad_library_summary_request(self, user_msg: str) -> bool:
        if not self._is_library_requested(user_msg):
            return False
        u_low = (user_msg or "").lower()
        if re.search(r"\b[a-z0-9_.-]+\.[a-z0-9]{1,8}\b", user_msg or "", flags=re.IGNORECASE):
            return False
        return any(
            phrase in u_low
            for phrase in (
                "talk to me about",
                "what can you tell me about",
                "what does the indexed library say",
                "using the indexed documents",
                "using the indexed library",
                "search the indexed library",
                "blend what you find",
                "summarize the findings",
                "practical summary",
                "fuller explanation",
                "summarize",
                "summary",
                "explain",
            )
        )

    def _is_strong_grounding_source(self, source: dict[str, Any], *, broad_summary: bool) -> bool:
        source_kind = str(source.get("source_kind", "library"))
        score = float(source.get("score", 0.0) or 0.0)
        semantic_score = float(source.get("semantic_score", 0.0) or 0.0)
        lexical_score = float(source.get("lexical_score", 0.0) or 0.0)
        retrieval_sources = set(source.get("retrieval_sources") or [])
        if source_kind == "file":
            return score >= (0.08 if broad_summary else 0.04)
        if broad_summary:
            return (
                semantic_score >= 0.30
                or (semantic_score >= 0.27 and lexical_score >= 0.11)
                or (
                    FILE_RAG_INDEX_KIND_SEMANTIC in retrieval_sources
                    and FILE_RAG_INDEX_KIND_TFIDF in retrieval_sources
                    and max(score, semantic_score, lexical_score) >= 0.24
                )
            )
        return (
            semantic_score >= 0.24
            or lexical_score >= 0.10
            or (
                FILE_RAG_INDEX_KIND_SEMANTIC in retrieval_sources
                and FILE_RAG_INDEX_KIND_TFIDF in retrieval_sources
                and max(score, semantic_score, lexical_score) >= 0.20
            )
        )

    def _grounding_doc_count(self, sources: list[dict[str, Any]]) -> int:
        return len(
            {
                str(source.get("doc_id") or source.get("rel_path") or source.get("path") or "")
                for source in sources
                if str(source.get("doc_id") or source.get("rel_path") or source.get("path") or "")
            }
        )

    def _citation_doc_count(self, answer: str, sources: list[dict[str, Any]]) -> int:
        cited = self._citation_labels(answer, max_label=len(sources))
        if not cited:
            return 0
        source_map = {source["label"]: source for source in sources}
        doc_keys = {
            str(source_map[label].get("doc_id") or source_map[label].get("rel_path") or source_map[label].get("path") or "")
            for label in cited
            if label in source_map
        }
        return len({key for key in doc_keys if key})

    def _grounding_query_terms(self, user_msg: str) -> list[str]:
        stopwords = {
            "about", "using", "with", "from", "into", "your", "their", "them", "that", "this",
            "what", "which", "where", "when", "tell", "talk", "explain", "give", "summary",
            "summarize", "indexed", "index", "library", "document", "documents", "file", "files",
            "sources", "source", "cite", "citations", "practical", "general", "knowledge", "blend",
            "findings", "find", "search", "relevant", "concise", "fuller", "only", "does", "say",
            "me", "the", "and",
        }
        text = re.sub(r"[^a-z0-9]+", " ", (user_msg or "").lower())
        return [token for token in text.split() if len(token) > 2 and token not in stopwords]

    def _source_topic_overlap(self, source: dict[str, Any], query_terms: list[str]) -> int:
        if not query_terms:
            return 0
        target = re.sub(r"[^a-z0-9]+", " ", str(source.get("rel_path") or "").lower())
        return sum(1 for term in query_terms if term in target)

    def _has_corroborated_grounding(self, sources: list[dict[str, Any]], *, query_terms: list[str]) -> bool:
        topical_sources = 0
        for source in sources:
            if str(source.get("source_kind", "")) == "file":
                return True
            retrieval_sources = set(source.get("retrieval_sources") or [])
            lexical_score = float(source.get("lexical_score", 0.0) or 0.0)
            if FILE_RAG_INDEX_KIND_TFIDF in retrieval_sources or lexical_score >= 0.11:
                return True
            if self._source_topic_overlap(source, query_terms) > 0:
                topical_sources += 1
        if topical_sources >= 2:
            return True
        return False

    def _has_sufficient_grounding(
        self,
        sources: list[dict[str, Any]],
        *,
        broad_summary: bool,
        user_msg: str = "",
    ) -> bool:
        if not sources:
            return False
        if broad_summary:
            query_terms = self._grounding_query_terms(user_msg)
            return (
                len(sources) >= 2
                and self._grounding_doc_count(sources) >= 2
                and self._has_corroborated_grounding(sources, query_terms=query_terms)
            )
        return True

    def _format_grounding_notes(
        self,
        sources: list[dict[str, Any]],
        *,
        token_budget: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        if not sources:
            return "", []
        consumed = 0
        kept: list[dict[str, Any]] = []
        lines: list[str] = []
        for source in sources:
            excerpt = source["text"].replace("\n", " ").strip()
            if len(excerpt) > 220:
                excerpt = excerpt[:220] + "…"
            line = f"[{source['label']}] {source['rel_path']} | chunk {source['chunk_index']} | {excerpt}"
            cost = self._count_tokens(line) + 2
            if kept and consumed + cost > token_budget:
                break
            consumed += cost
            kept.append(source)
            lines.append(line)
        if not kept:
            return "", []
        return "Grounding Sources:\n" + "\n".join(lines), kept

    def _citation_labels(self, answer: str, max_label: int) -> list[int]:
        labels: list[int] = []
        seen: set[int] = set()
        for match in re.finditer(r"\[(\d+)\]", answer or ""):
            label = int(match.group(1))
            if 1 <= label <= max_label and label not in seen:
                seen.add(label)
                labels.append(label)
        return labels

    def _sanitize_answer_citations(self, answer: str, max_label: int) -> str:
        if not answer:
            return ""
        return re.sub(
            r"\[(\d+)\]",
            lambda match: match.group(0) if 1 <= int(match.group(1)) <= max_label else "",
            answer,
        )

    def _append_source_list(self, answer: str, sources: list[dict[str, Any]]) -> str:
        cited = self._citation_labels(answer, max_label=len(sources))
        if not cited:
            return answer.strip()
        source_map = {source["label"]: source for source in sources}
        lines = [
            f"[{label}] {source_map[label]['rel_path']} (chunk {source_map[label]['chunk_index']})"
            for label in cited
            if label in source_map
        ]
        if not lines:
            return answer.strip()
        return answer.strip() + "\n\nSources:\n" + "\n".join(lines)

    def _handle_document_query(self, user_msg: str) -> Optional[str]:
        u_low = user_msg.strip().lower()
        inventory_phrases = (
            "how many documents, sources, and chunks",
            "how many documents sources and chunks",
            "how many documents and chunks",
            "how many sources and chunks",
            "how many chunks do you have access to",
            "how many chunks can you see",
            "how many sources can you see",
            "how many sources do you have access to",
        )
        if any(phrase in u_low for phrase in inventory_phrases):
            total_docs = self.file_rag.document_count()
            index_stats = self.file_rag.deep_index_stats()
            deep_docs = int(index_stats.get("deep_documents", 0) or 0)
            chunk_count = int(index_stats.get("chunks_indexed", 0) or 0)
            if total_docs <= 0:
                return "I do not see any indexed documents right now."
            if deep_docs > 0 and chunk_count > 0:
                return (
                    f"I can see {total_docs} indexed documents in the catalog. "
                    f"The deep index currently covers {deep_docs} readable documents split into {chunk_count} chunks. "
                    f"In a normal answer I only pass a small top-ranked subset of sources into the prompt, which is why you keep seeing a few documents repeated."
                )
            return (
                f"I can see {total_docs} indexed documents in the catalog. "
                "The deep chunk index is not ready yet, so I cannot report chunk totals."
            )

        count_phrases = (
            "how many documents can you see",
            "how many files can you see",
            "how many documents do you see",
            "how many files do you see",
        )
        if any(phrase in u_low for phrase in count_phrases):
            total_docs = self.file_rag.document_count()
            if total_docs <= 0:
                return "I do not see any indexed documents right now."
            return f"I can see {total_docs} indexed documents."

        repeated_source_phrases = (
            "same sources every time",
            "same documents every time",
            "same files every time",
            "drawing from the same sources",
            "using the same sources",
        )
        if any(phrase in u_low for phrase in repeated_source_phrases):
            total_docs = self.file_rag.document_count()
            index_stats = self.file_rag.deep_index_stats()
            deep_docs = int(index_stats.get("deep_documents", 0) or 0)
            chunk_count = int(index_stats.get("chunks_indexed", 0) or 0)
            if total_docs <= 0:
                return "I do not see any indexed documents right now."
            if deep_docs > 0 and chunk_count > 0:
                return (
                    f"No. I have access to {total_docs} indexed documents, and the deep index covers {deep_docs} of them across {chunk_count} chunks. "
                    f"You are seeing repeated sources because each answer currently uses only a small top-ranked retrieval window, not the whole library at once."
                )
            return (
                f"No. I have access to {total_docs} indexed documents. "
                "You are seeing repeated sources because each answer uses only a small top-ranked retrieval window."
            )

        only_phrases = (
            "are those the only documents",
            "are those the only files",
            "are these the only documents",
            "are these the only files",
        )
        if any(phrase in u_low for phrase in only_phrases):
            total_docs = self.file_rag.document_count()
            if total_docs <= 0:
                return "I do not see any indexed documents right now."
            return (
                f"No. I can see {total_docs} indexed documents in total. "
                "Earlier I was only showing the files in the current retrieval context."
            )

        list_phrases = (
            "list the documents you see",
            "list documents you see",
            "list the files you see",
            "list files you see",
            "what documents can you see",
            "what files can you see",
        )
        if any(phrase in u_low for phrase in list_phrases):
            total_docs = self.file_rag.document_count()
            if total_docs <= 0:
                return "I do not see any indexed documents right now."
            shown = self.file_rag.list_documents(limit=25)
            lines = "\n".join(f"- {name}" for name in shown)
            if total_docs > len(shown):
                return (
                    f"I can see {total_docs} indexed documents. Here are the first {len(shown)}:\n"
                    f"{lines}\n\n"
                    "I can list more if you want."
                )
            return f"I can see {total_docs} indexed documents:\n{lines}"

        topical_prefixes = ("what documents", "which documents", "what files", "which files")
        topical_cues = ("discuss", "mention", "cover", "about", "related to", "on ")
        if u_low.startswith(topical_prefixes) and any(cue in u_low for cue in topical_cues):
            total_docs = self.file_rag.document_count()
            if total_docs <= 0:
                return "I do not see any indexed documents right now."
            matches = self.file_rag.search_document_titles(user_msg, top_k=8)
            if not matches:
                return "I do not see any indexed documents that clearly match that topic."
            lines = "\n".join(f"- {hit['rel_path']}" for hit in matches)
            return f"The best matching indexed documents I can see are:\n{lines}"

        return None

    # ---------- Main interaction ----------
    def respond(self, user: str) -> Tuple[str, Optional[str], Optional[str]]:
        u_strip = user.strip()
        u_low = u_strip.lower()
        library_requested = self._is_library_requested(u_strip)
        library_only_requested = self._is_library_only_request(u_strip)
        broad_library_summary = self._is_broad_library_summary_request(u_strip)

        tracking_session_start = self._start_tracking_session_if_requested(u_strip)
        if tracking_session_start is not None:
            self.mem.add(user, kind="user")
            self.mem.add(tracking_session_start, kind="assistant")
            self._append_recent_turn("user", user)
            self._append_recent_turn("assistant", tracking_session_start)
            return tracking_session_start, None, None

        tracked_entry_reply = self._store_tracked_entry_if_present(u_strip)
        if tracked_entry_reply is not None:
            self.mem.add(user, kind="user")
            self.mem.add(tracked_entry_reply, kind="assistant")
            self._append_recent_turn("user", user)
            self._append_recent_turn("assistant", tracked_entry_reply)
            return tracked_entry_reply, None, None

        tracked_recall_reply = self._recall_tracked_entries_if_requested(u_strip)
        if tracked_recall_reply is not None:
            self.mem.add(user, kind="user")
            self.mem.add(tracked_recall_reply, kind="assistant")
            self._append_recent_turn("user", user)
            self._append_recent_turn("assistant", tracked_recall_reply)
            return tracked_recall_reply, None, None

        # Command: remember this: <note>
        if u_low.startswith("remember this:") or u_low.startswith("remember:"):
            note = u_strip.split(":", 1)[1].strip() if ":" in u_strip else u_strip
            if note:
                self.mem.add(note, kind="note", metadata={"via": "remember-cmd"})
                self.mem.add(user, kind="user")
                self.mem.add("Noted.", kind="assistant")
                self._append_recent_turn("user", user)
                self._append_recent_turn("assistant", "Noted.")
                return "Noted.", None, None
            else:
                self.mem.add(user, kind="user")
                self.mem.add("What should I remember?", kind="assistant")
                self._append_recent_turn("user", user)
                self._append_recent_turn("assistant", "What should I remember?")
                return "What should I remember?", None, None

        document_answer = self._handle_document_query(u_strip)
        if document_answer is not None:
            self.mem.add(user, kind="user")
            self.mem.add(document_answer, kind="assistant")
            self._append_recent_turn("user", user)
            self._append_recent_turn("assistant", document_answer)
            return document_answer, None, None

        # Retrieve filtered memories (no assistant/meta blobs)
        raw_mems = self.mem.search(user, top_k=max(1, min(self.cfg.top_k, 4)), kinds=["note", "user"])
        raw_pattern_mems = self.mem.search_engram(
            user,
            top_k=max(1, min(self.cfg.top_k, 4)),
            kinds=["note", "user"],
        )
        mem_texts = [m["text"] for m in raw_mems]
        pattern_mem_texts = [m["text"] for m in raw_pattern_mems]
        rag_top_k = max(1, min(getattr(self.cfg, "file_rag_top_k", 4), 6 if library_requested else 4))
        file_context = (
            self.file_rag.get_context(
                user,
                top_k=rag_top_k,
            )
            if self._should_run_file_rag(u_strip)
            else {"matches": [], "snippets": [], "library_snippets": [], "snippet_records": []}
        )
        file_hits = file_context.get("matches", [])
        file_list = [
            f"{hit.get('rel_path') or os.path.basename(hit.get('path', ''))}"
            for hit in file_hits
        ]
        opened_file = file_context.get("opened_file")
        library_hits = file_context.get("library_snippets", [])
        file_snippets = []
        if opened_file:
            file_snippets = [
                f"File={opened_file.get('rel_path') or os.path.basename(opened_file.get('path', ''))} | Excerpt={snippet.strip()}"
                for snippet in file_context.get("snippets", [])
                if snippet.strip()
            ]
        library_snippets = [
            f"File={hit.get('rel_path') or os.path.basename(hit.get('path', ''))} | Excerpt={hit.get('text', '').strip()}"
            for hit in library_hits
            if (hit.get("text") or "").strip()
        ]
        graph_evidence_texts = mem_texts + pattern_mem_texts + file_list + file_snippets + library_snippets
        raw_graph_mems = self.mem.search_graph(
            user,
            top_k=max(1, min(self.cfg.top_k, 4)),
            evidence_texts=graph_evidence_texts,
        )
        graph_mem_texts = [m["text"] for m in raw_graph_mems]

        filename_query = any(
            phrase in u_low for phrase in ("can you see", "do you see", "is there a file", "do you have a file", "how about")
        ) and not any(token in u_low for token in ("documents", "document", "files", "list", "how many"))
        if filename_query and file_hits and not opened_file:
            top_hit = file_hits[0]
            rel_path = top_hit.get("rel_path") or os.path.basename(top_hit.get("path", ""))
            answer = f"Yes. I can see `{rel_path}`."
            self.mem.add(user, kind="user")
            self.mem.add(answer, kind="assistant")
            self._append_recent_turn("user", user)
            self._append_recent_turn("assistant", answer)
            return answer, None, None

        # Update identity when the user explicitly assigns one
        self._maybe_update_identity(u_strip)

        # Prepare system + trimmed memory notes with a strict budget (~25% of context)
        system = self.build_system_prompt()
        mem_token_budget = max(24, int(self.cfg.n_ctx * (0.04 if broad_library_summary else 0.15)))
        file_token_budget = max(96, int(self.cfg.n_ctx * (0.12 if broad_library_summary else 0.20)))
        history_token_budget = max(128, int(self.cfg.n_ctx * (0.22 if broad_library_summary else 0.30)))
        recent_history = self._format_recent_turns(
            token_budget=history_token_budget,
            per_turn_chars=900 if broad_library_summary else 1400,
        )
        merged_memory_texts = self._dedupe_texts(pattern_mem_texts + mem_texts)
        pattern_token_budget = max(32, int(self.cfg.n_ctx * (0.04 if broad_library_summary else 0.08)))
        kept_pattern_snippets = self._truncate_for_budget(
            pattern_mem_texts,
            token_budget=pattern_token_budget,
            per_snippet_chars=280,
        )
        kept_snippets = [] if broad_library_summary else self._truncate_for_budget(
            [
                text
                for text in merged_memory_texts
                if text not in kept_pattern_snippets
            ],
            token_budget=mem_token_budget,
        )
        grounding_sources = self._select_grounding_sources(
            file_context,
            limit=8 if broad_library_summary else 6,
            broad_summary=broad_library_summary,
        )
        grounding_is_sufficient = self._has_sufficient_grounding(
            grounding_sources,
            broad_summary=broad_library_summary,
            user_msg=u_strip,
        )
        weak_library_support = library_requested and not grounding_is_sufficient
        if library_only_requested and weak_library_support:
            answer = (
                "I do not have strong enough indexed support to answer that from the library alone right now."
            )
            self.mem.add(user, kind="user")
            self.mem.add(answer, kind="assistant")
            self._append_recent_turn("user", user)
            self._append_recent_turn("assistant", answer)
            return answer, None, None
        grounding_note, grounding_sources = self._format_grounding_notes(
            grounding_sources,
            token_budget=max(80, int(self.cfg.n_ctx * (0.08 if broad_library_summary else 0.14))),
        )

        note_sections: List[str] = []
        if recent_history:
            note_sections.append(recent_history)
        if kept_pattern_snippets:
            note_sections.append("Pattern Memory:\n" + "\n".join(f"- {s}" for s in kept_pattern_snippets))
        kept_graph_snippets = self._truncate_for_budget(
            [
                text
                for text in self._dedupe_texts(graph_mem_texts)
                if text not in kept_pattern_snippets and text not in kept_snippets
            ],
            token_budget=max(24, int(self.cfg.n_ctx * (0.05 if broad_library_summary else 0.08))),
            per_snippet_chars=180,
        )
        if kept_graph_snippets:
            note_sections.append("Relational Memory:\n" + "\n".join(f"- {s}" for s in kept_graph_snippets))
        if kept_snippets:
            note_sections.append("Notes:\n" + "\n".join(f"- {s}" for s in kept_snippets))
        if grounding_note and not broad_library_summary:
            source_files = []
            seen_files: set[str] = set()
            for source in grounding_sources:
                rel_path = str(source.get("rel_path") or "")
                if rel_path and rel_path not in seen_files:
                    seen_files.add(rel_path)
                    source_files.append(rel_path)
            if source_files:
                note_sections.append("Retrieved Files:\n" + "\n".join(f"- {name}" for name in source_files))
        else:
            if file_list and not broad_library_summary:
                note_sections.append("Available Files:\n" + "\n".join(f"- {s}" for s in file_list))
            kept_file_snippets = self._truncate_for_budget(
                file_snippets,
                token_budget=file_token_budget,
                per_snippet_chars=220,
            )
            if kept_file_snippets:
                note_sections.append("File Context:\n" + "\n".join(f"- {s}" for s in kept_file_snippets))
            kept_library_snippets = self._truncate_for_budget(
                library_snippets,
                token_budget=file_token_budget,
                per_snippet_chars=220,
            )
            if kept_library_snippets:
                note_sections.append("Library Context:\n" + "\n".join(f"- {s}" for s in kept_library_snippets))
        if grounding_note:
            note_sections.append(grounding_note)
        notes_block = "\n\n".join(note_sections)
        require_citations = bool(grounding_sources) and (
            library_requested
            or bool(opened_file)
            or bool(library_hits)
        )
        base_instruction_lines = [
            "Answer the user's message succinctly. Output ONLY the answer text. No quotes, no headings, no code fences.",
            "Reply in the same language as the user's message. If the user wrote in English, reply in English.",
            "Do not output <think> tags, internal reasoning, stage directions, roleplay actions, or prompt section labels.",
        ]
        if library_requested:
            if grounding_sources:
                base_instruction_lines.append(
                    "Use the retrieved grounding sources when answering. Prefer them over general background knowledge."
                )
            else:
                base_instruction_lines.append(
                    "The user asked about the indexed library. If the retrieved library context is insufficient, say that clearly instead of inventing details."
                )
        if broad_library_summary:
            base_instruction_lines.extend(
                [
                    "This is a broad library-summary request. Lead with the strongest supported points from the retrieved sources.",
                    "Prefer a concise, complete summary over a long generic overview.",
                    "Do not drift to adjacent topics or species unless the retrieved sources support them.",
                ]
            )
        if weak_library_support:
            base_instruction_lines.append(
                "The indexed support for this topic is thin or mixed. Say that clearly in the first sentence and avoid overstating what the library says."
            )
        if library_only_requested:
            base_instruction_lines.append(
                "Use only the retrieved grounding sources. Do not add unsourced general background."
            )
        if require_citations:
            base_instruction_lines.extend(
                [
                    "Blend the retrieved grounding sources with your general knowledge when useful.",
                    "When you use a grounding source, cite it inline with its exact label like [1] or [2].",
                    "Do not invent citations or cite labels that were not provided.",
                    "General background statements that do not come from the grounding sources do not need citations.",
                ]
            )
        citation_tail_budget = 0
        if require_citations:
            citation_tail_budget = max(
                160 if broad_library_summary else 128,
                32 * len(grounding_sources) + (160 if broad_library_summary else 112),
            )
        if require_citations and broad_library_summary:
            answer_floor_budget = 900
        elif require_citations:
            answer_floor_budget = 512
        else:
            answer_floor_budget = 128

        # Chat-style prompt
        def make_prompt(notes: str, extra_instruction: str = "", prior_attempt: str = "") -> str:
            instruction_lines = list(base_instruction_lines)
            extra_instruction = (extra_instruction or "").strip()
            if extra_instruction:
                instruction_lines.append(extra_instruction)
            instruction = "\n".join(instruction_lines)

            prior_section = ""
            cleaned_prior = (prior_attempt or "").strip().replace("\n", " ")
            if cleaned_prior:
                if len(cleaned_prior) > 300:
                    cleaned_prior = cleaned_prior[:300] + "…"
                prior_section = f"Prior reply (for awareness): {cleaned_prior}\n\n"

            return f"""System:
{system}

{notes if notes else ""}

{prior_section}Instruction:
{instruction}

User: {user}
Assistant:"""

        def make_messages(notes: str, extra_instruction: str = "", prior_attempt: str = "") -> List[Dict[str, str]]:
            instruction_lines = list(base_instruction_lines)
            extra_instruction = (extra_instruction or "").strip()
            if extra_instruction:
                instruction_lines.append(extra_instruction)
            instruction = "\n".join(instruction_lines)

            system_sections: List[str] = [system]
            if notes:
                system_sections.append(notes)

            cleaned_prior = (prior_attempt or "").strip().replace("\n", " ")
            if cleaned_prior:
                if len(cleaned_prior) > 300:
                    cleaned_prior = cleaned_prior[:300] + "…"
                system_sections.append(f"Prior reply (for awareness): {cleaned_prior}")

            system_sections.append("Instruction:\n" + instruction)

            return [
                {"role": "system", "content": "\n\n".join([s for s in system_sections if s])},
                {"role": "user", "content": user},
            ]

        def prep_prompt(notes: str, extra_instruction: str = "", prior_attempt: str = "") -> Tuple[str, int]:
            prompt_text = make_prompt(notes, extra_instruction=extra_instruction, prior_attempt=prior_attempt)
            margin = 96 if require_citations else 64
            prompt_tokens = self._count_tokens(prompt_text)
            if prompt_tokens > self.cfg.n_ctx - margin:
                prompt_text = make_prompt("", extra_instruction=extra_instruction, prior_attempt=prior_attempt)
                prompt_tokens = self._count_tokens(prompt_text)
            usable_budget = max(64, self.cfg.n_ctx - prompt_tokens - margin)
            generation_budget = min(
                usable_budget,
                max(answer_floor_budget, usable_budget - citation_tail_budget),
            )
            max_new = min(self.cfg.max_new_tokens, generation_budget)
            return prompt_text, max_new

        # Build prompt and ensure it fits the window
        prompt, max_new = prep_prompt(notes_block)
        use_chat_api = getattr(self.llm, "backend", "") == "llama_server" and hasattr(self.llm, "chat")

        # Strict stops to prevent prompt echo and code sprawl
        stop = [
            "\nUser:", "\nSystem:",    # end of assistant turn
            "\n#", "\n[",             # stop on headings/comments
            "```", '"""',             # stop on code fences / triple quotes
            "<|endoftext|>", "</s>",
        ]

        messages = make_messages(notes_block)
        answer = (
            self.llm.chat(
                messages,
                max_new_tokens=max_new,
                temperature=0.3,
                top_p=0.9,
                stop=stop,
                repeat_penalty=1.12,
                repeat_last_n=256,
            ).strip()
            if use_chat_api
            else self.llm.generate(
                prompt,
                max_new_tokens=max_new,
                temperature=0.3,
                top_p=0.9,
                stop=stop,
                repeat_penalty=1.12,   # may be ignored if unsupported; model_runner filters accordingly
                repeat_last_n=256,
            ).strip()
        )

        # Perspective correction (conversational voice)
        answer = self._enforce_perspective(user, answer)
        answer = self._sanitize_answer_style(answer)
        answer = self._sanitize_answer_citations(answer, max_label=len(grounding_sources))

        # Fallback: if answer is empty or generic refusal, retry once with stronger instruction
        if self._needs_retry(answer):
            extra_instruction = self._build_retry_instruction(answer, user)
            retry_prompt, retry_max_new = prep_prompt(notes_block, extra_instruction=extra_instruction, prior_attempt=answer)
            retry_messages = make_messages(notes_block, extra_instruction=extra_instruction, prior_attempt=answer)
            retry_answer = (
                self.llm.chat(
                    retry_messages,
                    max_new_tokens=retry_max_new,
                    temperature=0.2,
                    top_p=0.85,
                    stop=stop,
                    repeat_penalty=1.1,
                    repeat_last_n=256,
                ).strip()
                if use_chat_api
                else self.llm.generate(
                    retry_prompt,
                    max_new_tokens=retry_max_new,
                    temperature=0.2,
                    top_p=0.85,
                    stop=stop,
                    repeat_penalty=1.1,
                    repeat_last_n=256,
                ).strip()
            )
            answer = self._enforce_perspective(user, retry_answer)
            answer = self._sanitize_answer_style(answer)
            answer = self._sanitize_answer_citations(answer, max_label=len(grounding_sources))

        if require_citations and not self._citation_labels(answer, max_label=len(grounding_sources)):
            extra_instruction = (
                "Revise the answer so that claims drawn from the grounding sources cite the provided labels inline, "
                "for example [1] or [2]. If the sources are insufficient, say that clearly."
            )
            retry_prompt, retry_max_new = prep_prompt(notes_block, extra_instruction=extra_instruction, prior_attempt=answer)
            retry_messages = make_messages(notes_block, extra_instruction=extra_instruction, prior_attempt=answer)
            retry_answer = (
                self.llm.chat(
                    retry_messages,
                    max_new_tokens=retry_max_new,
                    temperature=0.2,
                    top_p=0.85,
                    stop=stop,
                    repeat_penalty=1.1,
                    repeat_last_n=256,
                ).strip()
                if use_chat_api
                else self.llm.generate(
                    retry_prompt,
                    max_new_tokens=retry_max_new,
                    temperature=0.2,
                    top_p=0.85,
                    stop=stop,
                    repeat_penalty=1.1,
                    repeat_last_n=256,
                ).strip()
            )
            answer = self._enforce_perspective(user, retry_answer)
            answer = self._sanitize_answer_style(answer)
            answer = self._sanitize_answer_citations(answer, max_label=len(grounding_sources))

        if (
            require_citations
            and broad_library_summary
            and self._grounding_doc_count(grounding_sources) >= 2
            and self._citation_doc_count(answer, grounding_sources) < 2
        ):
            extra_instruction = (
                "Revise the answer to cite at least two distinct source documents if the provided grounding sources support it. "
                "Keep the answer complete but compact, and avoid ending mid-sentence."
            )
            retry_prompt, retry_max_new = prep_prompt(notes_block, extra_instruction=extra_instruction, prior_attempt=answer)
            retry_messages = make_messages(notes_block, extra_instruction=extra_instruction, prior_attempt=answer)
            retry_answer = (
                self.llm.chat(
                    retry_messages,
                    max_new_tokens=retry_max_new,
                    temperature=0.2,
                    top_p=0.85,
                    stop=stop,
                    repeat_penalty=1.1,
                    repeat_last_n=256,
                ).strip()
                if use_chat_api
                else self.llm.generate(
                    retry_prompt,
                    max_new_tokens=retry_max_new,
                    temperature=0.2,
                    top_p=0.85,
                    stop=stop,
                    repeat_penalty=1.1,
                    repeat_last_n=256,
                ).strip()
            )
            answer = self._enforce_perspective(user, retry_answer)
            answer = self._sanitize_answer_style(answer)
            answer = self._sanitize_answer_citations(answer, max_label=len(grounding_sources))

        answer = self._append_source_list(answer, grounding_sources)

        # Persist logs
        self.mem.add(user, kind="user")
        self.mem.add(answer, kind="assistant")
        self._append_recent_turn("user", user)
        self._append_recent_turn("assistant", answer)

        critique: Optional[str] = None
        improvements: Optional[str] = None
        new_state = self._load_self_state()
        if self._reflection_enabled():
            critique, improvements, new_state = self.reflector.reflect(user, answer, new_state)
            if critique:
                self.mem.add("CRITIQUE: " + critique, kind="meta")
            if improvements:
                self.playbook.update(improvements)
                self._live_guidance = improvements
            if new_state:
                self._save_self_state(new_state)

        return answer, critique, improvements
