#!/usr/bin/env python3
from __future__ import annotations

import html
import logging
import multiprocessing as mp
import os
import re
import sys
import threading
import traceback
from collections import deque

try:
    from PySide6.QtCore import QObject, QMetaObject, QThread, QTimer, Qt, Signal, Slot
    from PySide6.QtGui import QFont, QIcon, QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QProgressBar,
        QProgressDialog,
        QScrollArea,
        QSpinBox,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "PySide6 is not installed. Install it in the Celeste venv with:\n"
        "  pip install PySide6"
    ) from exc

from app_service import CelesteService
from config_types import AgentConfig
from app_paths import default_config_path, resource_path


_LOG_BUFFER: deque[str] = deque(maxlen=1500)
_LOG_EMITTER = None


class LogEmitter(QObject):
    message = Signal(str)


class QtSignalLogHandler(logging.Handler):
    def __init__(self, emitter: LogEmitter):
        super().__init__(level=logging.INFO)
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            rendered = self.format(record)
        except Exception:
            rendered = record.getMessage()
        _LOG_BUFFER.append(rendered)
        try:
            self.emitter.message.emit(rendered)
        except Exception:
            pass


def _get_log_emitter() -> LogEmitter:
    global _LOG_EMITTER
    if _LOG_EMITTER is None:
        _LOG_EMITTER = LogEmitter()
    return _LOG_EMITTER


class LiveLogDialog(QDialog):
    closed = Signal()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.closed.emit()
        super().closeEvent(event)


def _setup_app_logging(config_path: str) -> str:
    config_dir = os.path.dirname(os.path.abspath(config_path))
    os.makedirs(config_dir, exist_ok=True)
    log_path = os.path.join(config_dir, "celeste_desktop.log")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    qt_handler = QtSignalLogHandler(_get_log_emitter())
    qt_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler, qt_handler],
        force=True,
    )
    return log_path


def _install_exception_logging() -> None:
    def _log_unhandled(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            return
        logging.critical(
            "Unhandled exception:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )

    sys.excepthook = _log_unhandled


class RulebookDialog(QDialog):
    """Modal dialog for reviewing and editing the behavioral rulebook."""

    def __init__(self, rules: list, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Rulebook")
        self.setMinimumSize(660, 460)
        self.resize(720, 540)
        self._rules = rules
        self._pending_deletes: set[int] = set()
        self._pending_updates: dict[int, str] = {}
        self._editors: dict[int, QPlainTextEdit] = {}
        self._frames: dict[int, QFrame] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QLabel(
            "Review and edit the behavioral rulebook.\n"
            "[user] rules are protected from automatic pruning.  "
            "[teacher] rules are managed automatically."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(8)
        inner_layout.setContentsMargins(0, 0, 0, 0)

        if not self._rules:
            empty = QLabel("No rules yet. Use  rule: <text>  in chat to add a rule.")
            empty.setWordWrap(True)
            inner_layout.addWidget(empty)
        else:
            for rule in self._rules:
                rule_id = int(rule.get("id", -1))
                source = str(rule.get("source", "teacher"))
                text = str(rule.get("text", ""))

                frame = QFrame()
                frame.setObjectName("settingsCard")
                frame.setFrameShape(QFrame.StyledPanel)
                self._frames[rule_id] = frame
                row = QHBoxLayout(frame)
                row.setContentsMargins(10, 8, 10, 8)
                row.setSpacing(10)

                tag = QLabel(f"[{source}]")
                tag.setFixedWidth(62)
                tag.setStyleSheet(
                    "color: #a8ffcf; font-weight: bold;"
                    if source == "user"
                    else "color: #9fd3ff;"
                )

                editor = QPlainTextEdit(text)
                editor.setMaximumHeight(62)
                editor.setMinimumHeight(42)
                self._editors[rule_id] = editor

                del_btn = QPushButton("Delete")
                del_btn.setFixedWidth(58)
                del_btn.clicked.connect(
                    lambda _checked, rid=rule_id: self._mark_delete(rid)
                )

                row.addWidget(tag)
                row.addWidget(editor, 1)
                row.addWidget(del_btn)
                inner_layout.addWidget(frame)

        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        save_btn = QPushButton("Save Changes")
        close_btn = QPushButton("Close")
        save_btn.setMinimumHeight(30)
        close_btn.setMinimumHeight(30)
        save_btn.clicked.connect(self._collect_and_accept)
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _mark_delete(self, rule_id: int) -> None:
        self._pending_deletes.add(rule_id)
        frame = self._frames.get(rule_id)
        if frame:
            frame.setEnabled(False)
            frame.setStyleSheet("background: #3a1515; border-radius: 8px;")

    def _collect_and_accept(self) -> None:
        for rule_id, editor in self._editors.items():
            if rule_id in self._pending_deletes:
                continue
            new_text = editor.toPlainText().strip()
            orig = next((r["text"] for r in self._rules if r.get("id") == rule_id), "")
            if new_text and new_text != orig:
                self._pending_updates[rule_id] = new_text
        self.accept()

    def get_deletes(self) -> set[int]:
        return self._pending_deletes

    def get_updates(self) -> dict[int, str]:
        return self._pending_updates


class PersonaDialog(QDialog):
    """Modal dialog for editing the system preamble / persona."""

    def __init__(self, current_preamble: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Persona")
        self.resize(520, 300)
        self._preamble = current_preamble
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        hint = QLabel(
            "Define the assistant's identity, name, and behavioral baseline. "
            "This text is prepended to every prompt — changes take effect immediately."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._editor = QPlainTextEdit(current_preamble)
        self._editor.setPlaceholderText(
            "e.g. Your name is Aria. You are a personal research assistant for Shane. "
            "Never claim to be the user or adopt the user's name or identity."
        )
        layout.addWidget(self._editor, 1)

        buttons = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        buttons.addStretch()
        buttons.addWidget(cancel_btn)
        buttons.addWidget(save_btn)
        layout.addLayout(buttons)

        save_btn.clicked.connect(self._save)
        cancel_btn.clicked.connect(self.reject)

    def _save(self) -> None:
        self._preamble = self._editor.toPlainText().strip()
        self.accept()

    def get_preamble(self) -> str:
        return self._preamble


class ServiceWorker(QObject):
    initialized = Signal(object)
    models_ready = Signal(list)
    reloaded = Signal(object)
    rag_updated = Signal(object, str)
    memory_updated = Signal(object, str)
    reply_ready = Signal(str, str, str)
    chat_finished = Signal(str)
    response_stopped = Signal(str)
    token_received = Signal(str)
    token_usage = Signal(int, int)
    failed = Signal(str)
    status = Signal(str)
    progress = Signal(int, str)
    rulebook_flagged = Signal(str)

    def __init__(self, config_path: str):
        super().__init__()
        self.service = CelesteService(config_path)
        self._chat_thread: threading.Thread | None = None
        self._reload_thread: threading.Thread | None = None
        self._deep_index_thread: threading.Thread | None = None
        self._chat_cancel_requested = False

    @Slot()
    def initialize(self) -> None:
        try:
            logging.info("Worker initialize requested.")
            self.status.emit("Starting Celeste backend...")
            cfg = self.service.start(status_cb=self.status.emit)
            self.service.set_reflection_flag_cb(lambda reason: self.rulebook_flagged.emit(reason))
            self.initialized.emit(cfg)
            self.status.emit("Celeste ready.")
        except Exception as exc:
            logging.exception("Celeste backend initialization failed")
            self.failed.emit(str(exc))

    @Slot()
    def load_models(self) -> None:
        try:
            logging.info("Worker model discovery requested.")
            self.models_ready.emit(self.service.available_models())
        except Exception as exc:
            logging.exception("Model discovery failed")
            self.failed.emit(str(exc))

    @Slot(str)
    def send_message(self, message: str) -> None:
        if self._chat_thread is not None and self._chat_thread.is_alive():
            self.failed.emit("A chat request is already in progress.")
            return

        logging.info("Worker chat requested (chars=%s).", len(message or ""))
        self.status.emit("Generating reply...")
        self._chat_cancel_requested = False

        def _run_chat() -> None:
            try:
                logging.info("Worker calling service.chat() from Python thread.")
                answer, critique, improvements = self.service.chat(
                    message,
                    token_cb=lambda tok: self.token_received.emit(tok),
                )
                if self._chat_cancel_requested:
                    logging.info("Worker chat result discarded because stop was requested.")
                    self.response_stopped.emit("Response stopped.")
                    return
                logging.info(
                    "Worker service.chat() returned (answer_chars=%s, critique_chars=%s, improvements_chars=%s).",
                    len(answer or ""),
                    len(critique or ""),
                    len(improvements or ""),
                )
                self.reply_ready.emit(answer, critique or "", improvements or "")
                used, ctx = self.service.get_token_usage()
                self.token_usage.emit(used, ctx)
                logging.info("Worker emitted reply_ready.")
                if answer.strip() and not self._chat_cancel_requested:
                    self.status.emit("Speaking reply...")
                    logging.info("Worker calling service.speak() from Python thread.")
                    self.service.speak(answer)
                    logging.info("Worker service.speak() returned.")
                if self._chat_cancel_requested:
                    logging.info("Worker speech interrupted because stop was requested.")
                    self.response_stopped.emit("Response stopped.")
                    return
                self.status.emit("Ready.")
                logging.info("Worker marked chat request ready.")
                self.chat_finished.emit("Ready.")
            except Exception as exc:
                if self._chat_cancel_requested:
                    logging.info("Worker chat interrupted by stop request.")
                    self.response_stopped.emit("Response stopped.")
                    return
                logging.exception("Chat request failed")
                self.failed.emit(str(exc))
            finally:
                self._chat_thread = None
                self._chat_cancel_requested = False

        self._chat_thread = threading.Thread(
            target=_run_chat,
            name="celeste-chat-worker",
            daemon=True,
        )
        self._chat_thread.start()

    @Slot()
    def stop_response(self) -> None:
        if self._chat_thread is None or not self._chat_thread.is_alive():
            self.response_stopped.emit("No response is currently active.")
            return
        logging.info("Worker stop-response requested.")
        self._chat_cancel_requested = True
        self.status.emit("Stopping response...")
        try:
            self.service.shutdown()
        except Exception:
            logging.exception("Stopping response failed")

    @Slot(object, bool)
    def reload_config(self, overrides: object, persist: bool) -> None:
        if self._reload_thread is not None and self._reload_thread.is_alive():
            self.failed.emit("A reload is already in progress.")
            return

        data = dict(overrides) if isinstance(overrides, dict) else {}
        logging.info("Worker reload requested (persist=%s).", persist)
        self.status.emit("Reloading Celeste with updated settings...")

        def _run_reload() -> None:
            try:
                logging.info("Worker calling service.reload() from Python thread.")
                cfg = self.service.reload(data, persist=persist, status_cb=self.status.emit)
                self.service.set_reflection_flag_cb(lambda reason: self.rulebook_flagged.emit(reason))
                logging.info("Worker service.reload() returned.")
                self.reloaded.emit(cfg)
                self.models_ready.emit(self.service.available_models())
                self.status.emit("Reload complete.")
            except Exception as exc:
                logging.exception("Reload failed")
                self.failed.emit(str(exc))
            finally:
                self._reload_thread = None

        self._reload_thread = threading.Thread(
            target=_run_reload,
            name="celeste-reload-worker",
            daemon=True,
        )
        self._reload_thread.start()

    @Slot(str)
    def add_rag_directory(self, directory: str) -> None:
        try:
            active_path = os.path.abspath(os.path.expanduser(os.path.expandvars(directory)))
            logging.info("Worker add RAG directory requested: %s", active_path)
            self.status.emit(f"Indexing files under {active_path}...")
            cfg, stats = self.service.add_rag_directory(directory)
            message = (
                f"Indexed {stats.get('files_indexed', 0)} files into "
                f"{stats.get('chunks_indexed', 0)} chunks."
            )
            self.rag_updated.emit(cfg, message)
            self.status.emit("File index updated.")
        except Exception as exc:
            logging.exception("Adding RAG directory failed")
            self.failed.emit(str(exc))

    @Slot()
    def reindex_rag(self) -> None:
        try:
            logging.info("Worker RAG reindex requested.")
            self.status.emit("Reindexing file knowledge base...")
            cfg, stats = self.service.reindex_rag()
            message = (
                f"Reindexed {stats.get('files_indexed', 0)} files into "
                f"{stats.get('chunks_indexed', 0)} chunks."
            )
            self.rag_updated.emit(cfg, message)
            self.status.emit("File index updated.")
        except Exception as exc:
            logging.exception("RAG reindex failed")
            self.failed.emit(str(exc))

    @Slot()
    def build_deep_index(self) -> None:
        if self._deep_index_thread is not None and self._deep_index_thread.is_alive():
            self.failed.emit("A deep index build is already in progress.")
            return

        logging.info("Worker deep-index build requested.")

        def on_progress(message: str, percent: int = 0) -> None:
            self.status.emit(message)
            self.progress.emit(int(percent), message)

        def _run_deep_index() -> None:
            try:
                on_progress("Building deep library index...", 0)
                cfg, stats = self.service.build_deep_rag_index(progress_cb=on_progress)
                message = (
                    f"Deep index ready: {stats.get('files_indexed', 0)} files, "
                    f"{stats.get('chunks_indexed', 0)} chunks."
                )
                self.rag_updated.emit(cfg, message)
                self.progress.emit(100, "Deep index ready.")
                self.status.emit("Deep index ready.")
            except Exception as exc:
                logging.exception("Deep index build failed")
                self.failed.emit(str(exc))
            finally:
                self._deep_index_thread = None

        self._deep_index_thread = threading.Thread(
            target=_run_deep_index,
            name="celeste-deep-index-worker",
            daemon=True,
        )
        self._deep_index_thread.start()

    @Slot(str)
    def remove_rag_directory(self, directory: str) -> None:
        try:
            logging.info("Worker remove RAG directory requested: %s", directory)
            self.status.emit(f"Removing {directory} from Celeste file access...")
            cfg, stats = self.service.remove_rag_directory(directory)
            message = (
                f"Removed directory. Remaining index contains "
                f"{stats.get('files_indexed', 0)} files across {stats.get('chunks_indexed', 0)} chunks."
            )
            self.rag_updated.emit(cfg, message)
            self.status.emit("File access updated.")
        except Exception as exc:
            logging.exception("Removing RAG directory failed")
            self.failed.emit(str(exc))

    @Slot()
    def shutdown(self) -> None:
        logging.info("Worker shutdown requested.")
        self.service.shutdown()

    @Slot(object)
    def purge_engram_memory(self, seconds: object) -> None:
        try:
            logging.info("Worker Engram purge requested: seconds=%s", seconds)
            purge_seconds = None if seconds is None else int(seconds)
            if purge_seconds is None:
                self.status.emit("Purging all Engram memory...")
            else:
                self.status.emit("Purging recent Engram memory...")
            cfg, stats = self.service.purge_engram_memory(seconds=purge_seconds)
            deleted = int(stats.get("purged", stats.get("entries_deleted", 0)))
            remaining = int(stats.get("entries", stats.get("entries_remaining", 0)))
            message = f"Purged {deleted} Engram entries. Remaining Engram entries: {remaining}."
            self.memory_updated.emit(cfg, message)
            self.status.emit("Engram memory updated.")
        except Exception as exc:
            logging.exception("Engram purge failed")
            self.failed.emit(str(exc))

    @Slot(bool)
    def set_engram_auto_prune(self, enabled: bool) -> None:
        try:
            logging.info("Worker Engram auto-prune requested: enabled=%s", enabled)
            self.status.emit(
                "Enabling Engram auto-purge..." if enabled else "Disabling Engram auto-purge..."
            )
            cfg, stats = self.service.set_engram_auto_prune(bool(enabled))
            retention_days = int(stats.get("retention_days", 30))
            keep_min_uses = int(stats.get("keep_min_uses", 3))
            if enabled:
                message = (
                    f"Engram auto-purge enabled: entries older than {retention_days} days "
                    f"are pruned unless used at least {keep_min_uses} times."
                )
            else:
                message = "Engram auto-purge disabled."
            self.memory_updated.emit(cfg, message)
            self.status.emit("Engram memory settings updated.")
        except Exception as exc:
            logging.exception("Updating Engram auto-prune failed")
            self.failed.emit(str(exc))


class CelesteWindow(QMainWindow):
    initialize_requested = Signal()
    load_models_requested = Signal()
    send_requested = Signal(str)
    reload_requested = Signal(object, bool)
    add_rag_directory_requested = Signal(str)
    remove_rag_directory_requested = Signal(str)
    reindex_rag_requested = Signal()
    build_deep_index_requested = Signal()
    purge_engram_requested = Signal(object)
    set_engram_auto_prune_requested = Signal(bool)
    shutdown_requested = Signal()
    stop_response_requested = Signal()

    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.log_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), "celeste_desktop.log")
        self.cfg: AgentConfig | None = None
        self.busy = False
        self._busy_reason: str | None = None
        self._force_quit = False
        self._build_ui()
        self._connect_live_log()
        self._build_worker()
        self.load_models_requested.emit()
        self.initialize_requested.emit()

    def _build_ui(self) -> None:
        self.setWindowTitle("Celeste")
        icon_path = resource_path("assets", "celeste_icon.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.resize(1280, 820)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)

        settings_card = QFrame()
        settings_card.setObjectName("settingsCard")
        settings_card.setMinimumWidth(120)
        settings_card.setMaximumWidth(420)
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(14, 14, 14, 14)
        settings_layout.setSpacing(10)

        title = QLabel("Control Deck")
        title.setObjectName("panelTitle")
        settings_layout.addWidget(title)

        subtitle = QLabel("Switch models, toggle speech, and reload the backend without leaving the app.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("panelSubtitle")
        settings_layout.addWidget(subtitle)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(14)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        self.model_combo.setMinimumHeight(32)
        form.addRow("LLM Model", self.model_combo)
        _spacer = QWidget()
        _spacer.setFixedHeight(6)
        form.addRow(_spacer)

        self.tts_toggle = QCheckBox("Enable Piper speech")
        form.addRow("Speech", self.tts_toggle)

        voice_row = QWidget()
        voice_row_layout = QHBoxLayout(voice_row)
        voice_row_layout.setContentsMargins(0, 0, 0, 0)
        voice_row_layout.setSpacing(6)
        self.piper_voice_label = QLabel("(none)")
        self.piper_voice_label.setStyleSheet("color: #8da2b5;")
        self.piper_voice_label.setMinimumWidth(120)
        voice_row_layout.addWidget(self.piper_voice_label, 1)
        self.piper_voice_browse = QPushButton("Browse…")
        self.piper_voice_browse.setFixedWidth(80)
        self.piper_voice_browse.clicked.connect(self._browse_piper_voice)
        voice_row_layout.addWidget(self.piper_voice_browse)
        form.addRow("Piper Voice", voice_row)
        self._piper_voice_path: str = ""

        self.memory_toggle = QCheckBox("Use vector memory")
        form.addRow("Memory", self.memory_toggle)

        self.reflection_toggle = QCheckBox("Enable reflection pass")
        form.addRow("Reflection", self.reflection_toggle)

        self.reflection_model_combo = QComboBox()
        self.reflection_model_combo.setEditable(True)
        self.reflection_model_combo.setInsertPolicy(QComboBox.NoInsert)
        self.reflection_model_combo.setMinimumHeight(32)
        self.reflection_model_combo.addItem("(use main model)", userData="")
        form.addRow("Reflect Model", self.reflection_model_combo)

        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(64, 8192)
        self.tokens_spin.setSingleStep(64)
        self.tokens_spin.setMinimumHeight(32)
        form.addRow("Max Tokens", self.tokens_spin)

        settings_layout.addSpacing(6)
        settings_layout.addLayout(form)
        settings_layout.addSpacing(10)

        button_grid = QGridLayout()
        button_grid.setHorizontalSpacing(10)
        button_grid.setVerticalSpacing(10)
        self.reload_button = QPushButton("Apply and Reload")
        self.refresh_models_button = QPushButton("Refresh Models")
        self.rulebook_button = QPushButton("View Rulebook")
        self.persona_button = QPushButton("Edit Persona")
        self.live_log_button = QPushButton("Live Log")
        self.live_log_button.setCheckable(True)
        self.shutdown_button = QPushButton("Shutdown Celeste")
        for button in (
            self.reload_button,
            self.refresh_models_button,
            self.rulebook_button,
            self.persona_button,
        ):
            button.setMinimumHeight(30)
            button.setMinimumWidth(0)
        button_grid.addWidget(self.reload_button, 0, 0)
        button_grid.addWidget(self.refresh_models_button, 0, 1)
        button_grid.addWidget(self.rulebook_button, 1, 0)
        button_grid.addWidget(self.persona_button, 1, 1)
        button_grid.setColumnStretch(0, 1)
        button_grid.setColumnStretch(1, 1)
        settings_layout.addLayout(button_grid)
        settings_layout.addSpacing(10)

        engram_title = QLabel("Engram Memory")
        engram_title.setObjectName("panelTitle")
        settings_layout.addWidget(engram_title)

        engram_subtitle = QLabel(
            "Purge exact-recall Engram traces manually, or keep auto-purge enabled for stale entries."
        )
        engram_subtitle.setWordWrap(True)
        engram_subtitle.setObjectName("panelSubtitle")
        settings_layout.addWidget(engram_subtitle)

        self.engram_auto_toggle = QCheckBox(
            "Auto-purge Engram entries older than 30 days unless frequently used"
        )
        settings_layout.addWidget(self.engram_auto_toggle)

        self.engram_count_label = QLabel("Stored memories: —")
        self.engram_count_label.setObjectName("panelSubtitle")
        settings_layout.addWidget(self.engram_count_label)

        engram_row = QHBoxLayout()
        engram_row.setSpacing(10)
        self.engram_purge_combo = QComboBox()
        self.engram_purge_combo.addItem("Purge all", userData=None)
        self.engram_purge_combo.addItem("Purge last 24 hours", userData=24 * 60 * 60)
        self.engram_purge_combo.addItem("Purge last 7 days", userData=7 * 24 * 60 * 60)
        self.engram_purge_combo.addItem("Purge last 30 days", userData=30 * 24 * 60 * 60)
        self.engram_purge_combo.addItem("Purge last 90 days", userData=90 * 24 * 60 * 60)
        self.engram_purge_combo.setMinimumHeight(32)
        self.engram_purge_button = QPushButton("Purge Engrams")
        self.engram_purge_button.setMinimumHeight(30)
        engram_row.addWidget(self.engram_purge_combo, 1)
        engram_row.addWidget(self.engram_purge_button)
        settings_layout.addLayout(engram_row)

        rag_title = QLabel("File RAG")
        rag_title.setObjectName("panelTitle")
        settings_layout.addWidget(rag_title)

        rag_subtitle = QLabel("Choose directories Celeste can search when answering questions about local files.")
        rag_subtitle.setWordWrap(True)
        rag_subtitle.setObjectName("panelSubtitle")
        settings_layout.addWidget(rag_subtitle)

        self.deep_index_status_label = QLabel("Deep index: unknown")
        self.deep_index_status_label.setObjectName("panelSubtitle")
        settings_layout.addWidget(self.deep_index_status_label)

        self.rag_dirs_list = QListWidget()
        self.rag_dirs_list.setMinimumHeight(150)
        settings_layout.addWidget(self.rag_dirs_list)

        rag_buttons = QGridLayout()
        rag_buttons.setHorizontalSpacing(10)
        rag_buttons.setVerticalSpacing(10)
        self.add_directory_button = QPushButton("Add Directory")
        self.remove_directory_button = QPushButton("Remove Directory")
        self.reindex_button = QPushButton("Reindex Files")
        self.build_deep_button = QPushButton("Build Deep Index")
        for button in (
            self.add_directory_button,
            self.remove_directory_button,
            self.reindex_button,
            self.build_deep_button,
        ):
            button.setMinimumHeight(30)
            button.setMinimumWidth(0)
        rag_buttons.addWidget(self.add_directory_button, 0, 0)
        rag_buttons.addWidget(self.remove_directory_button, 0, 1)
        rag_buttons.addWidget(self.reindex_button, 1, 0)
        rag_buttons.addWidget(self.build_deep_button, 1, 1)
        rag_buttons.setColumnStretch(0, 1)
        rag_buttons.setColumnStretch(1, 1)
        settings_layout.addLayout(rag_buttons)

        self.status_label = QLabel("Starting...")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        settings_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)
        settings_layout.addWidget(self.progress_bar)
        settings_layout.addStretch(1)

        chat_card = QFrame()
        chat_card.setObjectName("chatCard")
        chat_layout = QVBoxLayout(chat_card)
        chat_layout.setContentsMargins(18, 18, 18, 18)
        chat_layout.setSpacing(12)

        chat_title = QLabel("Conversation")
        chat_title.setObjectName("panelTitle")
        chat_layout.addWidget(chat_title)

        self.chat_hint = QLabel(
            'Library lookup hint: mention "indexed library", "documents", "files", or ask Celeste to search the library when you want document-backed answers.'
        )
        self.chat_hint.setWordWrap(True)
        self.chat_hint.setObjectName("panelSubtitle")
        chat_layout.addWidget(self.chat_hint)

        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setReadOnly(True)
        chat_layout.addWidget(self.chat_view, 1)

        # Streaming preview — visible only while a reply is being generated
        self.stream_frame = QFrame()
        self.stream_frame.setObjectName("streamFrame")
        self.stream_frame.setStyleSheet(
            "#streamFrame { background: #0c1116; border: 1px solid #25313d; border-radius: 10px; padding: 6px; }"
        )
        stream_frame_layout = QVBoxLayout(self.stream_frame)
        stream_frame_layout.setContentsMargins(8, 6, 8, 6)
        stream_frame_layout.setSpacing(4)
        self._stream_speaker_label = QLabel("Celeste")
        self._stream_speaker_label.setStyleSheet("font-weight: 700; color: #a8ffcf;")
        stream_frame_layout.addWidget(self._stream_speaker_label)
        self.stream_preview = QPlainTextEdit()
        self.stream_preview.setReadOnly(True)
        self.stream_preview.setMaximumHeight(200)
        self.stream_preview.setStyleSheet(
            "QPlainTextEdit { background: transparent; border: none; color: #d6e8f7; }"
        )
        stream_frame_layout.addWidget(self.stream_preview)
        self.stream_frame.hide()
        chat_layout.addWidget(self.stream_frame)

        self.input_box = QPlainTextEdit()
        self.input_box.setPlaceholderText("Type a prompt for Celeste...")
        self.input_box.setFixedHeight(120)
        chat_layout.addWidget(self.input_box)

        send_row = QHBoxLayout()
        send_row.setSpacing(10)
        self.send_button = QPushButton("Send")
        self.clear_button = QPushButton("Clear Transcript")
        self.export_button = QPushButton("Export Chat")
        self.stop_button = QPushButton("Stop Response")
        send_row.addWidget(self.send_button)
        send_row.addWidget(self.clear_button)
        send_row.addWidget(self.export_button)
        send_row.addWidget(self.stop_button)
        send_row.addStretch(1)
        self.live_log_button.setMinimumHeight(30)
        self.shutdown_button.setMinimumHeight(30)
        send_row.addWidget(self.live_log_button)
        send_row.addWidget(self.shutdown_button)
        chat_layout.addLayout(send_row)

        token_row = QHBoxLayout()
        token_row.setSpacing(8)
        self.token_label = QLabel("Context: — / —")
        self.token_label.setObjectName("panelSubtitle")
        self.token_bar = QProgressBar()
        self.token_bar.setRange(0, 100)
        self.token_bar.setValue(0)
        self.token_bar.setTextVisible(False)
        self.token_bar.setFixedHeight(6)
        self.token_bar.setStyleSheet(
            "QProgressBar { background: #1a2530; border-radius: 3px; }"
            "QProgressBar::chunk { background: #3a8f5a; border-radius: 3px; }"
        )
        token_row.addWidget(self.token_label)
        token_row.addWidget(self.token_bar, 1)
        chat_layout.addLayout(token_row)

        layout.addWidget(settings_card, 0)
        layout.addWidget(chat_card, 1)
        layout.setStretch(0, 0)
        layout.setStretch(1, 1)
        self.setCentralWidget(root)

        self.deep_index_dialog = QProgressDialog("Building deep library index...", "Force Shutdown", 0, 100, self)
        self.deep_index_dialog.setWindowTitle("Building Deep Index")
        self.deep_index_dialog.setAutoClose(False)
        self.deep_index_dialog.setAutoReset(False)
        self.deep_index_dialog.setMinimumDuration(0)
        self.deep_index_dialog.setWindowModality(Qt.NonModal)
        self.deep_index_dialog.hide()

        mono = QFont("DejaVu Sans Mono", 10)
        self.chat_view.setFont(mono)
        self.input_box.setFont(mono)
        self.stream_preview.setFont(mono)

        self.log_dialog = LiveLogDialog(self)
        self.log_dialog.setWindowTitle("Celeste Live Log")
        self.log_dialog.resize(980, 520)
        self.log_dialog.setModal(False)
        dialog_layout = QVBoxLayout(self.log_dialog)
        dialog_layout.setContentsMargins(14, 14, 14, 14)
        dialog_layout.setSpacing(10)
        dialog_hint = QLabel("Runtime status, backend steps, and exceptions stream here in real time.")
        dialog_hint.setWordWrap(True)
        dialog_layout.addWidget(dialog_hint)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(1500)
        self.log_view.setFont(mono)
        dialog_layout.addWidget(self.log_view)

        self.reload_button.clicked.connect(self._apply_settings)
        self.refresh_models_button.clicked.connect(lambda: self.load_models_requested.emit())
        self.live_log_button.toggled.connect(self._toggle_live_log)
        self.shutdown_button.clicked.connect(self._shutdown_app)
        self.send_button.clicked.connect(self._send_message)
        self.clear_button.clicked.connect(self.chat_view.clear)
        self.export_button.clicked.connect(self._export_chat)
        self.stop_button.clicked.connect(self._stop_response)
        self.add_directory_button.clicked.connect(self._choose_rag_directory)
        self.remove_directory_button.clicked.connect(self._remove_selected_rag_directory)
        self.reindex_button.clicked.connect(lambda: self.reindex_rag_requested.emit())
        self.build_deep_button.clicked.connect(self._start_deep_index_build)
        self.engram_purge_button.clicked.connect(self._purge_engram_memory)
        self.engram_auto_toggle.toggled.connect(self._set_engram_auto_prune)
        self.log_dialog.closed.connect(self._on_log_dialog_closed)
        self.deep_index_dialog.canceled.connect(self._shutdown_app)

        self._set_busy(True, "Starting Celeste...")
        self.setStyleSheet(
            """
            QWidget {
                background: #0f1419;
                color: #edf2f7;
                font-family: "DejaVu Sans";
                font-size: 14px;
            }
        QFrame#settingsCard, QFrame#chatCard {
            background: #162029;
            border: 1px solid #274050;
            border-radius: 16px;
        }
        QLabel#panelTitle {
            font-size: 18px;
            font-weight: 700;
            color: #a8ffcf;
        }
        QLabel#panelSubtitle, QLabel#statusLabel {
            color: #a9b7c6;
            font-size: 12px;
        }
        QTextBrowser, QPlainTextEdit, QComboBox, QSpinBox {
            background: #0c1116;
            border: 1px solid #2b3946;
            border-radius: 10px;
            padding: 4px 8px;
            min-height: 24px;
            font-size: 12px;
        }
        QPushButton {
            background: #1f7a5c;
            border: none;
            border-radius: 10px;
            padding: 5px 8px;
            font-weight: 600;
            font-size: 11px;
            min-width: 0px;
        }
        QPushButton:disabled {
            background: #33414d;
            color: #7f8b96;
        }
        QPushButton:hover:!disabled {
            background: #27956f;
        }
        QCheckBox {
            spacing: 6px;
            font-size: 12px;
        }
            """
        )

    def _build_worker(self) -> None:
        self.worker_thread = QThread(self)
        self.worker = ServiceWorker(self.config_path)
        self.worker.moveToThread(self.worker_thread)

        self.initialize_requested.connect(self.worker.initialize)
        self.load_models_requested.connect(self.worker.load_models)
        self.send_requested.connect(self.worker.send_message)
        self.reload_requested.connect(self.worker.reload_config)
        self.add_rag_directory_requested.connect(self.worker.add_rag_directory)
        self.remove_rag_directory_requested.connect(self.worker.remove_rag_directory)
        self.reindex_rag_requested.connect(self.worker.reindex_rag)
        self.build_deep_index_requested.connect(self.worker.build_deep_index)
        self.purge_engram_requested.connect(self.worker.purge_engram_memory)
        self.set_engram_auto_prune_requested.connect(self.worker.set_engram_auto_prune)
        self.shutdown_requested.connect(self.worker.shutdown)
        self.stop_response_requested.connect(self.worker.stop_response)

        self.worker.initialized.connect(self._on_initialized)
        self.worker.models_ready.connect(self._on_models_ready)
        self.worker.reloaded.connect(self._on_reloaded)
        self.worker.rag_updated.connect(self._on_rag_updated)
        self.worker.memory_updated.connect(self._on_memory_updated)
        self.worker.token_received.connect(self._on_token_received)
        self.worker.token_usage.connect(self._on_token_usage)
        self.worker.reply_ready.connect(self._on_reply_ready)
        self.worker.chat_finished.connect(self._on_chat_finished)
        self.worker.response_stopped.connect(self._on_response_stopped)
        self.worker.failed.connect(self._on_failed)
        self.worker.status.connect(self._set_status)
        self.worker.progress.connect(self._on_progress)
        self.worker.rulebook_flagged.connect(self._on_rulebook_flagged)

        self.rulebook_button.clicked.connect(self._open_rulebook)
        self.persona_button.clicked.connect(self._open_persona)

        self.worker_thread.start()

    def _connect_live_log(self) -> None:
        emitter = _get_log_emitter()
        emitter.message.connect(self._append_log_line)
        for line in list(_LOG_BUFFER):
            self._append_log_line(line)
        self._append_log_line(f"Live desktop log: {self.log_path}")

    @Slot(bool)
    def _toggle_live_log(self, checked: bool) -> None:
        if checked:
            self.log_dialog.show()
            self.log_dialog.raise_()
            self.log_dialog.activateWindow()
        else:
            self.log_dialog.hide()

    @Slot()
    def _on_log_dialog_closed(self) -> None:
        self.live_log_button.blockSignals(True)
        self.live_log_button.setChecked(False)
        self.live_log_button.blockSignals(False)

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        self.busy = busy
        self.send_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy)
        self.input_box.setEnabled(not busy)
        self.reload_button.setEnabled(not busy)
        self.refresh_models_button.setEnabled(not busy)
        self.add_directory_button.setEnabled(not busy)
        self.remove_directory_button.setEnabled(not busy)
        self.reindex_button.setEnabled(not busy)
        self.build_deep_button.setEnabled(not busy)
        self.engram_auto_toggle.setEnabled(not busy)
        self.engram_purge_combo.setEnabled(not busy)
        self.engram_purge_button.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.reflection_model_combo.setEnabled(not busy)
        self.tokens_spin.setEnabled(not busy)
        self.tts_toggle.setEnabled(not busy)
        self.piper_voice_browse.setEnabled(not busy)
        self.memory_toggle.setEnabled(not busy)
        self.reflection_toggle.setEnabled(not busy)
        self.rulebook_button.setEnabled(not busy)
        self.persona_button.setEnabled(not busy)
        self.rag_dirs_list.setEnabled(not busy)
        self.stop_button.setEnabled(bool(busy and self._busy_reason == "chat"))
        if status:
            self.status_label.setText(status)
        if not busy:
            self._busy_reason = None
            self._hide_progress_ui()

    def _show_deep_index_progress(self, message: str, percent: int = 0) -> None:
        self.progress_bar.setVisible(True)
        if int(percent) < 0:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Working...")
        else:
            bounded = max(0, min(100, int(percent)))
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(bounded)
            self.progress_bar.setFormat(f"{bounded}%")
        self.deep_index_dialog.setLabelText(message)
        if int(percent) < 0:
            self.deep_index_dialog.setRange(0, 0)
        else:
            bounded = max(0, min(100, int(percent)))
            self.deep_index_dialog.setRange(0, 100)
            self.deep_index_dialog.setValue(bounded)
        if not self.deep_index_dialog.isVisible():
            self.deep_index_dialog.show()

    def _hide_progress_ui(self) -> None:
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        if self.deep_index_dialog.isVisible():
            self.deep_index_dialog.hide()
        self.deep_index_dialog.setRange(0, 100)
        self.deep_index_dialog.setValue(0)

    @Slot(str)
    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        logging.info("STATUS %s", text)

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        if self._busy_reason == "deep_index":
            self._show_deep_index_progress(message, percent)

    @Slot(object)
    def _on_initialized(self, cfg: object) -> None:
        self.cfg = cfg if isinstance(cfg, AgentConfig) else None
        if self.cfg is not None:
            self._populate_from_config(self.cfg)
        if self.chat_view.toPlainText().strip():
            self._append_session_separator()
        self._append_system("Celeste backend is online.")
        self._set_busy(False, "Celeste ready.")

    @Slot(object)
    def _on_reloaded(self, cfg: object) -> None:
        self.cfg = cfg if isinstance(cfg, AgentConfig) else None
        if self.cfg is not None:
            self._populate_from_config(self.cfg)
        self._append_system("Settings applied. Backend reloaded.")
        self._set_busy(False, "Reload complete.")

    @Slot(list)
    def _on_models_ready(self, models: list) -> None:
        current = self._selected_model_path()
        name_counts: dict[str, int] = {}
        for model in models:
            base = os.path.basename(str(model))
            name_counts[base] = name_counts.get(base, 0) + 1
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model in models:
            path = str(model)
            label = self._model_label(path, duplicate=name_counts.get(os.path.basename(path), 0) > 1)
            self.model_combo.addItem(label, userData=path)
            index = self.model_combo.count() - 1
            self.model_combo.setItemData(index, path, Qt.ToolTipRole)
        if current:
            self._set_selected_model(current)
        elif self.cfg is not None:
            self._set_selected_model(self.cfg.model_path)
        self.model_combo.blockSignals(False)
        # Populate reflection model combo with the same model list
        current_reflect = self.reflection_model_combo.currentData() or ""
        self.reflection_model_combo.blockSignals(True)
        self.reflection_model_combo.clear()
        self.reflection_model_combo.addItem("(use main model)", userData="")
        for model in models:
            path = str(model)
            label = self._model_label(path, duplicate=name_counts.get(os.path.basename(path), 0) > 1)
            self.reflection_model_combo.addItem(label, userData=path)
        # Re-select previously chosen reflection model
        if not current_reflect and self.cfg is not None:
            current_reflect = str(
                dict(self.cfg.reflection or {}).get("model_path", "") or ""
            )
        for i in range(self.reflection_model_combo.count()):
            if self.reflection_model_combo.itemData(i) == current_reflect:
                self.reflection_model_combo.setCurrentIndex(i)
                break
        self.reflection_model_combo.blockSignals(False)

    @Slot(object, str)
    def _on_rag_updated(self, cfg: object, message: str) -> None:
        self.cfg = cfg if isinstance(cfg, AgentConfig) else self.cfg
        if self.cfg is not None:
            self._populate_from_config(self.cfg)
        self._append_system(message)
        if self._busy_reason == "deep_index":
            status = "Deep index ready."
        elif self._busy_reason == "rag_dir_change":
            status = "File RAG: New directory loaded \u2014 rebuild deep index"
        else:
            status = "Library index updated."
        self._set_busy(False, status)

    @Slot(object, str)
    def _on_memory_updated(self, cfg: object, message: str) -> None:
        self.cfg = cfg if isinstance(cfg, AgentConfig) else self.cfg
        if self.cfg is not None:
            self._populate_from_config(self.cfg)
        self._append_system(message)
        self._set_busy(False, "Engram memory updated.")

    @Slot(str)
    def _on_token_received(self, token: str) -> None:
        if not self.stream_frame.isVisible():
            self.stream_preview.setPlainText("")
            self.stream_frame.show()
        self.stream_preview.moveCursor(QTextCursor.End)
        self.stream_preview.insertPlainText(token)
        self.stream_preview.ensureCursorVisible()

    @Slot(int, int)
    def _on_token_usage(self, used: int, ctx: int) -> None:
        if ctx > 0:
            pct = min(100, int(used * 100 / ctx))
            self.token_label.setText(f"Context: {used:,} / {ctx:,} tokens")
            self.token_bar.setValue(pct)
            color = "#3a8f5a" if pct < 70 else "#c8842a" if pct < 90 else "#c83a3a"
            self.token_bar.setStyleSheet(
                "QProgressBar { background: #1a2530; border-radius: 3px; }"
                f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}"
            )
            if pct >= 90 and not getattr(self, "_ctx_warned", False):
                self._ctx_warned = True
                self._append_system("Context window is nearly full. Consider starting a new session for best results.")
            elif pct < 90:
                self._ctx_warned = False

    def _hide_stream_preview(self) -> None:
        self.stream_frame.hide()
        self.stream_preview.setPlainText("")

    @Slot(str, str, str)
    def _on_reply_ready(self, answer: str, critique: str, improvements: str) -> None:
        self._hide_stream_preview()
        self._append_chat("Celeste", answer, "#a8ffcf", split_sources=True)
        if critique.strip():
            self._append_chat("Critique", critique, "#ffd6a5")
        if improvements.strip():
            self._append_chat("Playbook", improvements, "#9fd3ff")
        self._refresh_engram_count()

    @Slot(str)
    def _on_chat_finished(self, message: str) -> None:
        self._set_busy(False, message or "Ready.")

    @Slot(str)
    def _on_response_stopped(self, message: str) -> None:
        self._hide_stream_preview()
        self._append_system(message or "Response stopped.")
        self._set_busy(False, message or "Response stopped.")

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._hide_stream_preview()
        self._append_system(f"Error: {message}")
        self._set_busy(False, "Error.")
        QMessageBox.critical(self, "Celeste Error", message)

    @Slot(str)
    def _on_rulebook_flagged(self, reason: str) -> None:
        self._append_system(
            f"Rulebook: {reason} — click \u2018View Rulebook\u2019 to review."
        )
        self._set_status("Rulebook review recommended.")

    def _open_rulebook(self) -> None:
        rules = self.worker.service.get_rulebook()
        dialog = RulebookDialog(rules, parent=self)
        if dialog.exec() == QDialog.Accepted:
            for rule_id in dialog.get_deletes():
                self.worker.service.delete_rulebook_rule(rule_id)
            for rule_id, text in dialog.get_updates().items():
                self.worker.service.update_rulebook_rule(rule_id, text)

    def _open_persona(self) -> None:
        current = str(getattr(self.worker.service.cfg, "system_preamble", "") or "")
        dialog = PersonaDialog(current, parent=self)
        if dialog.exec() == QDialog.Accepted:
            self.worker.service.set_persona(dialog.get_preamble())
            self._append_system("Persona updated.")

    def _export_chat(self) -> None:
        from datetime import datetime
        default_name = f"celeste_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        path, _ = QFileDialog.getSaveFileName(self, "Export Chat", default_name, "Markdown (*.md);;Text (*.txt)")
        if not path:
            return
        plain = self.chat_view.toPlainText().strip()
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Celeste Conversation Export\n")
                f.write(f"_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
                f.write(plain)
            self._append_system(f"Conversation exported to {os.path.basename(path)}")
        except Exception as exc:
            self._append_system(f"Export failed: {exc}")

    def _append_session_separator(self) -> None:
        self.chat_view.append(
            "<div style='margin: 14px 0; text-align: center;'>"
            "<hr style='border: none; border-top: 1px solid #25313d; margin: 0;'>"
            "<span style='color: #4a6070; font-size: 11px;'>— new session —</span>"
            "</div>"
        )

    def _refresh_engram_count(self) -> None:
        try:
            count = self.worker.service.get_engram_count()
            self.engram_count_label.setText(f"Stored memories: {count:,}")
        except Exception:
            pass

    def _refresh_deep_index_status(self, cfg) -> None:
        try:
            agent = self.worker.service.agent
            if agent is None:
                self.deep_index_status_label.setText("Deep index: not loaded")
                return
            use_embeddings = bool(getattr(cfg, "file_rag_use_embeddings", False))
            if not use_embeddings:
                self.deep_index_status_label.setText("Deep index: disabled (embeddings off)")
            elif agent.file_rag.deep_index_available():
                self.deep_index_status_label.setText("Deep index: active (semantic search enabled)")
            else:
                self.deep_index_status_label.setText("Deep index: not built — use Build Deep Index")
        except Exception:
            pass

    def _browse_piper_voice(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Piper Voice Model", "", "Piper ONNX Models (*.onnx)"
        )
        if path:
            self._piper_voice_path = path
            self.piper_voice_label.setText(os.path.basename(path))
            self.piper_voice_label.setToolTip(path)

    def _populate_from_config(self, cfg: AgentConfig) -> None:
        self._set_selected_model(cfg.model_path)
        self.tts_toggle.setChecked(bool(cfg.tts_enabled))
        piper_model = str(getattr(cfg, "tts_piper_model", "") or "")
        self._piper_voice_path = piper_model
        self.piper_voice_label.setText(os.path.basename(piper_model) if piper_model else "(none)")
        self.piper_voice_label.setToolTip(piper_model)
        self.memory_toggle.setChecked(bool(cfg.use_chroma))
        reflection_cfg = cfg.reflection or {}
        self.reflection_toggle.setChecked(bool(reflection_cfg.get("enabled", False)))
        self.tokens_spin.setValue(int(cfg.max_new_tokens))
        memory_cfg = dict(cfg.memory or {})
        self.engram_auto_toggle.blockSignals(True)
        self.engram_auto_toggle.setChecked(bool(memory_cfg.get("engram_auto_prune", True)))
        self.engram_auto_toggle.blockSignals(False)
        self.rag_dirs_list.clear()
        counts = self.worker.service.rag_directory_counts()
        for path in cfg.file_rag_dirs:
            count = counts.get(path)
            label = f"{path} ({count} files)" if count is not None else path
            self.rag_dirs_list.addItem(label)
        self._refresh_engram_count()
        self._refresh_deep_index_status(cfg)

    def _model_label(self, path: str, *, duplicate: bool = False) -> str:
        base = os.path.basename(path)
        if not duplicate:
            return base
        parent = os.path.basename(os.path.dirname(path)) or "models"
        return f"{base} [{parent}]"

    def _set_selected_model(self, model_path: str) -> None:
        index = self.model_combo.findData(model_path)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
            return
        label = self._model_label(model_path)
        self.model_combo.setEditText(label)
        self.model_combo.setToolTip(model_path)

    def _selected_model_path(self) -> str:
        data = self.model_combo.currentData()
        if isinstance(data, str) and data.strip():
            return data
        return self.model_combo.currentText().strip()

    @staticmethod
    def _markdown_to_html(text: str) -> str:
        """Convert a subset of markdown to HTML for chat display."""
        # Escape HTML first, then convert markdown patterns back to tags
        escaped = html.escape(text)
        # Headers (## Header → bold line)
        escaped = re.sub(r"^######\s+(.+)$", r"<b>\1</b>", escaped, flags=re.MULTILINE)
        escaped = re.sub(r"^#####\s+(.+)$", r"<b>\1</b>", escaped, flags=re.MULTILINE)
        escaped = re.sub(r"^####\s+(.+)$", r"<b>\1</b>", escaped, flags=re.MULTILINE)
        escaped = re.sub(r"^###\s+(.+)$", r"<b>\1</b>", escaped, flags=re.MULTILINE)
        escaped = re.sub(r"^##\s+(.+)$", r"<b>\1</b>", escaped, flags=re.MULTILINE)
        escaped = re.sub(r"^#\s+(.+)$", r"<b>\1</b>", escaped, flags=re.MULTILINE)
        # Bold+italic ***text*** or ___text___
        escaped = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", escaped)
        escaped = re.sub(r"___(.+?)___", r"<b><i>\1</i></b>", escaped)
        # Bold **text** or __text__
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
        escaped = re.sub(r"__(.+?)__", r"<b>\1</b>", escaped)
        # Italic *text* or _text_
        escaped = re.sub(r"\*(.+?)\*", r"<i>\1</i>", escaped)
        escaped = re.sub(r"_(.+?)_", r"<i>\1</i>", escaped)
        # Inline code `text`
        escaped = re.sub(
            r"`(.+?)`",
            r"<code style='background:#1a2530;padding:1px 4px;border-radius:3px;'>\1</code>",
            escaped,
        )
        # Bullet lists (- item or * item)
        escaped = re.sub(r"^[-*]\s+(.+)$", r"&nbsp;&nbsp;• \1", escaped, flags=re.MULTILINE)
        # Newlines to <br>
        escaped = escaped.replace("\n", "<br>")
        return escaped

    def _append_system(self, text: str) -> None:
        self.chat_view.append(
            f"<div style='margin: 6px 0; color: #8da2b5;'><i>{html.escape(text)}</i></div>"
        )

    @Slot(str)
    def _append_log_line(self, text: str) -> None:
        clean = (text or "").rstrip()
        if not clean:
            return
        self.log_view.appendPlainText(clean)

    def _split_sources_block(self, text: str) -> tuple[str, list[str]]:
        match = re.search(r"\n\s*Sources:\s*\n", text or "")
        if not match:
            return text, []
        body = (text or "")[: match.start()].rstrip()
        tail = (text or "")[match.end():]
        lines = [line.strip() for line in tail.splitlines() if line.strip()]
        return body, lines

    def _append_chat(self, speaker: str, text: str, color: str, *, split_sources: bool = False) -> None:
        body_text = text
        source_lines: list[str] = []
        if split_sources:
            body_text, source_lines = self._split_sources_block(text)
        safe_text = self._markdown_to_html(body_text)
        safe_speaker = html.escape(speaker)
        sources_html = ""
        if source_lines:
            rendered_sources = "".join(
                f"<div style='margin: 4px 0;'>{html.escape(line)}</div>"
                for line in source_lines
            )
            sources_html = (
                "<div style='margin-top: 8px; background: #121922; border: 1px solid #314353; "
                "border-radius: 10px; padding: 10px;'>"
                "<div style='font-weight: 700; color: #9fd3ff; margin-bottom: 6px;'>Sources</div>"
                f"{rendered_sources}</div>"
            )
        self.chat_view.append(
            f"<div style='margin: 10px 0;'>"
            f"<div style='font-weight: 700; color: {color}; margin-bottom: 4px;'>{safe_speaker}</div>"
            f"<div style='background: #0c1116; border: 1px solid #25313d; border-radius: 10px; padding: 10px;'>"
            f"{safe_text}</div>"
            f"{sources_html}</div>"
        )

    def _send_message(self) -> None:
        if self.busy:
            return
        message = self.input_box.toPlainText().strip()
        if not message:
            return
        self.input_box.clear()
        self._append_chat("You", message, "#9fd3ff")
        self._busy_reason = "chat"
        self._set_busy(True, "Generating reply...")
        self.send_requested.emit(message)

    def _stop_response(self) -> None:
        if self._busy_reason != "chat":
            return
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopping response...")
        self.stop_response_requested.emit()

    def _choose_rag_directory(self) -> None:
        if self.busy:
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose Directory for Celeste File Access")
        if not directory:
            return
        self._busy_reason = "rag_dir_change"
        self._set_busy(True, f"Indexing {directory}...")
        self.add_rag_directory_requested.emit(directory)

    def _remove_selected_rag_directory(self) -> None:
        if self.busy:
            return
        item = self.rag_dirs_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Remove Directory", "Select a directory from the list first.")
            return
        directory = item.text().strip()
        if not directory:
            return
        self._busy_reason = "rag_dir_change"
        self._set_busy(True, f"Removing {directory}...")
        self.remove_rag_directory_requested.emit(directory)

    def _apply_settings(self) -> None:
        if self.busy:
            return
        reflection = dict((self.cfg.reflection if self.cfg is not None else {}) or {})
        reflection["enabled"] = self.reflection_toggle.isChecked()
        reflection_model_path = self.reflection_model_combo.currentData() or ""
        reflection["model_path"] = str(reflection_model_path)
        memory_cfg = dict((self.cfg.memory if self.cfg is not None else {}) or {})
        memory_cfg["engram_auto_prune"] = self.engram_auto_toggle.isChecked()
        overrides = {
            "model_path": self._selected_model_path(),
            "tts_enabled": self.tts_toggle.isChecked(),
            "tts_piper_model": self._piper_voice_path,
            "tts_piper_config": (
                self._piper_voice_path + ".json"
                if self._piper_voice_path and os.path.isfile(self._piper_voice_path + ".json")
                else None
            ),
            "use_chroma": self.memory_toggle.isChecked(),
            "max_new_tokens": int(self.tokens_spin.value()),
            "reflection": reflection,
            "memory": memory_cfg,
        }
        self._set_busy(True, "Applying settings...")
        self.reload_requested.emit(overrides, True)

    def _set_engram_auto_prune(self, enabled: bool) -> None:
        if self.busy:
            return
        self._set_busy(True, "Updating Engram auto-purge...")
        self.set_engram_auto_prune_requested.emit(bool(enabled))

    def _purge_engram_memory(self) -> None:
        if self.busy:
            return
        seconds = self.engram_purge_combo.currentData()
        label = self.engram_purge_combo.currentText().strip() or "this Engram range"
        reply = QMessageBox.question(
            self,
            "Purge Engram Memory",
            f"Proceed with: {label}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._set_busy(True, "Purging Engram memory...")
        self.purge_engram_requested.emit(seconds)

    def _start_deep_index_build(self) -> None:
        if self.busy:
            return
        self._busy_reason = "deep_index"
        self._set_busy(True, "Building deep library index...")
        self._show_deep_index_progress("Building deep library index...", 0)
        self.build_deep_index_requested.emit()

    def _shutdown_app(self) -> None:
        if self._busy_reason == "deep_index":
            reply = QMessageBox.question(
                self,
                "Force Shutdown Celeste",
                "Deep indexing is running. Force shutdown now? The partial deep-index build will be discarded.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            self._force_terminate_app("Force shutdown requested during deep indexing.")
            return
        if self.busy:
            reply = QMessageBox.question(
                self,
                "Shutdown Celeste",
                "Celeste is busy. Shut it down anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        else:
            reply = QMessageBox.question(
                self,
                "Shutdown Celeste",
                "Shut Celeste down now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self._set_busy(True, "Shutting down Celeste...")
        self.close()

    def _force_terminate_app(self, reason: str) -> None:
        logging.warning(reason)
        self._force_quit = True
        self._hide_progress_ui()
        self.status_label.setText("Force shutting down Celeste...")
        try:
            service = getattr(self.worker, "service", None)
            agent = getattr(service, "agent", None)
            if agent is not None:
                try:
                    agent.tts.shutdown()
                except Exception:
                    pass
                try:
                    agent.llm.shutdown()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.worker_thread.requestInterruption()
        except Exception:
            pass
        try:
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()
                self.worker_thread.wait(1000)
        except Exception:
            pass
        app = QApplication.instance()
        if app is not None:
            app.quit()
        QTimer.singleShot(250, lambda: os._exit(0))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._force_quit:
            event.accept()
            return
        try:
            QMetaObject.invokeMethod(self.worker, "shutdown", Qt.BlockingQueuedConnection)
        except Exception:
            self.shutdown_requested.emit()
        self.worker_thread.quit()
        self.worker_thread.wait(5000)
        super().closeEvent(event)


def main() -> int:
    mp.freeze_support()
    config_path = default_config_path()
    log_path = _setup_app_logging(config_path)
    _install_exception_logging()
    logging.info("Launching Celeste desktop app")
    logging.info("Desktop log path: %s", log_path)
    app = QApplication(sys.argv)
    icon_path = resource_path("assets", "celeste_icon.png")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    if not os.path.exists(config_path):
        from setup_wizard import ensure_config_with_wizard

        if not ensure_config_with_wizard(config_path, app=app):
            return 0
    window = CelesteWindow(config_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
