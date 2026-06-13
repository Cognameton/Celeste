#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import fnmatch
import os
import platform
import shutil
import sys

import yaml

try:
    from PySide6.QtCore import Qt, QThread, QTimer, Slot
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QDialog,
        QFileDialog,
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "PySide6 is required for the Celeste setup wizard. Install dependencies first:\n"
        "  pip install -r requirements.txt"
    ) from exc

from app_paths import default_config_path, resource_path, runtime_root
from validate_environment import format_checks, has_errors, validate_config_file


# ---------------------------------------------------------------------------
# Embedded AI engine setup widget (Linux packaged builds only)
# ---------------------------------------------------------------------------

def _needs_engine_setup() -> bool:
    """True when running as a packaged Linux app without llama-server yet."""
    if not getattr(sys, "frozen", False):
        return False
    if platform.system() != "Linux":
        return False
    from llama_installer import installed_llama_server
    return installed_llama_server() is None


class EngineSetupWidget(QFrame):
    """
    Self-contained widget that detects hardware and downloads/compiles
    llama-server automatically.  Embed at the top of the setup wizard.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self._done = False
        self._thread = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        title = QLabel("<b>Step 1 — Setting up the AI Engine</b>")
        layout.addWidget(title)

        from llama_installer import detect_hardware, fetch_release_assets, select_best_asset
        hw = detect_hardware()
        self._hw = hw

        hw_label = QLabel(f"Detected: {hw.describe()}")
        hw_label.setWordWrap(True)
        layout.addWidget(hw_label)

        try:
            tag, assets = fetch_release_assets()
            asset = select_best_asset(assets, hw)
            self._asset = asset
            if asset:
                size_mb = asset.size_bytes / 1_048_576
                plan = f"Will download: <code>{asset.name}</code> ({size_mb:.0f} MB)"
            else:
                from llama_installer import check_build_deps
                missing = check_build_deps(bool(hw.cuda_version))
                plan = "Will compile from source" + (
                    f" — missing tools: {', '.join(missing)}" if missing else ""
                )
        except Exception as exc:
            self._asset = None
            plan = f"Could not reach GitHub releases ({exc}). Will compile from source."

        plan_label = QLabel(plan)
        plan_label.setWordWrap(True)
        layout.addWidget(plan_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        self._log.setVisible(False)
        layout.addWidget(self._log)

        self._status = QLabel("Starting…")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # Auto-start after the dialog renders
        QTimer.singleShot(200, self._start)

    def _start(self):
        from llama_installer import InstallerWorker
        self._worker = InstallerWorker.make_qobject()
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    @Slot(int, int)
    def _on_progress(self, downloaded: int, total: int):
        pct = int(downloaded * 100 / total)
        self._progress.setValue(pct)
        mb_d = downloaded / 1_048_576
        mb_t = total / 1_048_576
        self._status.setText(f"Downloading… {mb_d:.1f} / {mb_t:.1f} MB")

    @Slot(str)
    def _on_log(self, line: str):
        self._log.setVisible(True)
        self._log.appendPlainText(line)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
        self._status.setText(line[:120])

    @Slot(bool, str)
    def _on_finished(self, success: bool, message: str):
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        if success:
            self._done = True
            self._progress.setValue(100)
            self._status.setText("AI engine ready.")
            self.setStyleSheet("QFrame { border: 1px solid #4CAF50; border-radius: 4px; }")
            # Notify parent wizard
            wizard = self.parent()
            while wizard and not isinstance(wizard, SetupWizardDialog):
                wizard = wizard.parent()
            if wizard:
                wizard.on_engine_ready()
        else:
            self._status.setText(f"Setup failed: {message}")
            retry = QPushButton("Retry")
            retry.clicked.connect(self._start)
            self.layout().addWidget(retry)

    def is_done(self) -> bool:
        return self._done


def _project_root() -> str:
    return runtime_root()


def _template_path() -> str:
    return resource_path("config.example.yaml")


def _default_paths() -> dict[str, str]:
    # Packaged app: use %LOCALAPPDATA%\Celeste on Windows, ~/.config/Celeste on Linux
    # Dev mode: use ~/Celeste as a sensible default outside the repo
    if getattr(sys, "frozen", False):
        if os.name == "nt":
            celeste_home = os.path.join(
                os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "Celeste"
            )
        else:
            base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
            celeste_home = os.path.join(base, "Celeste")
    else:
        celeste_home = os.path.join(os.path.expanduser("~"), "Celeste")

    bundled_model = _bundled_default_model_path()
    bundled_embedding = _bundled_default_embedding_dir()
    bundled_llama = _bundled_llama_server_path()
    bundled_piper = _bundled_piper_executable()
    bundled_voice_model = _bundled_piper_voice_model()
    bundled_voice_config = _bundled_piper_voice_config()
    return {
        "model_path": bundled_model or os.path.join(celeste_home, "models", "default.gguf"),
        "embedding_model": bundled_embedding or os.path.join(celeste_home, "embeddings", "e5-small-v2"),
        "llama_server_executable": bundled_llama or os.path.join(
            _project_root(),
            "vendor",
            "llama.cpp",
            "build",
            "bin",
            "llama-server.exe" if os.name == "nt" else "llama-server",
        ),
        "data_dir": os.path.join(celeste_home, "data"),
        "persist_dir": os.path.join(celeste_home, "chroma"),
        "file_rag_dir": os.path.join(celeste_home, "library"),
        "tts_piper_model": bundled_voice_model or os.path.join(celeste_home, "voices", "voice.onnx"),
        "tts_piper_config": bundled_voice_config or os.path.join(celeste_home, "voices", "voice.onnx.json"),
        "tts_piper_executable": bundled_piper or "piper",
    }


def _find_first_matching(root: str, patterns: list[str], *, want_dir: bool = False) -> str | None:
    if not os.path.isdir(root):
        return None
    for dirpath, dirnames, filenames in os.walk(root):
        names = dirnames if want_dir else filenames
        for pattern in patterns:
            for name in sorted(names):
                if fnmatch.fnmatch(name.lower(), pattern.lower()):
                    return os.path.join(dirpath, name)
    return None


def _bundled_default_model_path() -> str | None:
    bundled_models_dir = os.path.join(_project_root(), "models")
    if not os.path.isdir(bundled_models_dir):
        return None
    gguf_files = sorted(
        os.path.join(bundled_models_dir, name)
        for name in os.listdir(bundled_models_dir)
        if name.lower().endswith(".gguf")
    )
    return gguf_files[0] if gguf_files else None


def _bundled_default_embedding_dir() -> str | None:
    bundled_embedding_dir = os.path.join(_project_root(), "embeddings", "e5-small-v2")
    return bundled_embedding_dir if os.path.isdir(bundled_embedding_dir) else None


def _bundled_llama_server_path() -> str | None:
    candidates = [
        os.path.join(_project_root(), "vendor", "llama.cpp", "build", "bin", "Release", "llama-server.exe"),
        os.path.join(_project_root(), "vendor", "llama.cpp", "build", "bin", "llama-server.exe"),
        os.path.join(_project_root(), "vendor", "llama.cpp", "build", "bin", "llama-server"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _bundled_piper_executable() -> str | None:
    search_root = os.path.join(_project_root(), "piper", "windows" if os.name == "nt" else "linux")
    if os.name == "nt":
        return _find_first_matching(search_root, ["piper.exe"])
    return _find_first_matching(search_root, ["piper"])


def _bundled_piper_voice_model() -> str | None:
    return _find_first_matching(os.path.join(_project_root(), "voices"), ["*.onnx"])


def _bundled_piper_voice_config() -> str | None:
    return _find_first_matching(os.path.join(_project_root(), "voices"), ["*.onnx.json"])


def _load_template(template_path: str | None = None) -> dict:
    with open(template_path or _template_path(), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class SetupWizardDialog(QDialog):
    def __init__(self, config_path: str, template_path: str | None = None, parent=None):
        super().__init__(parent)
        self.config_path = os.path.abspath(config_path)
        self.template = _load_template(template_path)
        self.defaults = _default_paths()
        self.inputs: dict[str, QLineEdit] = {}
        self._engine_widget: EngineSetupWidget | None = None
        self.setWindowTitle("Celeste First-Run Setup")
        icon_path = resource_path("assets", "celeste_icon.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.resize(900, 800)
        self._build_ui()

    # Called by EngineSetupWidget when download/compile finishes successfully
    def on_engine_ready(self):
        self.save_button.setEnabled(True)
        self.validate_button.setEnabled(True)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Celeste Setup")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(title)

        # On packaged Linux builds, embed the engine setup widget first
        show_engine_setup = _needs_engine_setup()
        self._show_engine_setup = show_engine_setup
        if show_engine_setup:
            self._engine_widget = EngineSetupWidget(self)
            layout.addWidget(self._engine_widget)

            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setFrameShadow(QFrame.Sunken)
            layout.addWidget(sep)

            step2_label = QLabel("<b>Step 2 — Configure paths</b>")
            layout.addWidget(step2_label)
        else:
            intro = QLabel(
                "Choose local model, embedding, document, and data paths. "
                "You can keep the suggested default folders or browse to your own."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)

        use_defaults = QPushButton("Use Default Celeste Folders")
        use_defaults.clicked.connect(self._apply_default_paths)
        layout.addWidget(use_defaults, alignment=Qt.AlignLeft)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form_layout = QVBoxLayout(content)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)

        self.inputs["model_path"] = self._add_path_row(form, "GGUF Model", self.defaults["model_path"], "file")
        self.inputs["embedding_model"] = self._add_path_row(
            form, "Embedding Model", self.defaults["embedding_model"], "directory",
        )

        # Only show llama-server path on non-packaged Linux builds; packaged
        # Linux handles it automatically via EngineSetupWidget.
        if not show_engine_setup:
            self.inputs["llama_server_executable"] = self._add_path_row(
                form, "llama-server", self.defaults["llama_server_executable"], "file",
            )

        self.inputs["data_dir"] = self._add_path_row(form, "Data Directory", self.defaults["data_dir"], "directory")
        self.inputs["persist_dir"] = self._add_path_row(
            form, "Vector DB Directory", self.defaults["persist_dir"], "directory",
        )
        self.inputs["file_rag_dir"] = self._add_path_row(
            form, "Document Library", self.defaults["file_rag_dir"], "directory",
        )

        self.tts_enabled = QCheckBox("Enable Piper TTS")
        self.tts_enabled.setChecked(False)
        form.addRow("Speech", self.tts_enabled)

        self.inputs["tts_piper_executable"] = self._add_path_row(
            form, "Piper Executable", self.defaults["tts_piper_executable"], "file",
        )
        self.inputs["tts_piper_model"] = self._add_path_row(
            form, "Piper Voice Model", self.defaults["tts_piper_model"], "file",
        )
        self.inputs["tts_piper_config"] = self._add_path_row(
            form, "Piper Voice Config", self.defaults["tts_piper_config"], "file",
        )

        form_layout.addLayout(form)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        self.validation_view = QTextBrowser()
        self.validation_view.setFixedHeight(160)
        self.validation_view.setPlaceholderText("Validation output will appear here.")
        layout.addWidget(self.validation_view)

        button_row = QHBoxLayout()
        self.validate_button = QPushButton("Validate")
        self.save_button = QPushButton("Save && Launch")
        self.cancel_button = QPushButton("Cancel")
        self.validate_button.clicked.connect(self._validate_current_settings)
        self.save_button.clicked.connect(self._save_config)
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.validate_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.cancel_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        # On packaged Linux, disable Save until the engine setup completes
        if show_engine_setup:
            self.save_button.setEnabled(False)
            self.validate_button.setEnabled(False)

    def _add_path_row(self, form: QFormLayout, label: str, value: str, mode: str) -> QLineEdit:
        edit = QLineEdit(value)
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda _, target=edit, browse_mode=mode: self._browse_path(target, browse_mode))
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(browse)
        form.addRow(label, row)
        return edit

    def _browse_path(self, target: QLineEdit, mode: str) -> None:
        start = target.text().strip() or os.path.expanduser("~")
        if mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, "Choose File", start)
        else:
            path = QFileDialog.getExistingDirectory(self, "Choose Directory", start)
        if path:
            target.setText(path)

    def _apply_default_paths(self) -> None:
        for key, value in self.defaults.items():
            if key in self.inputs:
                self.inputs[key].setText(value)

    def _collect_config(self) -> dict:
        cfg = copy.deepcopy(self.template)
        cfg["model_path"] = self.inputs["model_path"].text().strip()
        cfg["embedding_model"] = self.inputs["embedding_model"].text().strip()
        if "llama_server_executable" in self.inputs:
            cfg["llama_server_executable"] = self.inputs["llama_server_executable"].text().strip()
        else:
            # Packaged Linux: engine setup widget owns this; use the install path.
            cfg["llama_server_executable"] = self.defaults["llama_server_executable"]
        cfg["data_dir"] = self.inputs["data_dir"].text().strip()
        cfg["persist_dir"] = self.inputs["persist_dir"].text().strip()
        file_rag_dir = self.inputs["file_rag_dir"].text().strip()
        cfg["file_rag_dirs"] = [file_rag_dir] if file_rag_dir else []
        cfg["tts_enabled"] = self.tts_enabled.isChecked()
        cfg["tts_backend"] = "piper"
        cfg["tts_piper_executable"] = self.inputs["tts_piper_executable"].text().strip()
        cfg["tts_piper_model"] = self.inputs["tts_piper_model"].text().strip()
        cfg["tts_piper_config"] = self.inputs["tts_piper_config"].text().strip()
        cfg["tts_output_dir"] = None
        return cfg

    def _write_config(self) -> None:
        cfg = self._collect_config()
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        for key in ("data_dir", "persist_dir", "file_rag_dirs"):
            if key == "file_rag_dirs":
                for directory in cfg.get("file_rag_dirs", []) or []:
                    normalized = os.path.expandvars(os.path.expanduser(str(directory or "")))
                    if normalized:
                        os.makedirs(normalized, exist_ok=True)
                continue
            directory = os.path.expandvars(os.path.expanduser(str(cfg.get(key, "") or "")))
            if directory:
                os.makedirs(directory, exist_ok=True)

        model_path = os.path.expandvars(os.path.expanduser(str(cfg.get("model_path", "") or "")))
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            if not os.path.isfile(model_path):
                bundled_model = _bundled_default_model_path()
                if bundled_model and os.path.isfile(bundled_model):
                    shutil.copy2(bundled_model, model_path)

        embedding_dir = os.path.expandvars(os.path.expanduser(str(cfg.get("embedding_model", "") or "")))
        if embedding_dir:
            os.makedirs(os.path.dirname(embedding_dir), exist_ok=True)
            if not os.path.isdir(embedding_dir):
                bundled_embedding = _bundled_default_embedding_dir()
                if bundled_embedding and os.path.isdir(bundled_embedding):
                    shutil.copytree(bundled_embedding, embedding_dir)

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    def _validate_current_settings(self) -> bool:
        self._write_config()
        try:
            checks = validate_config_file(self.config_path)
        except Exception as exc:
            self.validation_view.setPlainText(f"[ERROR] config.yaml: {exc}")
            return False
        # On packaged Linux the engine widget manages llama-server; suppress
        # any residual error for that subject so it doesn't block the wizard.
        if getattr(self, "_show_engine_setup", False):
            checks = [c for c in checks if c.subject != "llama_server_executable"]
        rendered = format_checks(checks)
        self.validation_view.setPlainText(rendered)
        return not has_errors(checks)

    def _save_config(self) -> None:
        valid = self._validate_current_settings()
        if valid:
            QMessageBox.information(self, "Celeste Setup", f"Configuration saved to:\n{self.config_path}")
            self.accept()
            return
        reply = QMessageBox.question(
            self,
            "Save With Validation Errors?",
            "Validation reported one or more errors. Save config.yaml anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.accept()


def ensure_config_with_wizard(config_path: str, *, app: QApplication | None = None, parent=None) -> bool:
    normalized = os.path.abspath(config_path)
    if os.path.isfile(normalized):
        return True
    dialog = SetupWizardDialog(normalized, parent=parent)
    return dialog.exec() == QDialog.Accepted


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Celeste config.yaml with a first-run setup wizard.")
    parser.add_argument(
        "--config",
        default=default_config_path(),
        help="Path where config.yaml should be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Open the wizard even if the config file already exists.",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if os.path.exists(config_path) and not args.force:
        print(f"Config already exists: {config_path}")
        return 0

    app = QApplication.instance() or QApplication(sys.argv)
    icon_path = resource_path("assets", "celeste_icon.png")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    dialog = SetupWizardDialog(config_path)
    return 0 if dialog.exec() == QDialog.Accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
