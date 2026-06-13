"""
llama_installer.py — Hardware-adaptive llama-server setup for Linux.

Detects NVIDIA GPU + CUDA version, downloads the best matching pre-built
llama.cpp release from GitHub, falls back to compiling from source when
no pre-built matches, and installs everything to ~/.local/share/Celeste/llama/.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, NamedTuple

log = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"


# ---------------------------------------------------------------------------
# Install location
# ---------------------------------------------------------------------------

def llama_install_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / "Celeste" / "llama"


def installed_llama_server() -> Path | None:
    p = llama_install_dir() / "llama-server"
    return p if p.is_file() else None


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

class HardwareInfo(NamedTuple):
    has_nvidia: bool
    cuda_version: tuple[int, int] | None  # (major, minor) from driver; None = no CUDA
    cpu_flags: frozenset[str]

    def describe(self) -> str:
        if self.cuda_version:
            return f"NVIDIA GPU detected, CUDA {self.cuda_version[0]}.{self.cuda_version[1]}"
        if self.has_nvidia:
            return "NVIDIA GPU detected (CUDA version unknown — will use CPU build)"
        return "No NVIDIA GPU detected — CPU build will be used"


def _cpu_flags() -> frozenset[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return frozenset(line.split(":", 1)[1].split())
    except OSError:
        pass
    return frozenset()


def _cuda_from_nvidia_smi() -> tuple[int, int] | None:
    try:
        summary = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.DEVNULL, timeout=10
        ).decode()
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", summary)
        if m:
            return int(m.group(1)), int(m.group(2))
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    return None


def _cuda_from_nvcc() -> tuple[int, int] | None:
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.DEVNULL, timeout=10
        ).decode()
        m = re.search(r"release (\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    return None


def _cuda_from_version_file() -> tuple[int, int] | None:
    for path in ("/usr/local/cuda/version.json", "/usr/local/cuda/version.txt"):
        try:
            with open(path) as f:
                content = f.read()
            m = re.search(r'"cuda"\s*:\s*"(\d+)\.(\d+)', content) or \
                re.search(r"CUDA Version (\d+)\.(\d+)", content)
            if m:
                return int(m.group(1)), int(m.group(2))
        except OSError:
            pass
    return None


def detect_hardware() -> HardwareInfo:
    if platform.system() != "Linux":
        return HardwareInfo(has_nvidia=False, cuda_version=None, cpu_flags=_cpu_flags())

    cuda = _cuda_from_nvidia_smi() or _cuda_from_nvcc() or _cuda_from_version_file()
    has_nvidia = cuda is not None or bool(shutil.which("nvidia-smi"))
    return HardwareInfo(has_nvidia=has_nvidia, cuda_version=cuda, cpu_flags=_cpu_flags())


# ---------------------------------------------------------------------------
# GitHub release fetching
# ---------------------------------------------------------------------------

class ReleaseAsset(NamedTuple):
    name: str
    download_url: str
    size_bytes: int
    cuda_version: tuple[int, int] | None  # None = CPU-only build


def _parse_cuda_from_name(name: str) -> tuple[int, int] | None:
    m = re.search(r"cu(\d+)\.(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def fetch_release_assets() -> tuple[str, list[ReleaseAsset]]:
    """Return (release_tag, linux_x64_assets) from the latest llama.cpp release."""
    req = urllib.request.Request(
        GITHUB_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Celeste-Installer/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    tag: str = data.get("tag_name", "unknown")
    assets: list[ReleaseAsset] = []
    for a in data.get("assets", []):
        name: str = a["name"]
        if not name.endswith("-x64.zip"):
            continue
        if "ubuntu" not in name and "linux" not in name.lower():
            continue
        cuda = _parse_cuda_from_name(name) if "cuda" in name else None
        # Skip CUDA assets where we can't parse the version
        if "cuda" in name and cuda is None:
            continue
        assets.append(ReleaseAsset(
            name=name,
            download_url=a["browser_download_url"],
            size_bytes=a.get("size", 0),
            cuda_version=cuda,
        ))
    return tag, assets


def select_best_asset(assets: list[ReleaseAsset], hw: HardwareInfo) -> ReleaseAsset | None:
    """Return the best pre-built asset for the given hardware."""
    if hw.cuda_version:
        cuda_assets = [a for a in assets if a.cuda_version is not None]
        compatible = [a for a in cuda_assets if a.cuda_version <= hw.cuda_version]
        if compatible:
            return max(compatible, key=lambda a: a.cuda_version)  # type: ignore[return-value]

    cpu_assets = [a for a in assets if a.cuda_version is None]
    return cpu_assets[0] if cpu_assets else None


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------

ProgressCb = Callable[[int, int], None]  # (downloaded_bytes, total_bytes)


def _download(url: str, dest: Path, progress_cb: ProgressCb | None = None) -> None:
    with urllib.request.urlopen(url, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_cb and total:
                    progress_cb(downloaded, total)


def _extract_server_files(zip_path: Path, install_dir: Path) -> None:
    install_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            basename = os.path.basename(member)
            if not basename:
                continue
            if basename == "llama-server" or basename.endswith(".so") or ".so." in basename:
                target = install_dir / basename
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    server = install_dir / "llama-server"
    if server.exists():
        server.chmod(0o755)


def install_from_asset(
    asset: ReleaseAsset,
    install_dir: Path,
    progress_cb: ProgressCb | None = None,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="celeste-llama-") as tmp:
        zip_path = Path(tmp) / asset.name
        log.info("Downloading %s (%.0f MB)…", asset.name, asset.size_bytes / 1_048_576)
        _download(asset.download_url, zip_path, progress_cb)
        log.info("Extracting llama-server…")
        _extract_server_files(zip_path, install_dir)

    server = install_dir / "llama-server"
    if not server.is_file():
        raise RuntimeError("llama-server not found in downloaded archive")
    log.info("Installed llama-server → %s", server)
    return server


# ---------------------------------------------------------------------------
# Compile fallback
# ---------------------------------------------------------------------------

LogCb = Callable[[str], None]


def check_build_deps(cuda: bool) -> list[str]:
    missing = [t for t in ("git", "cmake", "gcc", "g++") if not shutil.which(t)]
    if cuda and not shutil.which("nvcc"):
        missing.append("nvcc  (CUDA toolkit — install nvidia-cuda-toolkit)")
    return missing


def compile_llama_server(
    install_dir: Path,
    cuda: bool,
    log_cb: LogCb | None = None,
) -> Path:
    missing = check_build_deps(cuda)
    if missing:
        raise RuntimeError(
            "Missing build tools: " + ", ".join(missing) + "\n"
            "Install them then retry, or download a pre-built binary manually."
        )

    with tempfile.TemporaryDirectory(prefix="celeste-llama-build-") as tmp:
        src = Path(tmp) / "llama.cpp"
        _run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(src)], log_cb)

        build = src / "build"
        cmake_args = [
            "cmake", "-B", str(build), "-S", str(src),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        if cuda:
            cmake_args.append("-DGGML_CUDA=ON")
        _run(cmake_args, log_cb)
        _run(
            ["cmake", "--build", str(build), "--target", "llama-server",
             f"-j{os.cpu_count() or 4}"],
            log_cb,
        )

        bin_dir = build / "bin"
        server_src = bin_dir / "llama-server"
        if not server_src.exists():
            # Some cmake setups put it directly in build/
            server_src = build / "llama-server"
        if not server_src.exists():
            raise RuntimeError("Compile succeeded but llama-server binary not found")

        install_dir.mkdir(parents=True, exist_ok=True)
        for f in bin_dir.iterdir():
            if f.name == "llama-server" or f.suffix == ".so" or ".so." in f.name:
                shutil.copy2(f, install_dir / f.name)
        if not (install_dir / "llama-server").exists():
            shutil.copy2(server_src, install_dir / "llama-server")

        server = install_dir / "llama-server"
        server.chmod(0o755)
        return server


def _run(cmd: list[str], log_cb: LogCb | None) -> None:
    log.info("$ %s", " ".join(cmd))
    if log_cb:
        log_cb("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout
    for line in proc.stdout:
        line = line.rstrip()
        log.debug(line)
        if log_cb:
            log_cb(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def ensure_llama_server(
    progress_cb: ProgressCb | None = None,
    log_cb: LogCb | None = None,
) -> Path:
    """
    Detect hardware, download or compile the right llama-server, install it,
    and return the path to the binary.  Raises RuntimeError on failure.
    """
    install_dir = llama_install_dir()
    hw = detect_hardware()
    log.info("Hardware: %s", hw.describe())
    if log_cb:
        log_cb(hw.describe())

    try:
        if log_cb:
            log_cb("Fetching latest llama.cpp release info…")
        tag, assets = fetch_release_assets()
        asset = select_best_asset(assets, hw)
        if asset:
            size_mb = asset.size_bytes / 1_048_576
            log.info("Selected: %s (%.0f MB)", asset.name, size_mb)
            if log_cb:
                log_cb(f"Downloading {asset.name}  ({size_mb:.0f} MB)…")
            return install_from_asset(asset, install_dir, progress_cb)
        log.warning("No matching pre-built found for CUDA %s; falling back to compile", hw.cuda_version)
        if log_cb:
            log_cb("No matching pre-built found; compiling from source…")
    except (urllib.error.URLError, OSError) as exc:
        log.warning("Download failed (%s); falling back to compile", exc)
        if log_cb:
            log_cb(f"Download failed: {exc}\nFalling back to compile from source…")

    return compile_llama_server(install_dir, cuda=bool(hw.cuda_version), log_cb=log_cb)


# ---------------------------------------------------------------------------
# GUI dialog (PySide6)
# ---------------------------------------------------------------------------

def run_installer_dialog(app=None) -> bool:
    """
    Show the llama-server setup dialog.  Returns True if llama-server was
    successfully installed, False if the user cancelled or installation failed.
    """
    try:
        from PySide6.QtCore import QObject, QThread, Signal, Slot
        from PySide6.QtWidgets import (
            QDialog, QDialogButtonBox, QLabel, QPlainTextEdit,
            QProgressBar, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy,
        )
    except ImportError:
        log.error("PySide6 not available; cannot show installer dialog")
        return False

    hw = detect_hardware()

    try:
        tag, assets = fetch_release_assets()
        asset = select_best_asset(assets, hw)
    except Exception:
        asset = None
        tag = "unknown"

    class Worker(QObject):
        progress = Signal(int, int)   # downloaded, total
        log_line = Signal(str)
        finished = Signal(bool, str)  # success, message

        def run(self):
            try:
                def pcb(d, t):
                    self.progress.emit(d, t)
                def lcb(line):
                    self.log_line.emit(line)
                ensure_llama_server(progress_cb=pcb, log_cb=lcb)
                self.finished.emit(True, "")
            except Exception as exc:
                self.finished.emit(False, str(exc))

    class InstallerDialog(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Celeste — AI Engine Setup")
            self.setMinimumWidth(520)
            self._success = False

            layout = QVBoxLayout(self)
            layout.setSpacing(12)

            # Hardware info
            hw_label = QLabel(f"<b>Hardware:</b> {hw.describe()}")
            hw_label.setWordWrap(True)
            layout.addWidget(hw_label)

            # What will be downloaded/compiled
            if asset:
                size_mb = asset.size_bytes / 1_048_576
                action_text = (
                    f"<b>Action:</b> Download <code>{asset.name}</code>"
                    f" ({size_mb:.0f} MB) from the latest llama.cpp release ({tag})"
                )
            else:
                missing = check_build_deps(bool(hw.cuda_version))
                if missing:
                    action_text = (
                        "<b>Action:</b> Compile from source<br>"
                        f"<span style='color:orange'>Missing build tools: {', '.join(missing)}</span>"
                    )
                else:
                    action_text = "<b>Action:</b> Compile from source (no pre-built matched your hardware)"
            action_label = QLabel(action_text)
            action_label.setWordWrap(True)
            layout.addWidget(action_label)

            # Progress bar
            self._progress = QProgressBar()
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._progress.setVisible(False)
            layout.addWidget(self._progress)

            # Log output
            self._log = QPlainTextEdit()
            self._log.setReadOnly(True)
            self._log.setMaximumHeight(160)
            self._log.setVisible(False)
            layout.addWidget(self._log)

            # Status label
            self._status = QLabel("Click 'Set Up' to begin.")
            self._status.setWordWrap(True)
            layout.addWidget(self._status)

            # Buttons
            btn_row = QHBoxLayout()
            self._setup_btn = QPushButton("Set Up")
            self._setup_btn.setDefault(True)
            self._skip_btn = QPushButton("Skip (launch without AI engine)")
            btn_row.addWidget(self._setup_btn)
            btn_row.addWidget(self._skip_btn)
            layout.addLayout(btn_row)

            self._setup_btn.clicked.connect(self._start_install)
            self._skip_btn.clicked.connect(self.reject)

            self._thread: QThread | None = None
            self._worker: Worker | None = None

        def _start_install(self):
            self._setup_btn.setEnabled(False)
            self._skip_btn.setEnabled(False)
            self._progress.setVisible(True)
            self._log.setVisible(True)
            self._status.setText("Setting up AI engine…")

            self._worker = Worker()
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
            mb_done = downloaded / 1_048_576
            mb_total = total / 1_048_576
            self._status.setText(f"Downloading… {mb_done:.1f} / {mb_total:.1f} MB")

        @Slot(str)
        def _on_log(self, line: str):
            self._log.appendPlainText(line)
            self._log.verticalScrollBar().setValue(
                self._log.verticalScrollBar().maximum()
            )
            self._status.setText(line[:120])

        @Slot(bool, str)
        def _on_finished(self, success: bool, message: str):
            if self._thread:
                self._thread.quit()
                self._thread.wait()
            self._progress.setValue(100)
            if success:
                self._success = True
                self._status.setText("AI engine installed successfully.")
                done_btn = QPushButton("Launch Celeste")
                done_btn.clicked.connect(self.accept)
                layout = self.layout()
                layout.addWidget(done_btn)
                done_btn.setFocus()
            else:
                self._status.setText(f"Installation failed:\n{message}")
                self._setup_btn.setText("Retry")
                self._setup_btn.setEnabled(True)
                self._skip_btn.setEnabled(True)

        def succeeded(self) -> bool:
            return self._success

    dialog = InstallerDialog()
    result = dialog.exec()
    return dialog.succeeded()
