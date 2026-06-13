"""
llama_installer.py — Hardware-adaptive llama-server setup for Linux.

Detects NVIDIA GPU + CUDA version, downloads the best matching pre-built
llama.cpp release from GitHub, falls back to compiling from source when
no pre-built matches.

Install location:
  - Packaged app: <bundle>/vendor/llama.cpp/build/bin/  (self-contained)
  - Dev mode:     ~/.local/share/Celeste/llama/
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
import tarfile
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
    if getattr(sys, "frozen", False):
        # Packaged app — install into the bundle's own vendor tree so the app
        # is fully self-contained in whatever directory the user extracted it to.
        from app_paths import runtime_root
        return Path(runtime_root()) / "vendor" / "llama.cpp" / "build" / "bin"
    # Dev mode
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / "Celeste" / "llama"


def installed_llama_server() -> Path | None:
    p = llama_install_dir() / "llama-server"
    return p if p.is_file() else None


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

class GpuInfo(NamedTuple):
    vendor: str      # "nvidia" | "amd"
    name: str
    vram_mb: int


class HardwareInfo(NamedTuple):
    has_nvidia: bool
    cuda_version: tuple[int, int] | None  # (major, minor) from driver; None = no CUDA
    cpu_flags: frozenset[str]
    gpus: tuple[GpuInfo, ...]             # all detected GPUs with VRAM info
    has_rocm: bool                        # ROCm toolkit present (hipcc or rocm-smi)
    has_vulkan: bool                      # Vulkan runtime present

    def describe(self) -> str:
        nvidia = [g for g in self.gpus if g.vendor == "nvidia"]
        amd = [g for g in self.gpus if g.vendor == "amd"]
        if nvidia:
            # nvidia-smi already includes "NVIDIA" in the name — strip it to avoid doubling
            clean = [re.sub(r"^NVIDIA\s+", "", g.name, flags=re.IGNORECASE) for g in nvidia]
            # Collapse identical names: ["RTX 3060", "RTX 3060"] → "2× RTX 3060"
            if len(set(clean)) == 1:
                name_str = (f"{len(clean)}× " if len(clean) > 1 else "") + clean[0]
            else:
                name_str = ", ".join(clean)
            total_vram = sum(g.vram_mb for g in nvidia)
            cuda_str = f", CUDA {self.cuda_version[0]}.{self.cuda_version[1]}" if self.cuda_version else ""
            return f"NVIDIA {name_str}{cuda_str} ({total_vram} MB VRAM)"
        if amd:
            names = ", ".join(g.name for g in amd)
            total_vram = sum(g.vram_mb for g in amd)
            accel = "ROCm" if self.has_rocm else ("Vulkan" if self.has_vulkan else "no GPU accel")
            count = f"{len(amd)}× " if len(amd) > 1 else ""
            return f"{count}AMD {names} ({total_vram} MB VRAM, {accel})"
        return "No GPU detected — CPU inference"


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


def _enumerate_nvidia_gpus() -> list[GpuInfo]:
    """Return one GpuInfo per NVIDIA GPU using nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode()
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                name = parts[1]
                try:
                    vram_mb = int(parts[2])
                except ValueError:
                    vram_mb = 0
                gpus.append(GpuInfo(vendor="nvidia", name=name, vram_mb=vram_mb))
        return gpus
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []


def _enumerate_amd_gpus() -> list[GpuInfo]:
    """Return one GpuInfo per AMD GPU using rocm-smi or sysfs."""
    # Try rocm-smi first (most accurate for ROCm-capable cards)
    try:
        name_out = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--csv"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode()
        vram_out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode()
        names: list[str] = []
        for line in name_out.strip().splitlines():
            if line.startswith("card") or line.startswith("GPU"):
                parts = line.split(",")
                names.append(parts[-1].strip() if len(parts) > 1 else "AMD GPU")
        vrams: list[int] = []
        for line in vram_out.strip().splitlines():
            m = re.search(r"(\d+)", line)
            if m and (line.startswith("card") or line.startswith("GPU")):
                # rocm-smi reports VRAM in bytes
                vrams.append(int(m.group(1)) // (1024 * 1024))
        gpus = []
        for i, name in enumerate(names):
            vram = vrams[i] if i < len(vrams) else 0
            gpus.append(GpuInfo(vendor="amd", name=name, vram_mb=vram))
        if gpus:
            return gpus
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass

    # Fallback: sysfs vendor scan (detects card presence, VRAM via mem_info)
    gpus = []
    try:
        import glob
        for vendor_path in sorted(glob.glob("/sys/class/drm/card*/device/vendor")):
            try:
                vendor_id = open(vendor_path).read().strip()
                if vendor_id.lower() != "0x1002":  # AMD PCI vendor ID
                    continue
                card_dir = os.path.dirname(vendor_path)
                # Try to get product name
                name = "AMD GPU"
                label_path = os.path.join(card_dir, "product_name")
                if os.path.exists(label_path):
                    name = open(label_path).read().strip() or name
                # VRAM via mem_info_vram_total (bytes)
                vram_mb = 0
                vram_path = os.path.join(card_dir, "mem_info_vram_total")
                if os.path.exists(vram_path):
                    vram_mb = int(open(vram_path).read().strip()) // (1024 * 1024)
                gpus.append(GpuInfo(vendor="amd", name=name, vram_mb=vram_mb))
            except (OSError, ValueError):
                continue
    except Exception:
        pass
    return gpus


def _detect_rocm() -> bool:
    """True if the ROCm toolkit is installed (rocm-smi + hipcc reachable)."""
    if not shutil.which("rocm-smi"):
        return False
    # rocm-smi alone means the driver is present; hipcc means we can compile
    return (
        shutil.which("hipcc") is not None
        or os.path.isfile("/opt/rocm/bin/hipcc")
        or os.path.isfile("/usr/bin/hipcc")
    )


def _detect_vulkan() -> bool:
    """True if a Vulkan runtime is available."""
    if shutil.which("vulkaninfo"):
        try:
            subprocess.check_output(
                ["vulkaninfo", "--summary"],
                stderr=subprocess.DEVNULL, timeout=10,
            )
            return True
        except (subprocess.SubprocessError, OSError):
            pass
    # Fallback: check for the ICD loader library
    for lib in ("/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
                "/usr/lib/libvulkan.so.1",
                "/usr/local/lib/libvulkan.so.1"):
        if os.path.exists(lib):
            return True
    return False


def detect_hardware() -> HardwareInfo:
    if platform.system() != "Linux":
        return HardwareInfo(
            has_nvidia=False, cuda_version=None, cpu_flags=_cpu_flags(),
            gpus=(), has_rocm=False, has_vulkan=False,
        )

    cuda = _cuda_from_nvidia_smi() or _cuda_from_nvcc() or _cuda_from_version_file()
    nvidia_gpus = _enumerate_nvidia_gpus()
    has_nvidia = bool(nvidia_gpus) or cuda is not None or bool(shutil.which("nvidia-smi"))

    amd_gpus = _enumerate_amd_gpus()
    has_rocm = _detect_rocm()
    has_vulkan = _detect_vulkan()

    all_gpus = tuple(nvidia_gpus + amd_gpus)
    return HardwareInfo(
        has_nvidia=has_nvidia,
        cuda_version=cuda,
        cpu_flags=_cpu_flags(),
        gpus=all_gpus,
        has_rocm=has_rocm,
        has_vulkan=has_vulkan,
    )


def compute_gpu_config(hw: HardwareInfo) -> tuple[int, str | None]:
    """
    Return (n_gpu_layers, tensor_split_csv_or_None) for the detected hardware.
    n_gpu_layers=999 means "offload everything" — llama.cpp clips at actual layer count.
    tensor_split is a comma-separated VRAM-proportional string for multi-GPU setups.
    Falls back to (0, None) when no usable GPU acceleration is found.
    """
    nvidia = [g for g in hw.gpus if g.vendor == "nvidia"]
    amd = [g for g in hw.gpus if g.vendor == "amd"]

    if nvidia and hw.cuda_version:
        usable = nvidia
    elif amd and (hw.has_rocm or hw.has_vulkan):
        usable = amd
    else:
        return 0, None  # CPU fallback

    if len(usable) > 1:
        total_vram = sum(g.vram_mb for g in usable)
        if total_vram > 0:
            fracs = [g.vram_mb / total_vram for g in usable]
            return 999, ",".join(f"{f:.4f}" for f in fracs)

    return 999, None


# ---------------------------------------------------------------------------
# GitHub release fetching
# ---------------------------------------------------------------------------

class ReleaseAsset(NamedTuple):
    name: str
    download_url: str
    size_bytes: int
    cuda_version: tuple[int, int] | None  # None for non-CUDA builds
    asset_type: str                        # "cuda" | "vulkan" | "cpu"


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
        # Linux x86-64 tarballs (no arm64/s390x/sycl/openvino)
        if not name.endswith("-x64.tar.gz"):
            continue
        if "ubuntu" not in name and "linux" not in name.lower():
            continue
        if any(skip in name.lower() for skip in ("arm64", "s390x", "sycl", "openvino")):
            continue

        if "rocm" in name.lower():
            asset_type = "rocm"
        elif "vulkan" in name.lower():
            asset_type = "vulkan"
        else:
            asset_type = "cpu"

        assets.append(ReleaseAsset(
            name=name,
            download_url=a["browser_download_url"],
            size_bytes=a.get("size", 0),
            cuda_version=None,   # no CUDA-specific Linux pre-builts in recent releases
            asset_type=asset_type,
        ))
    return tag, assets


def select_best_asset(assets: list[ReleaseAsset], hw: HardwareInfo) -> ReleaseAsset | None:
    """
    Return the best pre-built asset for the detected hardware.

    Priority:
      AMD + ROCm runtime  → ROCm pre-built  (best AMD performance)
      AMD/NVIDIA + Vulkan → Vulkan build    (GPU-accelerated, no toolkit needed)
      anything else       → CPU build       (safe fallback)

    Note: llama.cpp no longer ships CUDA-specific Linux pre-builts; the Vulkan
    build covers NVIDIA via Vulkan compute and is the recommended download path.
    Compile-from-source with GGML_CUDA=ON remains available for users who need
    maximum CUDA performance.
    """
    nvidia = [g for g in hw.gpus if g.vendor == "nvidia"]
    amd = [g for g in hw.gpus if g.vendor == "amd"]

    # AMD with ROCm runtime installed → pre-built ROCm binary
    if amd and hw.has_rocm:
        rocm_assets = [a for a in assets if a.asset_type == "rocm"]
        if rocm_assets:
            return rocm_assets[0]

    # Any GPU-capable machine with Vulkan → Vulkan build
    if (nvidia or amd) and hw.has_vulkan:
        vulkan_assets = [a for a in assets if a.asset_type == "vulkan"]
        if vulkan_assets:
            return vulkan_assets[0]

    # CPU fallback
    cpu_assets = [a for a in assets if a.asset_type == "cpu"]
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


def _extract_server_files(archive_path: Path, install_dir: Path) -> None:
    """Extract llama-server and companion .so files from a .tar.gz or .zip archive."""
    install_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()

    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                basename = os.path.basename(member.name)
                if not basename:
                    continue
                if basename == "llama-server" or basename.endswith(".so") or ".so." in basename:
                    member.name = basename  # flatten into install_dir
                    tf.extract(member, install_dir, set_attrs=False)
    else:
        with zipfile.ZipFile(archive_path) as zf:
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
        archive_path = Path(tmp) / asset.name
        log.info("Downloading %s (%.0f MB)…", asset.name, asset.size_bytes / 1_048_576)
        _download(asset.download_url, archive_path, progress_cb)
        log.info("Extracting llama-server…")
        _extract_server_files(archive_path, install_dir)

    server = install_dir / "llama-server"
    if not server.is_file():
        raise RuntimeError("llama-server not found in downloaded archive")
    log.info("Installed llama-server → %s", server)
    return server


# ---------------------------------------------------------------------------
# Compile fallback
# ---------------------------------------------------------------------------

LogCb = Callable[[str], None]


def check_build_deps(cuda: bool, rocm: bool = False) -> list[str]:
    missing = [t for t in ("git", "cmake", "gcc", "g++") if not shutil.which(t)]
    if cuda and not shutil.which("nvcc"):
        missing.append("nvcc  (CUDA toolkit — install nvidia-cuda-toolkit)")
    if rocm:
        hipcc = shutil.which("hipcc") or os.path.isfile("/opt/rocm/bin/hipcc")
        if not hipcc:
            missing.append("hipcc  (ROCm toolkit — install rocm-hip-sdk)")
    return missing


def compile_llama_server(
    install_dir: Path,
    cuda: bool,
    log_cb: LogCb | None = None,
    rocm: bool = False,
) -> Path:
    missing = check_build_deps(cuda, rocm)
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
        elif rocm:
            cmake_args.append("-DGGML_HIP=ON")
            # Point cmake at the ROCm toolchain if hipcc isn't in PATH
            if not shutil.which("hipcc") and os.path.isfile("/opt/rocm/bin/hipcc"):
                cmake_args += [f"-DCMAKE_C_COMPILER=/opt/rocm/bin/hipcc",
                               f"-DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc"]
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

    amd_gpus = [g for g in hw.gpus if g.vendor == "amd"]
    use_rocm = bool(amd_gpus) and hw.has_rocm and not hw.cuda_version
    return compile_llama_server(
        install_dir, cuda=bool(hw.cuda_version), rocm=use_rocm, log_cb=log_cb,
    )


# ---------------------------------------------------------------------------
# PySide6 worker — used by setup_wizard.py's embedded setup section
# ---------------------------------------------------------------------------

class InstallerWorker:
    """
    Import this into setup_wizard.py to run ensure_llama_server in a QThread.
    Kept here so all installer logic stays in one module.
    """
    @staticmethod
    def make_qobject():
        from PySide6.QtCore import QObject, Signal

        class _Worker(QObject):
            progress = Signal(int, int)   # downloaded_bytes, total_bytes
            log_line = Signal(str)
            finished = Signal(bool, str)  # success, error_message

            def run(self):
                try:
                    ensure_llama_server(
                        progress_cb=lambda d, t: self.progress.emit(d, t),
                        log_cb=lambda line: self.log_line.emit(line),
                    )
                    self.finished.emit(True, "")
                except Exception as exc:
                    self.finished.emit(False, str(exc))

        return _Worker()
