# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


PROJECT_ROOT = Path(SPECPATH).resolve().parents[1]

# Packages whose collected files we strip from the bundle entirely.
# nvidia-* packages ship ~4 GB of CUDA runtime; triton ships ~640 MB.
# llama-server handles GPU inference out-of-process and brings its own CUDA
# libraries, so the app itself does not need them at runtime.
_STRIP_PREFIXES = ("nvidia", "triton")

def _is_stripped(path: str) -> bool:
    parts = Path(path).parts
    return any(
        part.lower() == prefix or part.lower().startswith(prefix + "_")
        for part in parts
        for prefix in _STRIP_PREFIXES
    )


datas = [
    (str(PROJECT_ROOT / "config.example.yaml"), "."),
    (str(PROJECT_ROOT / "assets" / "celeste_icon.png"), "assets"),
    (str(PROJECT_ROOT / "assets" / "celeste_icon.ico"), "assets"),
]
binaries = []
hiddenimports = []

for package in (
    "chromadb",
    "sentence_transformers",
    "sklearn",
    "transformers",
    "tokenizers",
    "huggingface_hub",
):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# Strip nvidia and triton bloat from collected files
datas = [(src, dst) for src, dst in datas if not _is_stripped(src)]
binaries = [(src, dst) for src, dst in binaries if not _is_stripped(src)]


a = Analysis(
    [str(PROJECT_ROOT / "desktop_app.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "llama_cpp",
        "triton",
        # All nvidia CUDA runtime packages pulled in transitively by torch.
        # llama-server ships its own CUDA runtime; the app itself doesn't need these.
        "nvidia",
        "nvidia.cublas",
        "nvidia.cuda_cupti",
        "nvidia.cuda_nvrtc",
        "nvidia.cuda_runtime",
        "nvidia.cudnn",
        "nvidia.cufft",
        "nvidia.cufile",
        "nvidia.curand",
        "nvidia.cusolver",
        "nvidia.cusparse",
        "nvidia.cusparselt",
        "nvidia.nccl",
        "nvidia.nvjitlink",
        "nvidia.nvshmem",
        "nvidia.nvtx",
    ],
    noarchive=False,
    optimize=0,
)

# Post-analysis strip: remove nvidia and triton files collected by built-in hooks.
# The excludes[] list only prevents Python module import graph traversal; binary
# collection hooks fire regardless.  Filtering a.binaries / a.datas here is the
# only reliable way to keep them out of the bundle.
a.binaries = TOC([
    (name, src, typ)
    for name, src, typ in a.binaries
    if not _is_stripped(name) and not _is_stripped(src or "")
])
a.datas = TOC([
    (name, src, typ)
    for name, src, typ in a.datas
    if not _is_stripped(name) and not _is_stripped(src or "")
])

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Celeste",
    icon=str(PROJECT_ROOT / "assets" / "celeste_icon.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Celeste",
)
