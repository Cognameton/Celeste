# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


PROJECT_ROOT = Path(SPECPATH).resolve().parents[1]

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


a = Analysis(
    [str(PROJECT_ROOT / "desktop_app.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["llama_cpp"],
    noarchive=False,
    optimize=0,
)

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
