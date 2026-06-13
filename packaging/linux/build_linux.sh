#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt

rm -rf "$PROJECT_ROOT/dist/Celeste" "$PROJECT_ROOT/build/Celeste"
python -m PyInstaller packaging/pyinstaller/celeste.spec --noconfirm --clean

# llama-server is NOT bundled — Celeste detects hardware and installs the
# correct pre-built (or compiles from source) on first launch via llama_installer.py.

if [ -d "$PROJECT_ROOT/models" ]; then
  cp -a "$PROJECT_ROOT/models" "$PROJECT_ROOT/dist/Celeste/models"
fi

if [ -d "$PROJECT_ROOT/embeddings" ]; then
  cp -a "$PROJECT_ROOT/embeddings" "$PROJECT_ROOT/dist/Celeste/embeddings"
fi

if [ -d "$PROJECT_ROOT/piper/linux" ]; then
  mkdir -p "$PROJECT_ROOT/dist/Celeste/piper"
  cp -a "$PROJECT_ROOT/piper/linux" "$PROJECT_ROOT/dist/Celeste/piper/linux"
fi

if [ -d "$PROJECT_ROOT/voices" ]; then
  cp -a "$PROJECT_ROOT/voices" "$PROJECT_ROOT/dist/Celeste/voices"
fi

mkdir -p "$PROJECT_ROOT/dist/packages"
tar -C "$PROJECT_ROOT/dist" -czf "$PROJECT_ROOT/dist/packages/Celeste-linux-x86_64.tar.gz" Celeste

if command -v appimagetool >/dev/null 2>&1; then
  APPDIR="$PROJECT_ROOT/dist/Celeste.AppDir"
  rm -rf "$APPDIR"
  mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications"
  cp -a "$PROJECT_ROOT/dist/Celeste/." "$APPDIR/usr/bin/"
  cat > "$APPDIR/AppRun" <<'SH'
#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/usr/bin/Celeste" "$@"
SH
  chmod +x "$APPDIR/AppRun"
  cat > "$APPDIR/Celeste.desktop" <<'DESKTOP'
[Desktop Entry]
Name=Celeste
Exec=Celeste
Icon=celeste_icon
Type=Application
Categories=Utility;
DESKTOP
  cp "$PROJECT_ROOT/assets/celeste_icon.png" "$APPDIR/celeste_icon.png"
  cp "$APPDIR/Celeste.desktop" "$APPDIR/usr/share/applications/Celeste.desktop"
  appimagetool "$APPDIR" "$PROJECT_ROOT/dist/packages/Celeste-x86_64.AppImage"
else
  echo "appimagetool not found. Generated dist/packages/Celeste-linux-x86_64.tar.gz, but skipped AppImage creation."
fi
