# Celeste

A fully local, privacy-first desktop AI assistant. Celeste runs entirely on your own hardware — no cloud, no subscriptions, no data leaving your machine.

---

## Runs Completely Offline

Every inference, embedding, and memory operation happens locally. Internet access is never required after setup.

- Powered by [llama.cpp](https://github.com/ggerganov/llama.cpp) via `llama_cpp` Python bindings or an embedded `llama-server` process
- Supports any GGUF-format model (Mistral, LLaMA, Qwen, DeepSeek, Phi, and others)
- CUDA GPU offloading configurable per layer (`n_gpu_layers`); falls back to CPU transparently
- Three selectable backends: `llama_cpp` (in-process), `llama_server` (subprocess), `transformers` (HuggingFace)

---

## Remembers Who You Are

Celeste builds a persistent picture of the user across sessions and uses it to ground every response.

- **Episodic memory** — conversation turns stored and retrieved from a local [Chroma](https://www.trychroma.com/) vector database using sentence-transformers embeddings
- **Graph memory** — SQLite-backed fact graph (subject → predicate → object) populated at startup with runtime and library facts; injected into the system prompt as structured context
- **Behavioral playbook** — persistent rules updated after every session via an async background reflection pass; loaded into the system preamble on next launch
- Memory and graph writes run in background threads to avoid blocking startup

---

## Knows Your Documents

Point Celeste at any folder of local files and it retrieves relevant content before answering.

- **Hybrid retrieval** — TF-IDF lexical index (always available, zero-latency) combined with a semantic deep index (sentence-transformer embeddings, built on demand)
- Deep index supports multi-GPU encoding (`file_rag_multi_gpu`) for large libraries
- Deep index is lazy-loaded and pre-warmed in a background thread at startup so the first query is fast
- Per-directory file counts exposed in the settings UI

---

## Speaks Back (Optional)

Local text-to-speech using [Piper](https://github.com/rhasspy/piper) with selectable voice models.

- Piper executable and voice model paths configurable from the settings panel
- Voice model browser built into the UI
- TTS is opt-in and adds zero latency when disabled

---

## Fully Configurable from the UI

No config file editing required for day-to-day use.

- Settings panel covers model selection, context window, GPU layers, embedding model, document directories, TTS, and persona
- First-run setup wizard auto-detects bundled models, embeddings, llama-server, and Piper executables
- Token usage progress bar color-coded green/amber/red; warns when context window approaches capacity
- Persona editor modal for customizing the system preamble
- Conversation export to plain text

---

## Streams Responses Live

Tokens appear as they are generated — no waiting for the full response.

- `Agent.respond()` accepts an optional `token_cb` callback; `model_runner.py` yields tokens via `stream=True`
- Live preview rendered in a `QPlainTextEdit` frame that hides on completion
- Markdown rendered to HTML inline (bold, italic, headers, inline code, bullet lists)

---

## Cross-Platform

- **Windows** — PyInstaller bundle + Inno Setup installer (`Celeste-Setup-0.1.0.exe`)
- **Linux** — PyInstaller bundle + `.tar.gz` + AppImage
- Platform-aware config paths: `%LOCALAPPDATA%\Celeste` on Windows, `~/.config/Celeste` on Linux when packaged

---

## Tech Stack

| Layer | Technology |
|---|---|
| GUI | Python, PySide6 (Qt6) |
| Inference | llama.cpp, llama_cpp Python bindings |
| Embeddings | sentence-transformers, PyTorch |
| Vector memory | Chroma |
| Graph memory | SQLite |
| Lexical retrieval | scikit-learn TF-IDF |
| TTS | Piper |
| Config | Pydantic, PyYAML |
| Packaging | PyInstaller, Inno Setup, AppImageTool |

---

## License

Copyright (c) 2025 Jeremy Findley. All rights reserved.
Source code is available for viewing and evaluation only. See [LICENSE](LICENSE).
