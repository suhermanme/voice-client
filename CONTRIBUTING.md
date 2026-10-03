# Contributing

Thank you for improving the local voice assistant. Keep [`voice_assistant.py`](voice_assistant.py) as the canonical launcher and preserve the documented Indonesian/English voice pipeline.

## Set up the project

Install the locked Python environment from the repository root:

```bash
uv sync --frozen
```

Model files are intentionally excluded from Git. Follow the pinned model download and verification steps in [`README.md`](README.md) instead of adding binaries to a change.

## Make a change

- Keep runtime defaults and CLI parsing in `voice_client/config.py`.
- Keep status visualization and Qt signal definitions in `voice_client/ui.py`.
- Keep orchestration and the canonical `main()` entry point in `voice_assistant.py`.
- Preserve compatibility with OpenAI-compatible LLM services and the HTTP interface shared by Metal, CUDA, Vulkan, and CPU whisper.cpp builds.
- Add behavior-focused tests for fixes that can regress without real audio hardware or model services.

Run the required checks:

```bash
uv lock --check
uv run python -m py_compile voice_assistant.py voice_client/*.py
uv run python -m unittest discover -s tests -v
```

For changes to audio or inference behavior, also run the documented application manually with the local services and describe the tested hardware and backend in the pull request.

## Submit a pull request

Explain the concrete problem, the resulting behavior, and the validation performed. Do not include secrets, recordings, generated WAV files, caches, virtual environments, or model artifacts.
