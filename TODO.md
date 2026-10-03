# Improvement roadmap

This roadmap tracks the code-review findings for the canonical `voice_assistant.py` application. Completed items retain the proven default models, endpoints, and behavior.

## Completed

- [x] Respect configured VAD durations. Convert milliseconds to whole audio chunks with ceiling division so pre-roll, minimum speech, and end silence are never shorter than configured.
- [x] Bound microphone buffering. Keep at most two seconds of queued input, retain the newest audio on overflow, and disable capture while STT, LLM, TTS, or playback is active.
- [x] Externalize runtime configuration. Support environment variables and CLI overrides for backend URLs, model IDs and paths, VAD timing, HTTP timeout, history size, API key, and audio devices while retaining proven defaults.
- [x] Reuse HTTP connections. Use one persistent `httpx.Client` for Whisper and LLM calls and close it during worker shutdown.
- [x] Make conversation updates transactional. Restore the exact prior history, including entries trimmed by the tentative user message, when the LLM fails or returns no content.
- [x] Validate startup dependencies before capture. Check the VAD model file, required audio formats, whisper-server health, and the configured model reported by the LLM backend.
- [x] Make shutdown cancellation-aware. Signal active requests, close the shared HTTP client, stop audio playback, and wait for the voice worker to exit.
- [x] Add structured runtime logging and a visible red `ERROR` orb state while preserving recovery on the next utterance.
- [x] Stream complete language-tagged sentences to TTS while the remainder of the LLM response is still being generated.
- [x] Split configuration/CLI concerns and the PySide6 orb into focused `voice_client` modules while retaining `voice_assistant.py` as the canonical launcher.
- [x] Expand behavior-focused tests for device selection, language parsing, startup model validation, HTTP/SSE handling, cancellation, TTS state transitions, and conversation recovery.
- [x] Add contribution guidance and structured GitHub bug/feature issue forms.
- [x] Publish the project under the MIT License selected by the project owner.
- [x] Add configurable `orb` and radial audio-wave status styles, including native compositor dragging for Wayland.
- [x] Add a compact `spectrum-pill` style with state labels, animated bars, and live microphone-level response.
- [x] Reduce the Qt installation to PySide6 Essentials because the UI only uses Qt Core, Gui, and Widgets.

## Next

No review findings remain open. Add new entries here as the project evolves.
