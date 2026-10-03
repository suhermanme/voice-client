# Improvement roadmap

This roadmap tracks the code-review findings for the canonical `voice_assistant.py` application. Completed items retain the proven default models, endpoints, and behavior.

## Completed

- [x] Respect configured VAD durations. Convert milliseconds to whole audio chunks with ceiling division so pre-roll, minimum speech, and end silence are never shorter than configured.
- [x] Bound microphone buffering. Keep at most two seconds of queued input, retain the newest audio on overflow, and disable capture while STT, LLM, TTS, or playback is active.
- [x] Externalize runtime configuration. Support environment variables and CLI overrides for backend URLs, model IDs and paths, VAD timing, HTTP timeout, history size, API key, and audio devices while retaining proven defaults.
- [x] Reuse HTTP connections. Use one persistent `httpx.Client` for Whisper and LLM calls and close it during worker shutdown.
- [x] Make conversation updates transactional. Restore the exact prior history, including entries trimmed by the tentative user message, when the LLM fails or returns no content.

## Next

- [ ] Add startup validation for model files, backend health, accepted LLM model IDs, and audio sample-rate support.
- [ ] Improve cancellation and shutdown so in-flight backend requests can be interrupted cleanly.
- [ ] Replace console prints and broad exception handling with structured logging, useful tracebacks, and a visible UI error state.
- [ ] Reduce speech latency by feeding complete streamed response spans to TTS before the full LLM response finishes.
- [ ] Split implementation details into focused modules while keeping `voice_assistant.py` as the canonical launcher.
- [ ] Expand tests for device selection, language parsing, endpoint state transitions, HTTP/SSE behavior, and failure recovery.
- [ ] Add an explicit open-source license after the project owner chooses the license terms; optionally add contribution and issue templates.
