Project: local voice assistant

Current canonical entry point:
- voice_assistant.py

Architecture:
- Silero VAD via ONNX Runtime
- whisper.cpp server on localhost:8081
- Qwen OpenAI-compatible server at 192.168.3.243:8080
- Supertonic local TTS
- PySide6 floating orb UI
- sounddevice for Core Audio
- uv for Python dependency management

Current functionality:
- automatic VAD / endpoint detection
- bilingual Indonesian + English STT
- streaming LLM
- bounded conversation context
- local TTS
- dynamic audio device enumeration
- --list-devices
- --mic and --speaker overrides
- orb states: idle/listening/hearing/thinking/speaking

Known-good environment:
- macOS on M4 Pro
- Python managed through uv

Do not remove working functionality.
Keep voice_assistant.py as the canonical launcher.