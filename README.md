# Local bilingual voice assistant

A local, continuously listening Indonesian/English voice assistant with automatic endpoint detection, HTTP speech recognition, a streaming OpenAI-compatible language-model backend, local text-to-speech, and a selectable floating PySide6 status visualization.

[`voice_assistant.py`](voice_assistant.py) is the canonical application and launcher. The Python client is independent of the hardware backend used by `whisper-server`: Metal, CUDA, Vulkan, and CPU builds expose the same HTTP API, so changing the whisper.cpp build does not require changes to the application.

The source is split by responsibility while keeping that launch contract:

| Path | Responsibility |
| --- | --- |
| `voice_assistant.py` | Audio devices, VAD and inference orchestration, conversation flow, and canonical `main()` |
| `voice_client/config.py` | Proven defaults, environment variables, CLI validation, and immutable runtime configuration |
| `voice_client/ui.py` | PySide6 signals, selectable visual styles, state painting, dragging, and close behavior |
| `tests/test_voice_assistant.py` | Hardware-free regression tests using mocked audio and HTTP transports |

## Architecture

```text
microphone -> Python Silero VAD -> utterance WAV in memory
           -> whisper.cpp HTTP server (STT)
           -> OpenAI-compatible chat-completions server (LLM)
           -> complete tagged sentences -> Supertonic (local TTS) -> speaker

                    PySide6 floating status visualization
             listening / hearing / thinking / speaking / error
```

The Python process captures mono 16 kHz audio in 512-sample blocks. Its ONNX Silero model detects the start and end of each utterance. The completed utterance is posted as an in-memory WAV to the persistent whisper.cpp server at `http://127.0.0.1:8081/inference`. The transcript and bounded conversation history are then streamed from an OpenAI-compatible chat-completions endpoint. As complete sentences arrive inside `<id>` or `<en>` spans, Supertonic begins local synthesis at 44.1 kHz instead of waiting for the whole response.

Two VAD stages are intentional:

- `models/silero_vad.onnx` runs continuously in Python for live endpoint detection and includes 250 ms of pre-roll.
- `ggml-silero-v6.2.0.bin` is used by whisper.cpp when processing each completed utterance.

## Proven configuration

The currently proven profile is:

| Component | Model or service | Settings |
| --- | --- | --- |
| STT | whisper.cpp commit `60c0be6a`; `ggml-small.bin` (~487 MB) | Auto language detection; persistent server on `127.0.0.1:8081` |
| whisper.cpp VAD | `ggml-silero-v6.2.0.bin` (~0.88 MB) | Threshold `0.60`; minimum speech `250 ms`; minimum/end silence `650 ms`; speech padding `120 ms` |
| Python live VAD | `models/silero_vad.onnx` | Threshold `0.60`; configured minimum speech `250 ms`; configured end silence `650 ms`; configured pre-roll `250 ms`; ONNX Runtime CPU provider |
| LLM | `Qwen3-8B-Q5_K_M.gguf` via llama.cpp | OpenAI-compatible endpoint `http://192.168.3.243:8080/v1/chat/completions`; context `16384`; full GPU offload where supported; reasoning disabled |
| TTS | Supertonic voice `F1` | Local synthesis; Indonesian and English language spans |

The exact bilingual Whisper initial prompt is:

```text
Bahasa Indonesia and English mixed conversation. GPU, VRAM, CPU, Docker, Proxmox, Linux, macOS, server, utilization, memory, AI, LLM, model, coding, network.
```

The client sends `language=auto`, that prompt, and `carry_initial_prompt=true` with every transcription request.

Live audio arrives in 512-sample chunks, or 32 ms at 16 kHz. Duration thresholds round upward to complete chunks so they are never shorter than configured: the default 250 ms minimum speech and pre-roll use 256 ms, while 650 ms end silence uses 672 ms.

## Prerequisites

- macOS or Linux with a working microphone, speaker, and desktop session
- Python 3.13, as selected by `.python-version`
- [uv](https://docs.astral.sh/uv/) for the Python environment and locked dependencies
- Git, CMake, and a C/C++ compiler to build whisper.cpp
- PortAudio for `sounddevice`
- A running OpenAI-compatible chat-completions server and enough memory for its model

Typical system packages:

```bash
# macOS
brew install cmake portaudio

# Debian/Ubuntu
sudo apt update
sudo apt install build-essential cmake git curl libportaudio2 portaudio19-dev
```

On Linux, install the desktop libraries required by the PySide6 wheels for your distribution. CUDA and Vulkan builds also require the vendor driver/toolkit described below.

## Install the Python application

From this repository:

```bash
uv sync --frozen
```

`uv.lock` records the complete Python resolution. Supertonic downloads its own runtime assets on first use (`TTS(auto_download=True)`), so the first application run needs network access unless those assets are already cached.

Download the pinned Python VAD model without committing it:

```bash
mkdir -p models
curl -L \
  -o models/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/v6.2.1/src/silero_vad/data/silero_vad.onnx
echo "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3  models/silero_vad.onnx" | shasum -a 256 -c -
```

On Linux, use `sha256sum -c` in place of `shasum -a 256 -c` if needed.

## Build the pinned whisper.cpp server

Clone whisper.cpp outside this repository, check out the proven commit, and download both models:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
git checkout 60c0be6a
sh ./models/download-ggml-model.sh small ./models
sh ./models/download-vad-model.sh silero-v6.2.0 ./models
echo "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b  models/ggml-small.bin" | shasum -a 256 -c -
echo "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987  models/ggml-silero-v6.2.0.bin" | shasum -a 256 -c -
```

Choose one build. All four produce `build/bin/whisper-server` with the same HTTP interface.

### macOS with Metal

Metal is the proven M4 Pro backend. Enabling it explicitly makes the build intent reproducible:

```bash
cmake -S . -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target whisper-server
```

### Linux with NVIDIA CUDA

Install a compatible NVIDIA driver and CUDA toolkit first:

```bash
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target whisper-server
```

If CMake cannot infer the target GPU, add the appropriate `-DCMAKE_CUDA_ARCHITECTURES=<value>` for that machine. Do not copy an architecture value from another GPU without checking it.

### Linux with Vulkan

Install a Vulkan-capable driver, Vulkan headers, and `glslc` first:

```bash
cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target whisper-server
```

### CPU-only fallback

This portable build needs no GPU runtime:

```bash
cmake -S . -B build \
  -DGGML_METAL=OFF \
  -DGGML_CUDA=OFF \
  -DGGML_VULKAN=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target whisper-server
```

## Start whisper-server

From the pinned whisper.cpp checkout, the exact proven server command is:

```bash
./build/bin/whisper-server \
  --model ./models/ggml-small.bin \
  --host 127.0.0.1 \
  --port 8081 \
  --language auto \
  --prompt "Bahasa Indonesia and English mixed conversation. GPU, VRAM, CPU, Docker, Proxmox, Linux, macOS, server, utilization, memory, AI, LLM, model, coding, network." \
  --carry-initial-prompt \
  --vad \
  --vad-model ./models/ggml-silero-v6.2.0.bin \
  --vad-threshold 0.60 \
  --vad-min-speech-duration-ms 250 \
  --vad-min-silence-duration-ms 650 \
  --vad-speech-pad-ms 120
```

Keep this process running. A quick HTTP test, using any 16 kHz mono WAV, is:

```bash
curl http://127.0.0.1:8081/inference \
  -F file=@sample.wav \
  -F language=auto \
  -F response_format=json
```

## Start an LLM backend

The proven backend is a llama.cpp OpenAI-compatible server with `Qwen3-8B-Q5_K_M.gguf`, a 16,384-token context, all model layers offloaded to the GPU where supported, and Qwen reasoning disabled for lower latency. The endpoint seen by this client is:

```text
http://192.168.3.243:8080/v1/chat/completions
```

The exact llama.cpp commit and original launch command used by that machine have not yet been recorded. With a llama.cpp version that supports these options, the equivalent profile is:

```bash
./build/bin/llama-server \
  --model /path/to/Qwen3-8B-Q5_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 16384 \
  --n-gpu-layers 99 \
  --reasoning off
```

Check `llama-server --help` for the installed version before copying the command; reasoning controls have changed between llama.cpp revisions. The server should return the configured model ID from `GET /v1/models` and accept streamed `POST /v1/chat/completions` requests.

Any local backend that implements the OpenAI-compatible streaming chat-completions route can replace llama.cpp. Set `VOICE_LLM_URL` and `VOICE_LLM_MODEL`, or pass the matching CLI options; no audio, VAD, UI, or TTS code depends on llama.cpp. The Whisper URL is configurable in the same way.

## Run the assistant

Start the STT and LLM servers first, then run the canonical launcher from the repository root:

```bash
uv run python voice_assistant.py
```

Before opening the microphone, the launcher verifies the Python VAD model file, input and output audio formats, `whisper-server` health, and the configured model reported by the LLM backend's `/v1/models` route. A failed check is logged and leaves the visualization red so the cause is visible. Closing the visualization or pressing Ctrl-C signals active work to stop, closes the persistent HTTP client, stops audio playback, and waits briefly for the worker to exit.

On macOS, grant microphone access to the terminal application when prompted.

Audio device options:

```bash
# Show device indexes, capabilities, and system defaults
uv run python voice_assistant.py --list-devices

# Select devices by full name, unique partial name, or numeric index
uv run python voice_assistant.py --mic "USB Microphone" --speaker "USB Speakers"
uv run python voice_assistant.py --mic 2 --speaker 4
```

Without overrides, the app first looks for `MacBook Pro Microphone` and `MacBook Pro Speakers`, then falls back to the system defaults. No Core Audio device index is hard-coded.

### Runtime configuration

CLI arguments take precedence over environment-backed defaults. The main settings are:

| Environment variable | CLI option | Proven default |
| --- | --- | --- |
| `VOICE_WHISPER_URL` | `--whisper-url` | `http://127.0.0.1:8081/inference` |
| `VOICE_LLM_URL` | `--llm-url` | `http://192.168.3.243:8080/v1/chat/completions` |
| `VOICE_LLM_MODEL` | `--llm-model` | `Qwen3-8B-Q5_K_M.gguf` |
| `VOICE_VAD_MODEL` | `--vad-model` | Repository-local `models/silero_vad.onnx` |
| `VOICE_VAD_THRESHOLD` | `--vad-threshold` | `0.60` |
| `VOICE_END_SILENCE_MS` | `--end-silence-ms` | `650` |
| `VOICE_PRE_ROLL_MS` | `--pre-roll-ms` | `250` |
| `VOICE_MIN_SPEECH_MS` | `--min-speech-ms` | `250` |
| `VOICE_HTTP_TIMEOUT` | `--http-timeout` | `60` seconds |
| `VOICE_MAX_HISTORY_MESSAGES` | `--max-history-messages` | `20` |
| `VOICE_LOG_LEVEL` | `--log-level` | `INFO` |
| `VOICE_UI_STYLE` | `--ui-style` | `orb` |
| `VOICE_MIC` | `--mic` | Preferred device, then system default |
| `VOICE_SPEAKER` | `--speaker` | Preferred device, then system default |

`VOICE_LLM_API_KEY` adds a bearer token to LLM requests and is intentionally environment-only so it does not need to appear in a command line. `VOICE_PREFERRED_MIC` and `VOICE_PREFERRED_SPEAKER` change the preferred-device fallback names.

For example:

```bash
VOICE_LLM_URL=http://127.0.0.1:8000/v1/chat/completions \
VOICE_LLM_MODEL=my-local-model \
uv run python voice_assistant.py
```

## UI styles, states, and interaction

Three styles use the same state colors and require only PySide6 Essentials:

| Style | Behavior |
| --- | --- |
| `orb` | Default softly pulsing filled sphere |
| `circular-wave` | Expanding, fading rings around a smaller center; rings contract while thinking |
| `spectrum-pill` | Compact state label with animated bars; live microphone level drives the bars while hearing |

Select a style for one run:

```bash
uv run python voice_assistant.py --ui-style circular-wave
uv run python voice_assistant.py --ui-style spectrum-pill
```

Or make it the local default:

```bash
export VOICE_UI_STYLE=spectrum-pill
uv run python voice_assistant.py
```

| State | Meaning |
| --- | --- |
| `IDLE` | Initializing or stopping |
| `LISTENING` | Waiting for speech |
| `HEARING` | Speech is currently being captured |
| `THINKING` | Running STT, LLM generation, or TTS synthesis |
| `SPEAKING` | Playing synthesized audio |
| `ERROR` | Startup, STT, LLM, or TTS failed; inspect the terminal log |

The visualization stays above other windows and can be dragged. On Wayland it uses the compositor's native system-move operation. Close it or press Ctrl-C in the terminal for a clean shutdown.

## Conversation memory

The system prompt is always retained. The client then keeps at most 20 conversation messages, approximately 10 user/assistant exchanges, in memory and sends them with each LLM request. Old complete exchanges are removed first. History exists only for the lifetime of the process and is never written to disk.

The LLM response limit is 180 tokens with temperature `0.7` and top-p `0.8`. The system prompt asks for concise speech, selects Indonesian or English from the current turn, and uses `<id>...</id>` / `<en>...</en>` spans so Supertonic can pronounce code-switched output appropriately. Complete sentences are queued to TTS while later tokens are still arriving. The clean final response is added to memory only after generation succeeds; a failed or empty response restores the exact previous history.

## Reproducibility manifest

Model binaries are deliberately excluded from Git.

| Artifact | Proven source | Size | SHA-256 |
| --- | --- | ---: | --- |
| whisper.cpp source | Git commit [`60c0be6a`](https://github.com/ggml-org/whisper.cpp/commit/60c0be6a) | — | Git commit ID is the source pin |
| `ggml-small.bin` | [`ggerganov/whisper.cpp` file revision `80da2d8`](https://huggingface.co/ggerganov/whisper.cpp/blob/80da2d8bfee42b0e836fc3a9890373e5defc00a6/ggml-small.bin) | 487,601,967 bytes (~487 MB) | `1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b` |
| `ggml-silero-v6.2.0.bin` | [`ggml-org/whisper-vad` file revision `9ffd54a`](https://huggingface.co/ggml-org/whisper-vad/blob/9ffd54a1e1ee413ddf265af9913beaf518d1639b/ggml-silero-v6.2.0.bin) | 885,098 bytes (~0.88 MB) | `2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987` |
| `models/silero_vad.onnx` | [`snakers4/silero-vad` tag `v6.2.1`](https://github.com/snakers4/silero-vad/blob/v6.2.1/src/silero_vad/data/silero_vad.onnx) | 2,327,524 bytes (~2.33 MB) | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |
| `Qwen3-8B-Q5_K_M.gguf` | [`Qwen/Qwen3-8B-GGUF` file revision `6a56986`](https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/6a569868d07d3bd59e8b97fb001bf8c0b254bb20/Qwen3-8B-Q5_K_M.gguf) | 5,851,112,224 bytes (~5.85 GB) | `068bae163faa96ad48032daf4e071a6a28fe67d8dcc95367609c2ff165e52738` |

The local `models/silero_vad.onnx` present while preparing this repository matched the recorded size and SHA-256. The other large models were not present locally; their sizes and hashes above are the publishers' recorded values. If a deployed LLM file came from a different quantizer repository despite having the same filename, its deployed checksum is not yet recorded and should be verified before treating the published Qwen hash as a match.

Verify downloaded files before use:

```bash
shasum -a 256 /path/to/model-file
```

The Python dependency versions and package hashes are recorded in `uv.lock`; use `uv sync --frozen` rather than resolving a new environment.

## Development checks

The focused regression suite covers VAD duration rounding, bounded capture behavior, startup validation, request cancellation, streamed language-span parsing, shared HTTP-client configuration, and transactional conversation recovery:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m py_compile voice_assistant.py voice_client/*.py
```

See [`TODO.md`](TODO.md) for the remaining improvement roadmap.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, module boundaries, required checks, and pull-request guidance. GitHub issue forms are included for reproducible bug reports and portable feature requests. Do not attach model binaries, secrets, or private recordings.

## License

This project is open source under the [MIT License](LICENSE).

## Troubleshooting

**`models/silero_vad.onnx` cannot be opened:** Run the pinned download command from the repository root and verify its checksum. The default path resolves relative to `voice_assistant.py`, so the launcher can be invoked from another working directory.

**Connection refused on port 8081:** Start the persistent `whisper-server`, confirm it reports `127.0.0.1:8081`, and check that another process is not using the port.

**LLM connection or model errors:** Confirm the configured machine is reachable, query `http://192.168.3.243:8080/v1/models`, and ensure its returned model ID accepts the `Qwen3-8B-Q5_K_M.gguf` value sent by the client. Change `VOICE_LLM_URL` and `VOICE_LLM_MODEL` together when using another backend.

**The visualization remains red:** Read the timestamped terminal error and traceback. At startup this usually identifies a missing VAD file, unsupported audio format, unhealthy Whisper server, unreachable LLM server, or a configured model absent from `/v1/models`. During operation it indicates an STT, LLM, or TTS failure; speaking again retries transient request failures.

**No microphone or speaker is selected:** Run `--list-devices`, then pass a unique name fragment or numeric index with `--mic` and `--speaker`. Numeric indexes can change when USB or Bluetooth devices reconnect.

**PortAudio errors:** Install the PortAudio system library, recreate the uv environment if necessary, and verify the selected device supports the requested input/output direction.

**Qt platform plugin errors on Linux:** Install the missing X11/Wayland Qt runtime libraries for the distribution and run inside a graphical desktop session.

**CUDA or Vulkan silently falls back or fails:** Inspect the first lines printed by `whisper-server`. Rebuild from a clean build directory after installing the correct toolkit and driver, and verify the corresponding CMake option appears in the configure summary.

**High latency:** Keep both servers resident, confirm GPU offload in their startup logs, avoid loading models from slow network storage, and check that the LLM is running with reasoning disabled. The CPU-only whisper.cpp build is functional but slower.
