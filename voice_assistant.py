#!/usr/bin/env python3

import argparse
import io
import json
import math
import os
import queue
import re
import signal
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import onnxruntime as ort
import sounddevice as sd
from supertonic import TTS

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QRadialGradient
from PySide6.QtWidgets import QApplication, QWidget


# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATE = 16000
CHUNK = 512
AUDIO_QUEUE_SECONDS = 2
HTTP_TIMEOUT_SECONDS = 60.0

PROJECT_DIR = Path(__file__).resolve().parent

# Preferred audio devices.
# Device indexes are intentionally NOT hard-coded because Core Audio indexes
# can change when AirPods / USB audio devices are connected or disconnected.
PREFERRED_MIC = "MacBook Pro Microphone"
PREFERRED_SPEAKER = "MacBook Pro Speakers"

# Silero VAD
VAD_THRESHOLD = 0.60
END_SILENCE_MS = 650
PRE_ROLL_MS = 250
MIN_SPEECH_MS = 250

# Services
WHISPER_URL = "http://127.0.0.1:8081/inference"
LLM_URL = "http://192.168.3.243:8080/v1/chat/completions"

# Models
SILERO_MODEL = str(
    PROJECT_DIR / "models" / "silero_vad.onnx"
)
LLM_MODEL = "Qwen3-8B-Q5_K_M.gguf"

# Supertonic
TTS_VOICE = "F1"
TTS_SAMPLE_RATE = 44100

# UI
WINDOW_SIZE = 220

# Conversation memory
# 20 messages ~= 10 user/assistant exchanges.
MAX_HISTORY_MESSAGES = 20


@dataclass(frozen=True)
class AppConfig:
    whisper_url: str
    llm_url: str
    llm_model: str
    llm_api_key: str | None
    silero_model: str
    vad_threshold: float
    end_silence_ms: int
    pre_roll_ms: int
    min_speech_ms: int
    http_timeout_seconds: float
    max_history_messages: int
    preferred_mic: str
    preferred_speaker: str


def positive_int(value):
    value = int(value)

    if value <= 0:
        raise argparse.ArgumentTypeError(
            "value must be greater than zero"
        )

    return value


def positive_float(value):
    value = float(value)

    if value <= 0:
        raise argparse.ArgumentTypeError(
            "value must be greater than zero"
        )

    return value


def history_limit(value):
    value = positive_int(value)

    if value < 2:
        raise argparse.ArgumentTypeError(
            "history must allow at least one user/assistant pair"
        )

    return value


def probability(value):
    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(
            "value must be between 0 and 1"
        )

    return value


def milliseconds_to_chunks(milliseconds):
    return max(
        1,
        math.ceil(
            milliseconds
            / 1000
            * SAMPLE_RATE
            / CHUNK
        ),
    )


# ============================================================================
# Command-line arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Local bilingual voice assistant"
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input/output devices and exit.",
    )

    parser.add_argument(
        "--mic",
        default=os.getenv("VOICE_MIC"),
        help=(
            "Microphone device name, partial name, or numeric device index. "
            "Example: --mic 'MacBook Pro Microphone'"
        ),
    )

    parser.add_argument(
        "--speaker",
        default=os.getenv("VOICE_SPEAKER"),
        help=(
            "Speaker device name, partial name, or numeric device index. "
            "Example: --speaker 'MacBook Pro Speakers'"
        ),
    )

    parser.add_argument(
        "--whisper-url",
        default=os.getenv(
            "VOICE_WHISPER_URL",
            WHISPER_URL,
        ),
        help="whisper-server inference URL.",
    )

    parser.add_argument(
        "--llm-url",
        default=os.getenv(
            "VOICE_LLM_URL",
            LLM_URL,
        ),
        help="OpenAI-compatible chat-completions URL.",
    )

    parser.add_argument(
        "--llm-model",
        default=os.getenv(
            "VOICE_LLM_MODEL",
            LLM_MODEL,
        ),
        help="Model ID sent to the LLM backend.",
    )

    parser.add_argument(
        "--vad-model",
        default=os.getenv(
            "VOICE_VAD_MODEL",
            SILERO_MODEL,
        ),
        help="Path to the Python Silero ONNX model.",
    )

    parser.add_argument(
        "--vad-threshold",
        type=probability,
        default=os.getenv(
            "VOICE_VAD_THRESHOLD",
            str(VAD_THRESHOLD),
        ),
        help="Python live VAD probability threshold.",
    )

    parser.add_argument(
        "--end-silence-ms",
        type=positive_int,
        default=os.getenv(
            "VOICE_END_SILENCE_MS",
            str(END_SILENCE_MS),
        ),
        help="Silence required to end an utterance.",
    )

    parser.add_argument(
        "--pre-roll-ms",
        type=positive_int,
        default=os.getenv(
            "VOICE_PRE_ROLL_MS",
            str(PRE_ROLL_MS),
        ),
        help="Audio retained before speech detection.",
    )

    parser.add_argument(
        "--min-speech-ms",
        type=positive_int,
        default=os.getenv(
            "VOICE_MIN_SPEECH_MS",
            str(MIN_SPEECH_MS),
        ),
        help="Minimum detected speech in an utterance.",
    )

    parser.add_argument(
        "--http-timeout",
        type=positive_float,
        default=os.getenv(
            "VOICE_HTTP_TIMEOUT",
            str(HTTP_TIMEOUT_SECONDS),
        ),
        help="HTTP timeout in seconds.",
    )

    parser.add_argument(
        "--max-history-messages",
        type=history_limit,
        default=os.getenv(
            "VOICE_MAX_HISTORY_MESSAGES",
            str(MAX_HISTORY_MESSAGES),
        ),
        help="Maximum user/assistant messages retained.",
    )

    return parser.parse_args()


def config_from_args(args):
    return AppConfig(
        whisper_url=args.whisper_url,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        llm_api_key=(
            os.getenv("VOICE_LLM_API_KEY")
            or None
        ),
        silero_model=args.vad_model,
        vad_threshold=args.vad_threshold,
        end_silence_ms=args.end_silence_ms,
        pre_roll_ms=args.pre_roll_ms,
        min_speech_ms=args.min_speech_ms,
        http_timeout_seconds=args.http_timeout,
        max_history_messages=(
            args.max_history_messages
        ),
        preferred_mic=os.getenv(
            "VOICE_PREFERRED_MIC",
            PREFERRED_MIC,
        ),
        preferred_speaker=os.getenv(
            "VOICE_PREFERRED_SPEAKER",
            PREFERRED_SPEAKER,
        ),
    )


# ============================================================================
# Audio-device helpers
# ============================================================================

def list_audio_devices():
    devices = sd.query_devices()

    default_input, default_output = sd.default.device

    print("\nAvailable audio devices:\n")

    for index, device in enumerate(devices):
        inputs = int(device["max_input_channels"])
        outputs = int(device["max_output_channels"])

        capabilities = []

        if inputs:
            capabilities.append(f"{inputs} in")

        if outputs:
            capabilities.append(f"{outputs} out")

        capability_text = ", ".join(capabilities) or "no I/O"

        markers = []

        if index == default_input:
            markers.append("default input")

        if index == default_output:
            markers.append("default output")

        marker_text = ""

        if markers:
            marker_text = " [" + ", ".join(markers) + "]"

        print(
            f"{index:2d}: "
            f"{device['name']} "
            f"({capability_text})"
            f"{marker_text}"
        )

    print()


def device_supports(
    device,
    *,
    input_device=False,
    output_device=False,
):
    if (
        input_device
        and device["max_input_channels"] < 1
    ):
        return False

    if (
        output_device
        and device["max_output_channels"] < 1
    ):
        return False

    return True


def find_audio_device(
    requested,
    *,
    input_device=False,
    output_device=False,
):
    devices = sd.query_devices()

    if requested is None:
        return None

    # ------------------------------------------------------------------------
    # Numeric index
    # ------------------------------------------------------------------------

    try:
        index = int(requested)

        if not (
            0 <= index < len(devices)
        ):
            return None

        device = devices[index]

        if not device_supports(
            device,
            input_device=input_device,
            output_device=output_device,
        ):
            role = (
                "input"
                if input_device
                else "output"
            )

            raise RuntimeError(
                f"Audio device {index} "
                f"({device['name']}) "
                f"is not a valid {role} device."
            )

        return index

    except ValueError:
        pass

    requested_lower = (
        requested
        .strip()
        .lower()
    )

    # ------------------------------------------------------------------------
    # Exact name
    # ------------------------------------------------------------------------

    for index, device in enumerate(devices):
        if (
            device["name"]
            .strip()
            .lower()
            != requested_lower
        ):
            continue

        if not device_supports(
            device,
            input_device=input_device,
            output_device=output_device,
        ):
            continue

        return index

    # ------------------------------------------------------------------------
    # Partial name
    # ------------------------------------------------------------------------

    matches = []

    for index, device in enumerate(devices):
        if (
            requested_lower
            not in device["name"].lower()
        ):
            continue

        if not device_supports(
            device,
            input_device=input_device,
            output_device=output_device,
        ):
            continue

        matches.append(index)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        lines = [
            f"{index}: {devices[index]['name']}"
            for index in matches
        ]

        raise RuntimeError(
            f'Audio device "{requested}" '
            "matches multiple devices:\n"
            + "\n".join(lines)
        )

    return None


def get_default_audio_device(
    *,
    input_device=False,
    output_device=False,
):
    default_input, default_output = sd.default.device

    if input_device:
        index = default_input
    else:
        index = default_output

    if index is None:
        return None

    try:
        index = int(index)
    except (TypeError, ValueError):
        return None

    if index < 0:
        return None

    devices = sd.query_devices()

    if index >= len(devices):
        return None

    device = devices[index]

    if not device_supports(
        device,
        input_device=input_device,
        output_device=output_device,
    ):
        return None

    return index


def choose_audio_device(
    requested,
    preferred,
    *,
    input_device=False,
    output_device=False,
):
    # ------------------------------------------------------------------------
    # Explicit CLI choice
    # ------------------------------------------------------------------------

    if requested:
        index = find_audio_device(
            requested,
            input_device=input_device,
            output_device=output_device,
        )

        if index is None:
            raise RuntimeError(
                f'Audio device "{requested}" was not found.'
            )

        return index

    # ------------------------------------------------------------------------
    # Preferred device
    # ------------------------------------------------------------------------

    if preferred:
        index = find_audio_device(
            preferred,
            input_device=input_device,
            output_device=output_device,
        )

        if index is not None:
            return index

    # ------------------------------------------------------------------------
    # System default
    # ------------------------------------------------------------------------

    index = get_default_audio_device(
        input_device=input_device,
        output_device=output_device,
    )

    if index is not None:
        return index

    role = (
        "input"
        if input_device
        else "output"
    )

    raise RuntimeError(
        f"No suitable {role} audio device found."
    )


# ============================================================================
# Prompts
# ============================================================================

WHISPER_PROMPT = (
    "Bahasa Indonesia and English mixed conversation. "
    "GPU, VRAM, CPU, Docker, Proxmox, Linux, macOS, server, "
    "utilization, memory, AI, LLM, model, coding, network."
)


SYSTEM_PROMPT = """
You are a fast bilingual personal AI assistant running fully locally on the
user's own hardware.

Answer only based on information available in the conversation or from tools
you actually use. Do not invent access to systems, devices, data, or services.

CONVERSATION:

- Use the previous messages in this conversation when they are relevant.
- Resolve references such as "it", "that", "the other one", "what about that",
  and similar follow-up questions from the conversation context.
- Do not repeat information unnecessarily when the user is asking a follow-up.

LANGUAGE RULES:

- Determine the response language primarily from the user's CURRENT message.
- If the current user message is English, answer in English.
- If the current user message is Bahasa Indonesia, answer in Bahasa Indonesia.
- If the current user message mixes Bahasa Indonesia and English, answer ONCE
  in the dominant language of that current message.
- Preserve English technical terms naturally when appropriate.
- Do not translate or repeat the same answer in a second language unless the
  user explicitly requests a translation.
- Do not let the language of previous turns override the current user's language.

STYLE:

- Prefer concise, natural conversational responses.
- Responses will be spoken aloud, so avoid unnecessarily long lists.
- Do not use markdown unless the user specifically requests formatted output.

TTS FORMAT:

Every piece of output must be wrapped in language tags.

Use:

<id>Bahasa Indonesia text here.</id>

or:

<en>English text here.</en>

For natural code-switching, alternate tags only where necessary.

Example:

<id>Server Anda sekarang menggunakan sekitar</id>
<en>8 GB of VRAM</en>
<id>dan masih memiliki ruang yang cukup.</id>

Do not output any text outside <id> or <en> tags.
""".strip()


# ============================================================================
# Conversation memory
# ============================================================================

class ConversationHistory:
    def __init__(
        self,
        system_prompt,
        max_messages=MAX_HISTORY_MESSAGES,
    ):
        self.system_prompt = system_prompt
        self.max_messages = max_messages

        self.reset()

    def reset(self):
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

    def add_user(self, text):
        self.messages.append(
            {
                "role": "user",
                "content": text,
            }
        )

        self._trim()

    def add_assistant(self, text):
        self.messages.append(
            {
                "role": "assistant",
                "content": text,
            }
        )

        self._trim()

    def restore(self, messages):
        self.messages = [
            dict(message)
            for message in messages
        ]

    def _trim(self):
        while (
            len(self.messages) - 1
            > self.max_messages
        ):
            self.messages.pop(1)

            # Keep conversation beginning with a user message when possible.
            if (
                len(self.messages) > 1
                and self.messages[1]["role"]
                == "assistant"
            ):
                self.messages.pop(1)

    def get_messages(self):
        return [
            dict(message)
            for message in self.messages
        ]

    @property
    def conversation_message_count(self):
        return (
            len(self.messages) - 1
        )

    @property
    def approximate_turn_count(self):
        return (
            self.conversation_message_count
            // 2
        )


# ============================================================================
# TTS language sanity checking
# ============================================================================

ENGLISH_HINTS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "and",
    "or",
    "but",
    "with",
    "without",
    "for",
    "from",
    "to",
    "of",
    "in",
    "on",
    "at",
    "this",
    "that",
    "these",
    "those",
    "what",
    "when",
    "where",
    "why",
    "how",
    "can",
    "could",
    "would",
    "should",
    "will",
    "your",
    "you",
    "running",
    "using",
    "local",
    "system",
    "server",
    "model",
    "performance",
    "faster",
    "privacy",
    "data",
    "memory",
}

INDONESIAN_HINTS = {
    "yang",
    "dan",
    "atau",
    "tetapi",
    "dengan",
    "tanpa",
    "untuk",
    "dari",
    "ke",
    "di",
    "pada",
    "ini",
    "itu",
    "adalah",
    "bisa",
    "dapat",
    "akan",
    "karena",
    "lebih",
    "saya",
    "anda",
    "kamu",
    "kita",
    "mereka",
    "menjalankan",
    "menggunakan",
    "secara",
    "sistem",
    "kecepatan",
    "penggunaan",
    "memori",
    "privasi",
    "lokal",
}


def normalize_tts_lang(
    lang,
    text,
):
    words = {
        re.sub(
            r"[^a-z]",
            "",
            word.lower(),
        )
        for word in text.split()
    }

    words.discard("")

    en_score = len(
        words & ENGLISH_HINTS
    )

    id_score = len(
        words & INDONESIAN_HINTS
    )

    if (
        en_score
        >= id_score + 2
    ):
        return "en"

    if (
        id_score
        >= en_score + 2
    ):
        return "id"

    return lang


# ============================================================================
# Language tag parsing
# ============================================================================

TAG_PATTERN = re.compile(
    r"<(id|en)>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)


def parse_language_spans(text):
    spans = []

    for match in TAG_PATTERN.finditer(text):
        lang = (
            match
            .group(1)
            .lower()
        )

        content = (
            match
            .group(2)
            .strip()
        )

        if not content:
            continue

        lang = normalize_tts_lang(
            lang,
            content,
        )

        spans.append(
            (
                lang,
                content,
            )
        )

    # Fallback if model forgets tags.
    if not spans:
        clean = re.sub(
            r"</?(?:id|en)>",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        if clean:
            guessed_lang = (
                normalize_tts_lang(
                    "id",
                    clean,
                )
            )

            spans.append(
                (
                    guessed_lang,
                    clean,
                )
            )

    return spans


def clean_response_for_display(text):
    text = re.sub(
        r"</?(?:id|en)>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================================
# Silero VAD
# ============================================================================

class SileroVAD:
    def __init__(self, model_path):
        options = (
            ort.SessionOptions()
        )

        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1

        self.session = (
            ort.InferenceSession(
                model_path,
                sess_options=options,
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )

        self.state = np.zeros(
            (2, 1, 128),
            dtype=np.float32,
        )

        self.sr = np.array(
            SAMPLE_RATE,
            dtype=np.int64,
        )

        self.context = np.zeros(
            (64,),
            dtype=np.float32,
        )

    def reset(self):
        self.state.fill(0)
        self.context.fill(0)

    def probability(self, chunk):
        chunk = np.asarray(
            chunk,
            dtype=np.float32,
        ).reshape(-1)

        x = np.concatenate(
            [
                self.context,
                chunk,
            ]
        )[None, :]

        (
            output,
            new_state,
        ) = self.session.run(
            None,
            {
                "input": x,
                "state": self.state,
                "sr": self.sr,
            },
        )

        self.state = new_state

        self.context = (
            x[0, -64:].copy()
        )

        return float(
            output.squeeze()
        )


# ============================================================================
# Audio helpers
# ============================================================================

def wav_bytes(samples):
    samples = np.clip(
        samples,
        -1.0,
        1.0,
    )

    pcm = (
        samples * 32767
    ).astype(np.int16)

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb",
    ) as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(
            SAMPLE_RATE
        )
        wav.writeframes(
            pcm.tobytes()
        )

    return buffer.getvalue()


def clear_audio_queue(
    audio_queue,
):
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


# ============================================================================
# Whisper STT
# ============================================================================

def transcribe(
    client,
    samples,
    config,
):
    audio = wav_bytes(
        samples
    )

    start = (
        time.perf_counter()
    )

    response = client.post(
        config.whisper_url,
        files={
            "file": (
                "utterance.wav",
                audio,
                "audio/wav",
            )
        },
        data={
            "language": "auto",
            "prompt": WHISPER_PROMPT,
            "carry_initial_prompt": "true",
            "response_format": "json",
        },
    )

    response.raise_for_status()

    text = (
        response
        .json()["text"]
        .strip()
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        text,
        elapsed,
    )


# ============================================================================
# Qwen LLM
# ============================================================================

def stream_llm(
    client,
    messages,
    config,
):
    payload = {
        "model": config.llm_model,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 180,
        "stream": True,
    }

    start = (
        time.perf_counter()
    )

    first_token_delay = None
    first_token_at = None

    pieces = []

    headers = {}

    if config.llm_api_key:
        headers["Authorization"] = (
            f"Bearer {config.llm_api_key}"
        )

    with client.stream(
        "POST",
        config.llm_url,
        json=payload,
        headers=headers,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line.startswith(
                "data: "
            ):
                continue

            data = line[6:]

            if data == "[DONE]":
                break

            try:
                event = json.loads(
                    data
                )

            except json.JSONDecodeError:
                continue

            content = (
                event
                .get(
                    "choices",
                    [{}],
                )[0]
                .get(
                    "delta",
                    {},
                )
                .get(
                    "content"
                )
            )

            if not content:
                continue

            if (
                first_token_delay
                is None
            ):
                first_token_at = (
                    time.perf_counter()
                )

                first_token_delay = (
                    first_token_at
                    - start
                )

            pieces.append(
                content
            )

    total = (
        time.perf_counter()
        - start
    )

    raw_response = (
        "".join(
            pieces
        ).strip()
    )

    return (
        raw_response,
        first_token_delay,
        first_token_at,
        total,
    )


# ============================================================================
# Supertonic TTS
# ============================================================================

class LocalTTS:
    def __init__(
        self,
        speaker_device,
    ):
        self.speaker_device = (
            speaker_device
        )

        print(
            "Loading Supertonic TTS...",
            flush=True,
        )

        start = (
            time.perf_counter()
        )

        self.tts = TTS(
            auto_download=True
        )

        self.voice_style = (
            self.tts
            .get_voice_style(
                voice_name=TTS_VOICE
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"Supertonic ready in "
            f"{elapsed:.2f} s.\n",
            flush=True,
        )

    def synthesize(
        self,
        spans,
    ):
        start = (
            time.perf_counter()
        )

        audio_chunks = []

        for (
            lang,
            text,
        ) in spans:
            text = (
                text.strip()
            )

            if not text:
                continue

            (
                audio,
                _duration,
            ) = self.tts.synthesize(
                text,
                voice_style=(
                    self.voice_style
                ),
                lang=lang,
            )

            audio = (
                np.asarray(
                    audio,
                    dtype=np.float32,
                )
                .squeeze()
            )

            if audio.size:
                audio_chunks.append(
                    audio
                )

        if not audio_chunks:
            return (
                None,
                0.0,
                0.0,
            )

        audio = np.concatenate(
            audio_chunks
        )

        synthesis_time = (
            time.perf_counter()
            - start
        )

        audio_duration = (
            len(audio)
            / TTS_SAMPLE_RATE
        )

        return (
            audio,
            synthesis_time,
            audio_duration,
        )

    def play(
        self,
        audio,
    ):
        if (
            audio is None
            or len(audio) == 0
        ):
            return

        sd.play(
            audio,
            samplerate=TTS_SAMPLE_RATE,
            device=self.speaker_device,
            blocking=True,
        )

    def stop(self):
        try:
            sd.stop()
        except Exception:
            pass


# ============================================================================
# Qt signal bridge
# ============================================================================

class AssistantSignals(QObject):
    state_changed = Signal(str)
    shutdown_requested = Signal()


# ============================================================================
# Orb UI
# ============================================================================

class Orb(QWidget):
    def __init__(
        self,
        signals,
        stop_event,
    ):
        super().__init__()

        self.signals = signals
        self.stop_event = stop_event

        self.state = "IDLE"
        self.phase = 0.0
        self.drag_offset = None

        self.setWindowFlags(
            Qt.Window
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
        )

        self.setAttribute(
            Qt.WA_TranslucentBackground,
            True,
        )

        self.setFixedSize(
            WINDOW_SIZE,
            WINDOW_SIZE,
        )

        (
            self.signals
            .state_changed
            .connect(
                self.set_state
            )
        )

        self.timer = QTimer(
            self
        )

        self.timer.timeout.connect(
            self.animate
        )

        self.timer.start(
            16
        )

    def set_state(
        self,
        state,
    ):
        if state in {
            "IDLE",
            "LISTENING",
            "HEARING",
            "THINKING",
            "SPEAKING",
        }:
            self.state = state
            self.update()

    def animate(self):
        speeds = {
            "IDLE": 0.020,
            "LISTENING": 0.040,
            "HEARING": 0.085,
            "THINKING": 0.075,
            "SPEAKING": 0.120,
        }

        self.phase += speeds.get(
            self.state,
            0.04,
        )

        self.update()

    def paintEvent(
        self,
        event,
    ):
        painter = QPainter(
            self
        )

        painter.setRenderHint(
            QPainter.Antialiasing
        )

        cx = (
            self.width()
            / 2
        )

        cy = (
            self.height()
            / 2
        )

        pulse = (
            math.sin(
                self.phase
            )
            + 1.0
        ) / 2.0

        if (
            self.state
            == "IDLE"
        ):
            color = QColor(
                110,
                120,
                145,
            )
            base_radius = 36

        elif (
            self.state
            == "LISTENING"
        ):
            color = QColor(
                70,
                175,
                255,
            )
            base_radius = 41

        elif (
            self.state
            == "HEARING"
        ):
            color = QColor(
                255,
                165,
                65,
            )
            base_radius = 45

        elif (
            self.state
            == "THINKING"
        ):
            color = QColor(
                155,
                95,
                255,
            )
            base_radius = 43

        else:
            color = QColor(
                70,
                245,
                180,
            )
            base_radius = 46

        radius = (
            base_radius
            + pulse * 8
        )

        # Outer glow
        glow_radius = (
            radius * 2.0
        )

        glow = QRadialGradient(
            cx,
            cy,
            glow_radius,
        )

        glow.setColorAt(
            0.0,
            QColor(
                color.red(),
                color.green(),
                color.blue(),
                185,
            ),
        )

        glow.setColorAt(
            0.35,
            QColor(
                color.red(),
                color.green(),
                color.blue(),
                105,
            ),
        )

        glow.setColorAt(
            1.0,
            QColor(
                color.red(),
                color.green(),
                color.blue(),
                0,
            ),
        )

        painter.setPen(
            Qt.NoPen
        )

        painter.setBrush(
            glow
        )

        painter.drawEllipse(
            int(
                cx
                - glow_radius
            ),
            int(
                cy
                - glow_radius
            ),
            int(
                glow_radius
                * 2
            ),
            int(
                glow_radius
                * 2
            ),
        )

        # Inner orb
        inner = QRadialGradient(
            cx
            - radius * 0.25,
            cy
            - radius * 0.30,
            radius * 1.35,
        )

        inner.setColorAt(
            0.0,
            QColor(
                255,
                255,
                255,
                245,
            ),
        )

        inner.setColorAt(
            0.20,
            QColor(
                min(
                    color.red()
                    + 70,
                    255,
                ),
                min(
                    color.green()
                    + 70,
                    255,
                ),
                min(
                    color.blue()
                    + 70,
                    255,
                ),
                245,
            ),
        )

        inner.setColorAt(
            0.70,
            color,
        )

        inner.setColorAt(
            1.0,
            QColor(
                color.red()
                // 2,
                color.green()
                // 2,
                color.blue()
                // 2,
                245,
            ),
        )

        painter.setBrush(
            inner
        )

        painter.drawEllipse(
            int(
                cx
                - radius
            ),
            int(
                cy
                - radius
            ),
            int(
                radius
                * 2
            ),
            int(
                radius
                * 2
            ),
        )

    def mousePressEvent(
        self,
        event,
    ):
        if (
            event.button()
            == Qt.LeftButton
        ):
            self.drag_offset = (
                event
                .globalPosition()
                .toPoint()
                - self
                .frameGeometry()
                .topLeft()
            )

    def mouseMoveEvent(
        self,
        event,
    ):
        if (
            self.drag_offset
            is not None
            and event.buttons()
            & Qt.LeftButton
        ):
            self.move(
                event
                .globalPosition()
                .toPoint()
                - self.drag_offset
            )

    def mouseReleaseEvent(
        self,
        event,
    ):
        self.drag_offset = None

    def closeEvent(
        self,
        event,
    ):
        print(
            "\nOrb closed. "
            "Stopping voice assistant..."
        )

        self.stop_event.set()

        (
            self.signals
            .shutdown_requested
            .emit()
        )

        event.accept()


# ============================================================================
# Voice assistant worker
# ============================================================================

class VoiceAssistant:
    def __init__(
        self,
        signals,
        stop_event,
        mic_device,
        speaker_device,
        config,
    ):
        self.signals = (
            signals
        )

        self.stop_event = (
            stop_event
        )

        self.mic_device = (
            mic_device
        )

        self.speaker_device = (
            speaker_device
        )

        self.config = config

        queue_chunks = (
            milliseconds_to_chunks(
                AUDIO_QUEUE_SECONDS * 1000
            )
        )

        self.audio_queue = (
            queue.Queue(
                maxsize=queue_chunks
            )
        )

        self.capture_enabled = (
            threading.Event()
        )

        self.vad = SileroVAD(
            self.config.silero_model
        )

        self.local_tts = None
        self.http_client = None

        self.history = (
            ConversationHistory(
                system_prompt=(
                    SYSTEM_PROMPT
                ),
                max_messages=(
                    self.config
                    .max_history_messages
                ),
            )
        )

    def set_state(
        self,
        state,
    ):
        (
            self.signals
            .state_changed
            .emit(
                state
            )
        )

    def audio_callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ):
        if status:
            print(
                f"\nAudio status: "
                f"{status}",
                flush=True,
            )

        if (
            self.stop_event.is_set()
            or not self.capture_enabled.is_set()
        ):
            return

        chunk = indata[:, 0].copy()

        try:
            self.audio_queue.put_nowait(
                chunk
            )

        except queue.Full:
            # Keep the newest audio if processing briefly falls behind.
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.audio_queue.put_nowait(
                    chunk
                )
            except queue.Full:
                pass

    def run(self):
        try:
            self._run()

        except Exception as exc:
            print(
                "\nVoice assistant "
                "fatal error: "
                f"{exc}"
            )

        finally:
            self.capture_enabled.clear()

            if self.http_client:
                self.http_client.close()

            if self.local_tts:
                self.local_tts.stop()

            self.set_state(
                "IDLE"
            )

            self.stop_event.set()

            (
                self.signals
                .shutdown_requested
                .emit()
            )

    def _run(self):
        self.http_client = httpx.Client(
            timeout=(
                self.config
                .http_timeout_seconds
            )
        )

        self.local_tts = (
            LocalTTS(
                self.speaker_device
            )
        )

        pre_roll_chunks = milliseconds_to_chunks(
            self.config.pre_roll_ms
        )

        silence_chunks_needed = milliseconds_to_chunks(
            self.config.end_silence_ms
        )

        min_speech_chunks = milliseconds_to_chunks(
            self.config.min_speech_ms
        )

        pre_roll = []
        utterance = []

        speaking = False
        silence_chunks = 0
        speech_chunks = 0

        print(
            "Listening continuously. "
            "Ctrl-C or close the orb to quit."
        )

        print(
            "Conversation memory: "
            f"up to approximately "
            f"{self.config.max_history_messages // 2} "
            "user/assistant exchanges."
        )

        print(
            "Speak naturally when ready.\n"
        )

        self.set_state(
            "LISTENING"
        )

        self.capture_enabled.set()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK,
            device=self.mic_device,
            callback=self.audio_callback,
        ):
            while (
                not self
                .stop_event
                .is_set()
            ):
                try:
                    chunk = (
                        self.audio_queue
                        .get(
                            timeout=0.1
                        )
                    )

                except queue.Empty:
                    continue

                vad_probability = (
                    self.vad
                    .probability(
                        chunk
                    )
                )

                # ============================================================
                # Waiting for speech
                # ============================================================

                if not speaking:
                    pre_roll.append(
                        chunk
                    )

                    if (
                        len(pre_roll)
                        > pre_roll_chunks
                    ):
                        pre_roll.pop(0)

                    if (
                        vad_probability
                        >= self.config.vad_threshold
                    ):
                        speaking = True

                        utterance = (
                            list(
                                pre_roll
                            )
                        )

                        speech_chunks = 1
                        silence_chunks = 0

                        self.set_state(
                            "HEARING"
                        )

                        print(
                            "Speech detected...",
                            flush=True,
                        )

                    continue

                # ============================================================
                # User speaking
                # ============================================================

                utterance.append(
                    chunk
                )

                if (
                    vad_probability
                    >= self.config.vad_threshold
                ):
                    speech_chunks += 1
                    silence_chunks = 0

                else:
                    silence_chunks += 1

                if (
                    silence_chunks
                    < silence_chunks_needed
                ):
                    continue

                # ============================================================
                # End of utterance
                # ============================================================

                self.capture_enabled.clear()

                if (
                    speech_chunks
                    >= min_speech_chunks
                ):
                    endpoint_detected = (
                        time.perf_counter()
                    )

                    samples = (
                        np.concatenate(
                            utterance
                        )
                    )

                    print(
                        "End of speech detected."
                    )

                    self.set_state(
                        "THINKING"
                    )

                    print(
                        "Transcribing..."
                    )

                    # --------------------------------------------------------
                    # STT
                    # --------------------------------------------------------

                    try:
                        (
                            user_text,
                            stt_time,
                        ) = transcribe(
                            self.http_client,
                            samples,
                            self.config,
                        )

                    except Exception as exc:
                        print(
                            f"STT error: "
                            f"{exc}"
                        )

                        user_text = ""

                    if (
                        self
                        .stop_event
                        .is_set()
                    ):
                        break

                    if user_text:
                        print(
                            f'You: '
                            f'"{user_text}"'
                        )

                        print(
                            f"STT latency: "
                            f"{stt_time * 1000:.0f} ms"
                        )

                        # ----------------------------------------------------
                        # Conversation memory
                        # ----------------------------------------------------

                        history_before_user = (
                            self.history.get_messages()
                        )

                        self.history.add_user(
                            user_text
                        )

                        print(
                            "Conversation context: "
                            f"{self.history.approximate_turn_count} "
                            "stored exchange(s)"
                        )

                        # ----------------------------------------------------
                        # LLM
                        # ----------------------------------------------------

                        try:
                            (
                                raw_response,
                                llm_ttft,
                                first_token_at,
                                llm_total,
                            ) = stream_llm(
                                self.http_client,
                                self.history
                                .get_messages(),
                                self.config,
                            )

                        except Exception as exc:
                            print(
                                f"LLM error: "
                                f"{exc}"
                            )

                            raw_response = ""
                            llm_ttft = None
                            first_token_at = None
                            llm_total = 0.0

                        if (
                            self
                            .stop_event
                            .is_set()
                        ):
                            break

                        if raw_response:
                            clean_response = (
                                clean_response_for_display(
                                    raw_response
                                )
                            )

                            # Store clean semantic response, not TTS tags.
                            self.history.add_assistant(
                                clean_response
                            )

                            print(
                                f"\nAssistant: "
                                f"{clean_response}"
                            )

                            if (
                                llm_ttft
                                is not None
                                and first_token_at
                                is not None
                            ):
                                endpoint_to_first = (
                                    first_token_at
                                    - endpoint_detected
                                )

                                print(
                                    f"\nLLM TTFT: "
                                    f"{llm_ttft * 1000:.0f} ms"
                                )

                                print(
                                    "Endpoint → first-token: "
                                    f"{endpoint_to_first * 1000:.0f} ms"
                                )

                            print(
                                f"LLM total: "
                                f"{llm_total:.2f} s"
                            )

                            print(
                                "Conversation memory: "
                                f"{self.history.approximate_turn_count} "
                                "exchange(s)"
                            )

                            # ------------------------------------------------
                            # TTS
                            # ------------------------------------------------

                            spans = (
                                parse_language_spans(
                                    raw_response
                                )
                            )

                            print(
                                "TTS spans:",
                                spans,
                            )

                            try:
                                (
                                    speech_audio,
                                    tts_time,
                                    speech_duration,
                                ) = (
                                    self
                                    .local_tts
                                    .synthesize(
                                        spans
                                    )
                                )

                                if (
                                    self
                                    .stop_event
                                    .is_set()
                                ):
                                    break

                                print(
                                    f"TTS synthesis: "
                                    f"{tts_time * 1000:.0f} ms "
                                    f"for "
                                    f"{speech_duration:.2f} s audio"
                                )

                                self.set_state(
                                    "SPEAKING"
                                )

                                print(
                                    "Speaking...",
                                    flush=True,
                                )

                                (
                                    self
                                    .local_tts
                                    .play(
                                        speech_audio
                                    )
                                )

                                if (
                                    self
                                    .stop_event
                                    .is_set()
                                ):
                                    break

                                self.set_state(
                                    "LISTENING"
                                )

                                print(
                                    "Listening again.\n",
                                    flush=True,
                                )

                            except Exception as exc:
                                print(
                                    f"TTS error: "
                                    f"{exc}"
                                )

                                self.set_state(
                                    "LISTENING"
                                )

                        else:
                            self.history.restore(
                                history_before_user
                            )

                            print(
                                "No LLM response."
                            )

                            self.set_state(
                                "LISTENING"
                            )

                    else:
                        print(
                            "No speech recognized."
                        )

                        self.set_state(
                            "LISTENING"
                        )

                # ============================================================
                # Reset only audio/VAD state.
                # Conversation history deliberately remains.
                # ============================================================

                clear_audio_queue(
                    self.audio_queue
                )

                self.vad.reset()

                speaking = False
                silence_chunks = 0
                speech_chunks = 0

                utterance = []
                pre_roll = []

                if not self.stop_event.is_set():
                    self.capture_enabled.set()

        print(
            "\nVoice loop stopped."
        )


# ============================================================================
# Main application
# ============================================================================

def main():
    args = parse_args()
    config = config_from_args(
        args
    )

    # ------------------------------------------------------------------------
    # Device listing mode
    # ------------------------------------------------------------------------

    if args.list_devices:
        list_audio_devices()
        return

    # ------------------------------------------------------------------------
    # Resolve audio devices BEFORE starting Qt/voice components.
    # ------------------------------------------------------------------------

    try:
        mic_device = (
            choose_audio_device(
                args.mic,
                config.preferred_mic,
                input_device=True,
            )
        )

        speaker_device = (
            choose_audio_device(
                args.speaker,
                config.preferred_speaker,
                output_device=True,
            )
        )

    except RuntimeError as exc:
        print(
            f"\nAudio configuration error: "
            f"{exc}"
        )

        list_audio_devices()

        sys.exit(1)

    mic_info = (
        sd.query_devices(
            mic_device
        )
    )

    speaker_info = (
        sd.query_devices(
            speaker_device
        )
    )

    print(
        "\nAudio configuration:"
    )

    print(
        f"  Microphone: "
        f"{mic_device} - "
        f"{mic_info['name']}"
    )

    print(
        f"  Speaker:    "
        f"{speaker_device} - "
        f"{speaker_info['name']}"
    )

    print()

    # ------------------------------------------------------------------------
    # Qt application
    # ------------------------------------------------------------------------

    app = QApplication(
        sys.argv
    )

    app.setQuitOnLastWindowClosed(
        True
    )

    stop_event = (
        threading.Event()
    )

    signals = (
        AssistantSignals()
    )

    # ------------------------------------------------------------------------
    # Orb
    # ------------------------------------------------------------------------

    orb = Orb(
        signals,
        stop_event,
    )

    screen = (
        app
        .primaryScreen()
        .availableGeometry()
    )

    x = (
        screen.right()
        - WINDOW_SIZE
        - 40
    )

    y = (
        screen.top()
        + 80
    )

    orb.move(
        x,
        y,
    )

    orb.show()
    orb.raise_()
    orb.activateWindow()

    # ------------------------------------------------------------------------
    # Voice worker
    # ------------------------------------------------------------------------

    assistant = (
        VoiceAssistant(
            signals,
            stop_event,
            mic_device,
            speaker_device,
            config,
        )
    )

    worker = (
        threading.Thread(
            target=assistant.run,
            name="voice-assistant",
            daemon=True,
        )
    )

    # ------------------------------------------------------------------------
    # Clean shutdown
    # ------------------------------------------------------------------------

    def shutdown():
        if (
            not stop_event
            .is_set()
        ):
            print(
                "\nStopping assistant..."
            )

        stop_event.set()

        try:
            sd.stop()
        except Exception:
            pass

        app.quit()

    (
        signals
        .shutdown_requested
        .connect(
            shutdown
        )
    )

    # Give Python a regular opportunity to process Ctrl-C.
    signal_timer = QTimer()

    signal_timer.timeout.connect(
        lambda: None
    )

    signal_timer.start(
        100
    )

    signal.signal(
        signal.SIGINT,
        lambda *_: shutdown(),
    )

    app.aboutToQuit.connect(
        stop_event.set
    )

    # ------------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------------

    worker.start()

    exit_code = (
        app.exec()
    )

    stop_event.set()

    try:
        sd.stop()
    except Exception:
        pass

    worker.join(
        timeout=2.0
    )

    print(
        "Stopped."
    )

    sys.exit(
        exit_code
    )


if __name__ == "__main__":
    main()
