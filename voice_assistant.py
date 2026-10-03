#!/usr/bin/env python3

import argparse
import io
import json
import logging
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


LOGGER = logging.getLogger("voice_client")


class RequestCancelled(Exception):
    pass


def configure_logging(level):
    logging.basicConfig(
        level=getattr(
            logging,
            level.upper(),
        ),
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s: %(message)s"
        ),
    )


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

    parser.add_argument(
        "--log-level",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
        default=os.getenv(
            "VOICE_LOG_LEVEL",
            "INFO",
        ).upper(),
        help="Runtime logging level.",
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


OPEN_TAG_PATTERN = re.compile(
    r"<(id|en)>",
    re.IGNORECASE,
)

SENTENCE_PATTERN = re.compile(
    r"\s*(.+?[.!?]+)(?=\s|$)",
    re.DOTALL,
)


class StreamingLanguageParser:
    def __init__(self):
        self.buffer = ""
        self.current_lang = None
        self.saw_language_tag = False

    def feed(self, text):
        self.buffer += text
        spans = []

        while True:
            if self.current_lang is None:
                match = OPEN_TAG_PATTERN.search(
                    self.buffer
                )

                if match is None:
                    break

                self.saw_language_tag = True
                self.current_lang = (
                    match.group(1).lower()
                )
                self.buffer = self.buffer[
                    match.end():
                ]

            close_pattern = re.compile(
                rf"</{self.current_lang}>",
                re.IGNORECASE,
            )
            close_match = close_pattern.search(
                self.buffer
            )

            if close_match is not None:
                content = self.buffer[
                    :close_match.start()
                ]
                spans.extend(
                    self._complete_sentences(
                        content,
                        flush=True,
                    )
                )
                self.buffer = self.buffer[
                    close_match.end():
                ]
                self.current_lang = None
                continue

            spans.extend(
                self._complete_sentences(
                    self.buffer,
                    flush=False,
                )
            )
            break

        return spans

    def finish(self, raw_response):
        spans = self.feed("")

        if self.saw_language_tag:
            if (
                self.current_lang
                and self.buffer.strip()
            ):
                spans.extend(
                    self._complete_sentences(
                        self.buffer,
                        flush=True,
                    )
                )

            self.buffer = ""
            self.current_lang = None
            return spans

        self.buffer = ""
        return parse_language_spans(
            raw_response
        )

    def _complete_sentences(
        self,
        text,
        *,
        flush,
    ):
        if flush:
            self.buffer = ""
            candidates = [text]
        else:
            candidates = []
            consumed = 0

            for match in SENTENCE_PATTERN.finditer(
                text
            ):
                candidates.append(
                    match.group(1)
                )
                consumed = match.end()

            self.buffer = text[consumed:]

        spans = []

        for content in candidates:
            content = content.strip()

            if not content:
                continue

            lang = normalize_tts_lang(
                self.current_lang,
                content,
            )
            spans.append((lang, content))

        return spans


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


def sibling_url(url, sibling):
    parsed = httpx.URL(url)
    parent = parsed.path.rsplit("/", 1)[0]
    path = f"{parent}/{sibling}"

    return str(
        parsed.copy_with(path=path)
    )


def llm_models_url(url):
    parsed = httpx.URL(url)
    marker = "/chat/completions"

    if parsed.path.endswith(marker):
        path = (
            parsed.path[:-len(marker)]
            + "/models"
        )
    else:
        path = "/v1/models"

    return str(
        parsed.copy_with(path=path)
    )


def validate_startup(
    client,
    config,
    mic_device,
    speaker_device,
):
    model_path = Path(
        config.silero_model
    )

    if not model_path.is_file():
        raise RuntimeError(
            "Python Silero VAD model not found: "
            f"{model_path}"
        )

    sd.check_input_settings(
        device=mic_device,
        channels=1,
        dtype="float32",
        samplerate=SAMPLE_RATE,
    )
    sd.check_output_settings(
        device=speaker_device,
        channels=1,
        dtype="float32",
        samplerate=TTS_SAMPLE_RATE,
    )

    whisper_health_url = sibling_url(
        config.whisper_url,
        "health",
    )
    response = client.get(
        whisper_health_url
    )
    response.raise_for_status()

    health = response.json()

    if health.get("status") != "ok":
        raise RuntimeError(
            "whisper-server is not ready: "
            f"{health}"
        )

    models_url = llm_models_url(
        config.llm_url
    )
    headers = {}

    if config.llm_api_key:
        headers["Authorization"] = (
            f"Bearer {config.llm_api_key}"
        )

    response = client.get(
        models_url,
        headers=headers,
    )
    response.raise_for_status()
    payload = response.json()
    model_ids = [
        str(item.get("id", ""))
        for item in payload.get("data", [])
        if item.get("id")
    ]

    configured_name = Path(
        config.llm_model
    ).name
    accepted = any(
        model_id == config.llm_model
        or Path(model_id).name
        == configured_name
        for model_id in model_ids
    )

    if not model_ids:
        raise RuntimeError(
            "LLM backend returned no model IDs from "
            f"{models_url}"
        )

    if not accepted:
        raise RuntimeError(
            "Configured LLM model was not reported by "
            f"{models_url}: {config.llm_model}; "
            f"available: {', '.join(model_ids)}"
        )

    LOGGER.info(
        "Startup validation passed: VAD model, audio devices, "
        "whisper-server, and LLM backend are ready"
    )


# ============================================================================
# Whisper STT
# ============================================================================

def transcribe(
    client,
    samples,
    config,
    stop_event=None,
):
    if stop_event and stop_event.is_set():
        raise RequestCancelled()

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

    if stop_event and stop_event.is_set():
        raise RequestCancelled()

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
    on_span=None,
    stop_event=None,
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
    parser = StreamingLanguageParser()

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
            if stop_event and stop_event.is_set():
                raise RequestCancelled()

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

            if on_span:
                for span in parser.feed(
                    content
                ):
                    on_span(span)

    total = (
        time.perf_counter()
        - start
    )

    raw_response = (
        "".join(
            pieces
        ).strip()
    )

    if on_span:
        for span in parser.finish(
            raw_response
        ):
            on_span(span)

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

        LOGGER.info("Loading Supertonic TTS...")

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

        LOGGER.info(
            "Supertonic ready in %.2f s",
            elapsed,
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
            LOGGER.debug(
                "TTS audio stop failed",
                exc_info=True,
            )


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
            "ERROR",
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
            "ERROR": 0.030,
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

        elif (
            self.state
            == "SPEAKING"
        ):
            color = QColor(
                70,
                245,
                180,
            )
            base_radius = 46

        else:
            color = QColor(
                255,
                70,
                85,
            )
            base_radius = 44

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
        LOGGER.info(
            "Orb closed; stopping voice assistant"
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

        self.vad = None

        self.local_tts = None
        self.http_client = None
        self.http_client_lock = (
            threading.Lock()
        )

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
            LOGGER.warning(
                "Audio callback status: %s",
                status,
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
        fatal_error = False

        try:
            self._run()

        except RequestCancelled:
            LOGGER.info(
                "Voice assistant request cancelled"
            )

        except Exception as exc:
            fatal_error = True
            self.set_state("ERROR")
            LOGGER.exception(
                "Voice assistant fatal error: %s",
                exc,
            )

        finally:
            self.capture_enabled.clear()
            self.close_http_client()

            if self.local_tts:
                self.local_tts.stop()

            self.stop_event.set()

            if not fatal_error:
                self.set_state(
                    "IDLE"
                )

                (
                    self.signals
                    .shutdown_requested
                    .emit()
                )

    def close_http_client(self):
        with self.http_client_lock:
            client = self.http_client
            self.http_client = None

        if client is None:
            return

        try:
            client.close()
        except Exception:
            LOGGER.debug(
                "HTTP client close failed",
                exc_info=True,
            )

    def stop(self):
        self.stop_event.set()
        self.capture_enabled.clear()
        self.close_http_client()

        try:
            sd.stop()
        except Exception:
            LOGGER.debug(
                "Audio stop failed",
                exc_info=True,
            )

    def stream_tts_worker(
        self,
        span_queue,
        generation_done,
        metrics,
    ):
        while not self.stop_event.is_set():
            span = span_queue.get()

            if span is None:
                break

            LOGGER.info(
                "TTS span (%s): %s",
                span[0],
                span[1],
            )

            try:
                (
                    speech_audio,
                    tts_time,
                    speech_duration,
                ) = self.local_tts.synthesize(
                    [span]
                )

                if self.stop_event.is_set():
                    break

                metrics["synthesis_time"] += (
                    tts_time
                )
                metrics["audio_duration"] += (
                    speech_duration
                )

                if speech_audio is None:
                    continue

                self.set_state("SPEAKING")
                self.local_tts.play(
                    speech_audio
                )

                if (
                    not self.stop_event.is_set()
                    and not generation_done.is_set()
                ):
                    self.set_state("THINKING")

            except Exception:
                metrics["failed"] = True
                self.set_state("ERROR")
                LOGGER.exception(
                    "Streaming TTS failed"
                )

        metrics["finished"] = True

    def _run(self):
        client = httpx.Client(
            timeout=(
                self.config
                .http_timeout_seconds
            )
        )

        with self.http_client_lock:
            self.http_client = client

        validate_startup(
            client,
            self.config,
            self.mic_device,
            self.speaker_device,
        )

        if self.stop_event.is_set():
            raise RequestCancelled()

        self.vad = SileroVAD(
            self.config.silero_model
        )

        self.local_tts = (
            LocalTTS(
                self.speaker_device
            )
        )

        if self.stop_event.is_set():
            raise RequestCancelled()

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

        LOGGER.info(
            "Listening continuously; Ctrl-C or close the orb to quit"
        )

        LOGGER.info(
            "Conversation memory: "
            f"up to approximately "
            f"{self.config.max_history_messages // 2} "
            "user/assistant exchanges"
        )

        LOGGER.info("Speak naturally when ready")

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

                        LOGGER.info("Speech detected")

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

                    LOGGER.info("End of speech detected")

                    self.set_state(
                        "THINKING"
                    )

                    LOGGER.info("Transcribing")

                    # --------------------------------------------------------
                    # STT
                    # --------------------------------------------------------

                    stt_failed = False

                    try:
                        (
                            user_text,
                            stt_time,
                        ) = transcribe(
                            client,
                            samples,
                            self.config,
                            self.stop_event,
                        )

                    except RequestCancelled:
                        break

                    except Exception as exc:
                        stt_failed = True
                        self.set_state("ERROR")
                        LOGGER.exception(
                            "STT request failed: %s",
                            exc,
                        )

                        user_text = ""

                    if (
                        self
                        .stop_event
                        .is_set()
                    ):
                        break

                    if user_text:
                        LOGGER.info("You: %s", user_text)
                        LOGGER.info(
                            "STT latency: %.0f ms",
                            stt_time * 1000,
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

                        LOGGER.info(
                            "Conversation context: %d stored exchange(s)",
                            self.history.approximate_turn_count,
                        )

                        # ----------------------------------------------------
                        # LLM
                        # ----------------------------------------------------

                        span_queue = queue.Queue()
                        generation_done = (
                            threading.Event()
                        )
                        tts_metrics = {
                            "synthesis_time": 0.0,
                            "audio_duration": 0.0,
                            "failed": False,
                            "finished": False,
                        }
                        tts_thread = threading.Thread(
                            target=self.stream_tts_worker,
                            args=(
                                span_queue,
                                generation_done,
                                tts_metrics,
                            ),
                            name="streaming-tts",
                            daemon=True,
                        )
                        tts_thread.start()
                        request_cancelled = False
                        llm_failed = False

                        def queue_tts_span(span):
                            if not self.stop_event.is_set():
                                span_queue.put(span)

                        try:
                            (
                                raw_response,
                                llm_ttft,
                                first_token_at,
                                llm_total,
                            ) = stream_llm(
                                client,
                                self.history
                                .get_messages(),
                                self.config,
                                on_span=queue_tts_span,
                                stop_event=self.stop_event,
                            )

                        except RequestCancelled:
                            request_cancelled = True
                            raw_response = ""
                            llm_ttft = None
                            first_token_at = None
                            llm_total = 0.0

                        except Exception as exc:
                            if self.stop_event.is_set():
                                request_cancelled = True
                            else:
                                llm_failed = True
                                self.set_state("ERROR")
                                LOGGER.exception(
                                    "LLM request failed: %s",
                                    exc,
                                )

                            raw_response = ""
                            llm_ttft = None
                            first_token_at = None
                            llm_total = 0.0

                        finally:
                            generation_done.set()
                            span_queue.put(None)
                            tts_thread.join()

                        if (
                            request_cancelled
                            or self.stop_event.is_set()
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

                            LOGGER.info(
                                "Assistant: %s",
                                clean_response,
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

                                LOGGER.info(
                                    "LLM TTFT: %.0f ms",
                                    llm_ttft * 1000,
                                )

                                LOGGER.info(
                                    "Endpoint to first token: %.0f ms",
                                    endpoint_to_first * 1000,
                                )

                            LOGGER.info(
                                "LLM generation: %.2f s",
                                llm_total,
                            )

                            LOGGER.info(
                                "Conversation memory: %d exchange(s)",
                                self.history.approximate_turn_count,
                            )

                            LOGGER.info(
                                "TTS synthesis: %.0f ms for %.2f s audio",
                                tts_metrics["synthesis_time"] * 1000,
                                tts_metrics["audio_duration"],
                            )
                            if tts_metrics["failed"]:
                                self.set_state("ERROR")
                            else:
                                self.set_state("LISTENING")

                        else:
                            self.history.restore(
                                history_before_user
                            )

                            LOGGER.warning(
                                "LLM returned no response"
                            )

                            if not llm_failed:
                                self.set_state(
                                    "LISTENING"
                                )

                    else:
                        LOGGER.info("No speech recognized")

                        if not stt_failed:
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

        LOGGER.info("Voice loop stopped")


# ============================================================================
# Main application
# ============================================================================

def main():
    args = parse_args()
    configure_logging(args.log_level)
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
        LOGGER.error("Audio configuration error: %s", exc)

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

    LOGGER.info(
        "Audio input: %s - %s",
        mic_device,
        mic_info["name"],
    )
    LOGGER.info(
        "Audio output: %s - %s",
        speaker_device,
        speaker_info["name"],
    )

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
            LOGGER.info("Stopping assistant")

        assistant.stop()

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
        assistant.stop
    )

    # ------------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------------

    worker.start()

    exit_code = (
        app.exec()
    )

    assistant.stop()

    worker.join(
        timeout=5.0
    )

    if worker.is_alive():
        LOGGER.warning(
            "Voice worker did not exit within 5 seconds"
        )
    else:
        LOGGER.info("Stopped")

    sys.exit(
        exit_code
    )


if __name__ == "__main__":
    main()
