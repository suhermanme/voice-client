"""Runtime defaults and command-line configuration."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


SAMPLE_RATE = 16000
CHUNK = 512
AUDIO_QUEUE_SECONDS = 2
HTTP_TIMEOUT_SECONDS = 60.0

PROJECT_DIR = Path(__file__).resolve().parent.parent

PREFERRED_MIC = "MacBook Pro Microphone"
PREFERRED_SPEAKER = "MacBook Pro Speakers"

VAD_THRESHOLD = 0.60
END_SILENCE_MS = 650
PRE_ROLL_MS = 250
MIN_SPEECH_MS = 250

WHISPER_URL = "http://127.0.0.1:8081/inference"
LLM_URL = "http://192.168.3.243:8080/v1/chat/completions"

SILERO_MODEL = str(PROJECT_DIR / "models" / "silero_vad.onnx")
LLM_MODEL = "Qwen3-8B-Q5_K_M.gguf"

TTS_VOICE = "F1"
TTS_SAMPLE_RATE = 44100
WINDOW_SIZE = 220
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
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
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
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return value


def parse_args(argv=None):
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
        default=os.getenv("VOICE_WHISPER_URL", WHISPER_URL),
        help="whisper-server inference URL.",
    )
    parser.add_argument(
        "--llm-url",
        default=os.getenv("VOICE_LLM_URL", LLM_URL),
        help="OpenAI-compatible chat-completions URL.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("VOICE_LLM_MODEL", LLM_MODEL),
        help="Model ID sent to the LLM backend.",
    )
    parser.add_argument(
        "--vad-model",
        default=os.getenv("VOICE_VAD_MODEL", SILERO_MODEL),
        help="Path to the Python Silero ONNX model.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=probability,
        default=os.getenv("VOICE_VAD_THRESHOLD", str(VAD_THRESHOLD)),
        help="Python live VAD probability threshold.",
    )
    parser.add_argument(
        "--end-silence-ms",
        type=positive_int,
        default=os.getenv("VOICE_END_SILENCE_MS", str(END_SILENCE_MS)),
        help="Silence required to end an utterance.",
    )
    parser.add_argument(
        "--pre-roll-ms",
        type=positive_int,
        default=os.getenv("VOICE_PRE_ROLL_MS", str(PRE_ROLL_MS)),
        help="Audio retained before speech detection.",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=positive_int,
        default=os.getenv("VOICE_MIN_SPEECH_MS", str(MIN_SPEECH_MS)),
        help="Minimum detected speech in an utterance.",
    )
    parser.add_argument(
        "--http-timeout",
        type=positive_float,
        default=os.getenv("VOICE_HTTP_TIMEOUT", str(HTTP_TIMEOUT_SECONDS)),
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-history-messages",
        type=history_limit,
        default=os.getenv(
            "VOICE_MAX_HISTORY_MESSAGES", str(MAX_HISTORY_MESSAGES)
        ),
        help="Maximum user/assistant messages retained.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=os.getenv("VOICE_LOG_LEVEL", "INFO").upper(),
        help="Runtime logging level.",
    )
    return parser.parse_args(argv)


def config_from_args(args):
    return AppConfig(
        whisper_url=args.whisper_url,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        llm_api_key=os.getenv("VOICE_LLM_API_KEY") or None,
        silero_model=args.vad_model,
        vad_threshold=args.vad_threshold,
        end_silence_ms=args.end_silence_ms,
        pre_roll_ms=args.pre_roll_ms,
        min_speech_ms=args.min_speech_ms,
        http_timeout_seconds=args.http_timeout,
        max_history_messages=args.max_history_messages,
        preferred_mic=os.getenv("VOICE_PREFERRED_MIC", PREFERRED_MIC),
        preferred_speaker=os.getenv(
            "VOICE_PREFERRED_SPEAKER", PREFERRED_SPEAKER
        ),
    )
