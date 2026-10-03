import argparse
import json
import queue
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import httpx
import numpy as np

from voice_client.config import parse_args
from voice_assistant import (
    AppConfig,
    CHUNK,
    ConversationHistory,
    RequestCancelled,
    StreamingLanguageParser,
    VoiceAssistant,
    find_audio_device,
    history_limit,
    milliseconds_to_chunks,
    stream_llm,
    transcribe,
    validate_startup,
)


def test_config(**overrides):
    values = {
        "whisper_url": "http://stt.test/inference",
        "llm_url": "http://llm.test/v1/chat/completions",
        "llm_model": "test-model",
        "llm_api_key": "secret",
        "silero_model": "unused.onnx",
        "vad_threshold": 0.6,
        "end_silence_ms": 650,
        "pre_roll_ms": 250,
        "min_speech_ms": 250,
        "http_timeout_seconds": 60,
        "max_history_messages": 20,
        "preferred_mic": "mic",
        "preferred_speaker": "speaker",
        "ui_style": "orb",
    }
    values.update(overrides)
    return AppConfig(**values)


class DurationConversionTests(unittest.TestCase):
    def test_configured_minimums_round_up(self):
        self.assertEqual(
            milliseconds_to_chunks(250),
            8,
        )
        self.assertEqual(
            milliseconds_to_chunks(650),
            21,
        )

    def test_history_limit_requires_a_complete_exchange(self):
        with self.assertRaises(
            argparse.ArgumentTypeError
        ):
            history_limit(1)

    def test_ui_style_can_be_selected_from_cli(self):
        for style in ("circular-wave", "spectrum-pill"):
            with self.subTest(style=style):
                args = parse_args(["--ui-style", style])
                self.assertEqual(args.ui_style, style)


class ConversationHistoryTests(unittest.TestCase):
    def test_restore_recovers_entries_trimmed_by_tentative_turn(self):
        history = ConversationHistory(
            "system",
            max_messages=2,
        )
        history.add_user("first")
        history.add_assistant("answer")

        before = history.get_messages()
        history.add_user("tentative")

        self.assertNotEqual(
            history.get_messages(),
            before,
        )

        history.restore(before)

        self.assertEqual(
            history.get_messages(),
            before,
        )


class AudioQueueTests(unittest.TestCase):
    def make_assistant(self):
        assistant = object.__new__(
            VoiceAssistant
        )
        assistant.stop_event = threading.Event()
        assistant.capture_enabled = threading.Event()
        assistant.signals = Mock()
        assistant.audio_queue = queue.Queue(
            maxsize=1
        )
        return assistant

    def test_callback_ignores_audio_while_capture_is_disabled(self):
        assistant = self.make_assistant()

        assistant.audio_callback(
            np.zeros((CHUNK, 1), dtype=np.float32),
            CHUNK,
            None,
            None,
        )

        self.assertTrue(
            assistant.audio_queue.empty()
        )

    def test_full_queue_keeps_the_newest_chunk(self):
        assistant = self.make_assistant()
        assistant.capture_enabled.set()

        old_chunk = np.zeros(
            (CHUNK, 1),
            dtype=np.float32,
        )
        new_chunk = np.ones(
            (CHUNK, 1),
            dtype=np.float32,
        )

        assistant.audio_callback(
            old_chunk,
            CHUNK,
            None,
            None,
        )
        assistant.audio_callback(
            new_chunk,
            CHUNK,
            None,
            None,
        )

        np.testing.assert_array_equal(
            assistant.audio_queue.get_nowait(),
            new_chunk[:, 0],
        )
        assistant.signals.audio_level_changed.emit.assert_called()


class AudioDeviceSelectionTests(unittest.TestCase):
    devices = [
        {
            "name": "Built-in Microphone",
            "max_input_channels": 1,
            "max_output_channels": 0,
        },
        {
            "name": "USB Audio Microphone",
            "max_input_channels": 2,
            "max_output_channels": 0,
        },
        {
            "name": "USB Audio Speakers",
            "max_input_channels": 0,
            "max_output_channels": 2,
        },
    ]

    def test_unique_partial_name_selects_matching_direction(self):
        with patch(
            "voice_assistant.sd.query_devices",
            return_value=self.devices,
        ):
            self.assertEqual(
                find_audio_device(
                    "audio microphone",
                    input_device=True,
                ),
                1,
            )
            self.assertEqual(
                find_audio_device(
                    "speakers",
                    output_device=True,
                ),
                2,
            )

    def test_ambiguous_partial_name_reports_candidates(self):
        duplicate_inputs = [
            self.devices[0],
            {
                "name": "External Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
        ]

        with patch(
            "voice_assistant.sd.query_devices",
            return_value=duplicate_inputs,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "matches multiple devices",
            ):
                find_audio_device(
                    "microphone",
                    input_device=True,
                )

    def test_numeric_device_must_support_requested_direction(self):
        with patch(
            "voice_assistant.sd.query_devices",
            return_value=self.devices,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "not a valid input device",
            ):
                find_audio_device("2", input_device=True)


class BackendClientTests(unittest.TestCase):
    def test_one_client_serves_configured_stt_and_llm_endpoints(self):
        requests = []

        def handler(request):
            requests.append(request)

            if request.url.path == "/inference":
                return httpx.Response(
                    200,
                    json={"text": "hello"},
                )

            payload = json.loads(
                request.content
            )
            self.assertEqual(
                payload["model"],
                "test-model",
            )
            self.assertEqual(
                request.headers["Authorization"],
                "Bearer secret",
            )

            return httpx.Response(
                200,
                headers={
                    "content-type": "text/event-stream"
                },
                content=(
                    b'data: {"choices":[{"delta":{"content":"<en>Hello</en>"}}]}\n\n'
                    b"data: [DONE]\n\n"
                ),
            )

        config = test_config()
        spans = []

        with httpx.Client(
            transport=httpx.MockTransport(
                handler
            )
        ) as client:
            transcript, _ = transcribe(
                client,
                np.zeros(CHUNK, dtype=np.float32),
                config,
            )
            response, *_ = stream_llm(
                client,
                [{"role": "user", "content": "Hi"}],
                config,
                on_span=spans.append,
            )

        self.assertEqual(transcript, "hello")
        self.assertEqual(response, "<en>Hello</en>")
        self.assertEqual(spans, [("en", "Hello")])
        self.assertEqual(
            [request.url.host for request in requests],
            ["stt.test", "llm.test"],
        )

    def test_transcription_honors_cancellation_before_request(self):
        stop_event = threading.Event()
        stop_event.set()

        def handler(_request):
            self.fail("cancelled request reached the backend")

        with httpx.Client(
            transport=httpx.MockTransport(handler)
        ) as client:
            with self.assertRaises(RequestCancelled):
                transcribe(
                    client,
                    np.zeros(CHUNK, dtype=np.float32),
                    test_config(),
                    stop_event,
                )

    def test_startup_validation_checks_services_and_audio(self):
        requested_paths = []

        def handler(request):
            requested_paths.append(request.url.path)

            if request.url.path == "/health":
                return httpx.Response(
                    200,
                    json={"status": "ok"},
                )

            self.assertEqual(
                request.headers["Authorization"],
                "Bearer secret",
            )
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "/models/test-model"}
                    ]
                },
            )

        with tempfile.NamedTemporaryFile() as model_file:
            config = test_config(
                silero_model=model_file.name
            )

            with (
                patch(
                    "voice_assistant.sd.check_input_settings"
                ) as check_input,
                patch(
                    "voice_assistant.sd.check_output_settings"
                ) as check_output,
                httpx.Client(
                    transport=httpx.MockTransport(handler)
                ) as client,
            ):
                validate_startup(
                    client,
                    config,
                    3,
                    4,
                )

        self.assertEqual(
            requested_paths,
            ["/health", "/v1/models"],
        )
        check_input.assert_called_once_with(
            device=3,
            channels=1,
            dtype="float32",
            samplerate=16000,
        )
        check_output.assert_called_once_with(
            device=4,
            channels=1,
            dtype="float32",
            samplerate=44100,
        )

    def test_startup_rejects_unavailable_llm_model(self):
        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(
                    200,
                    json={"status": "ok"},
                )

            return httpx.Response(
                200,
                json={"data": [{"id": "another-model"}]},
            )

        with tempfile.NamedTemporaryFile() as model_file:
            config = test_config(
                silero_model=model_file.name
            )

            with (
                patch(
                    "voice_assistant.sd.check_input_settings"
                ),
                patch(
                    "voice_assistant.sd.check_output_settings"
                ),
                httpx.Client(
                    transport=httpx.MockTransport(handler)
                ) as client,
                self.assertRaisesRegex(
                    RuntimeError,
                    "Configured LLM model was not reported",
                ),
            ):
                validate_startup(client, config, 3, 4)

    def test_stream_ignores_invalid_sse_and_flushes_final_span(self):
        def handler(_request):
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/event-stream"
                },
                content=(
                    b"event: message\n"
                    b"data: not-json\n\n"
                    b'data: {"choices":[{"delta":{"content":"<id>Halo"}}]}\n\n'
                    b'data: {"choices":[{"delta":{"content":" dunia</id>"}}]}\n\n'
                    b"data: [DONE]\n\n"
                ),
            )

        spans = []
        with httpx.Client(
            transport=httpx.MockTransport(handler)
        ) as client:
            response, *_ = stream_llm(
                client,
                [{"role": "user", "content": "Hi"}],
                test_config(),
                on_span=spans.append,
            )

        self.assertEqual(response, "<id>Halo dunia</id>")
        self.assertEqual(spans, [("id", "Halo dunia")])


class StreamingTTSWorkerTests(unittest.TestCase):
    class FakeTTS:
        def __init__(self):
            self.played = []

        def synthesize(self, spans):
            self.spans = spans
            return np.ones(32, dtype=np.float32), 0.01, 0.25

        def play(self, audio):
            self.played.append(audio)

    def test_worker_reports_speaking_then_thinking(self):
        assistant = object.__new__(VoiceAssistant)
        assistant.stop_event = threading.Event()
        assistant.local_tts = self.FakeTTS()
        states = []
        assistant.set_state = states.append

        span_queue = queue.Queue()
        span_queue.put(("id", "Halo dunia."))
        span_queue.put(None)
        generation_done = threading.Event()
        metrics = {
            "synthesis_time": 0.0,
            "audio_duration": 0.0,
            "failed": False,
            "finished": False,
        }

        assistant.stream_tts_worker(
            span_queue,
            generation_done,
            metrics,
        )

        self.assertEqual(states, ["SPEAKING", "THINKING"])
        self.assertEqual(
            assistant.local_tts.spans,
            [("id", "Halo dunia.")],
        )
        self.assertTrue(metrics["finished"])
        self.assertFalse(metrics["failed"])
        self.assertAlmostEqual(metrics["audio_duration"], 0.25)


class StreamingLanguageParserTests(unittest.TestCase):
    def test_emits_complete_sentences_before_tag_closes(self):
        parser = StreamingLanguageParser()

        self.assertEqual(parser.feed("<e"), [])
        self.assertEqual(
            parser.feed("n>Hello. Next"),
            [("en", "Hello.")],
        )
        self.assertEqual(
            parser.feed(" sentence!</en>"),
            [("en", "Next sentence!")],
        )
        self.assertEqual(
            parser.finish(
                "<en>Hello. Next sentence!</en>"
            ),
            [],
        )

    def test_falls_back_when_response_has_no_language_tags(self):
        parser = StreamingLanguageParser()
        parser.feed("Hello from the local server.")

        self.assertEqual(
            parser.finish("Hello from the local server."),
            [("en", "Hello from the local server.")],
        )


if __name__ == "__main__":
    unittest.main()
