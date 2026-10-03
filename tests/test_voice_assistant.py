import argparse
import json
import queue
import threading
import unittest

import httpx
import numpy as np

from voice_assistant import (
    AppConfig,
    CHUNK,
    ConversationHistory,
    VoiceAssistant,
    history_limit,
    milliseconds_to_chunks,
    stream_llm,
    transcribe,
)


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

        config = AppConfig(
            whisper_url="http://stt.test/inference",
            llm_url="http://llm.test/v1/chat/completions",
            llm_model="test-model",
            llm_api_key="secret",
            silero_model="unused.onnx",
            vad_threshold=0.6,
            end_silence_ms=650,
            pre_roll_ms=250,
            min_speech_ms=250,
            http_timeout_seconds=60,
            max_history_messages=20,
            preferred_mic="mic",
            preferred_speaker="speaker",
        )

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
            )

        self.assertEqual(transcript, "hello")
        self.assertEqual(response, "<en>Hello</en>")
        self.assertEqual(
            [request.url.host for request in requests],
            ["stt.test", "llm.test"],
        )


if __name__ == "__main__":
    unittest.main()
