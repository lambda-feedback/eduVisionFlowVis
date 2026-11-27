import io
import json
import threading
import time
import unittest
from unittest.mock import patch

from . import heartbeat
from .heartbeat import HeartbeatError, heartbeat_duration, start_heartbeat


class FakeStorage:
    """Collects uploads; `delay_on` makes those upload numbers slow."""

    def __init__(self, delay_on=(), delay=0.0):
        self.lock = threading.Lock()
        self.uploads = []
        self.delay_on = delay_on
        self.delay = delay

    def upload(self, data, filename, mime_type, folder):
        with self.lock:
            number = len(self.uploads)
            self.uploads.append({"filename": filename, "mime_type": mime_type, "folder": folder,
                                 "state": json.loads(data)})
        if number in self.delay_on:
            time.sleep(self.delay)
        return f"https://example.com/{folder}/{filename}"


class TestHeartbeatDuration(unittest.TestCase):
    def test_off_values(self):
        for value in (None, False, 0, -5, float("nan")):
            with self.subTest(value=value):
                self.assertIsNone(heartbeat_duration(value))

    def test_on_values(self):
        self.assertEqual(heartbeat_duration(True), heartbeat.DEFAULT_DURATION_SECONDS)
        self.assertEqual(heartbeat_duration(60), 60)
        self.assertEqual(heartbeat_duration("30"), 30)
        # Kept between one interval and the Lambda maximum.
        self.assertEqual(heartbeat_duration(1), heartbeat.INTERVAL_SECONDS)
        self.assertEqual(heartbeat_duration(10_000), heartbeat.MAX_DURATION_SECONDS)

    def test_invalid_value(self):
        with self.assertRaisesRegex(HeartbeatError, "number of seconds"):
            heartbeat_duration("soon")


@patch.object(heartbeat, "INTERVAL_SECONDS", 0.05)
class TestHeartbeat(unittest.TestCase):
    def test_keeps_overwriting_one_file_until_finished(self):
        storage = FakeStorage()

        hb = start_heartbeat("hb.json", 0.2, storage.upload, "eduvision/reports")
        hb.thread.join(timeout=5)

        self.assertFalse(hb.thread.is_alive())
        self.assertEqual(hb.url, "https://example.com/eduvision/reports/hb.json")
        self.assertEqual({(u["filename"], u["mime_type"]) for u in storage.uploads}, {("hb.json", "application/json")})
        states = [u["state"] for u in storage.uploads]
        self.assertGreaterEqual(len(states), 3)
        self.assertEqual([s["beats"] for s in states], list(range(len(states))))
        self.assertEqual(states[0]["status"], "running")
        self.assertEqual(states[-1]["status"], "finished")
        self.assertIn("finished", states[-1]["log"][-1])
        self.assertGreaterEqual(states[-1]["elapsed_seconds"], 0.2)

    def test_reports_a_pause(self):
        # A slow upload holds the thread up, like a frozen Lambda would.
        storage = FakeStorage(delay_on=(1,), delay=0.3)

        hb = start_heartbeat("hb.json", 0.4, storage.upload, "f")
        hb.thread.join(timeout=5)

        log = storage.uploads[-1]["state"]["log"]
        self.assertTrue(any("the thread was paused" in line for line in log), log)

    def test_limit_on_running_heartbeats(self):
        before = heartbeat._running
        with patch.object(heartbeat, "MAX_RUNNING", before):
            with self.assertRaisesRegex(HeartbeatError, "already running"):
                start_heartbeat("hb.json", 1, FakeStorage().upload, "f")
        self.assertEqual(heartbeat._running, before)

    def test_failed_first_upload_frees_its_slot(self):
        before = heartbeat._running

        def failing_upload(*args):
            raise RuntimeError("no bucket")

        with self.assertRaisesRegex(RuntimeError, "no bucket"):
            start_heartbeat("hb.json", 1, failing_upload, "f")
        self.assertEqual(heartbeat._running, before)

    def test_logs_to_stderr_not_stdout(self):
        # With the stdio transport, stdout carries the worker's messages.
        hb = heartbeat.Heartbeat("hb.json", 1, FakeStorage().upload, "f")
        with patch("sys.stdout", new_callable=io.StringIO) as out, \
                patch("sys.stderr", new_callable=io.StringIO) as err:
            hb.record("running", "started")
        self.assertEqual(out.getvalue(), "")
        self.assertIn("[heartbeat hb.json]", err.getvalue())
