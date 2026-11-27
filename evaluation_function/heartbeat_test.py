import io
import json
import math
import threading
import time
import unittest
from unittest.mock import patch

from . import heartbeat
from .heartbeat import HeartbeatError, heartbeat_duration, start_heartbeat


class FakeStorage:
    """Collects uploads; `delay_on` makes those upload numbers slow."""

    def __init__(self, delay_on=(), delay=0.0):
        self.changed = threading.Condition()
        self.uploads = []
        self.delay_on = delay_on
        self.delay = delay

    def upload(self, data, filename, mime_type, folder):
        with self.changed:
            number = len(self.uploads)
            self.uploads.append({"filename": filename, "mime_type": mime_type, "folder": folder,
                                 "state": json.loads(data)})
            self.changed.notify_all()
        if number in self.delay_on:
            time.sleep(self.delay)
        return f"https://example.com/{folder}/{filename}"

    def wait_for(self, count, timeout=5):
        with self.changed:
            self.changed.wait_for(lambda: len(self.uploads) >= count, timeout)
        return len(self.uploads) >= count

    def states(self, filename=None):
        return [u["state"] for u in self.uploads if filename in (None, u["filename"])]


class TestHeartbeatDuration(unittest.TestCase):
    def test_off_values(self):
        for value in (None, False, 0, -5, float("nan")):
            with self.subTest(value=value):
                self.assertIsNone(heartbeat_duration(value))

    def test_on_values(self):
        self.assertEqual(heartbeat_duration(True), math.inf)
        self.assertEqual(heartbeat_duration(60), 60)
        self.assertEqual(heartbeat_duration("30"), 30)
        self.assertEqual(heartbeat_duration(86_400), 86_400)
        # Never shorter than one interval.
        self.assertEqual(heartbeat_duration(1), heartbeat.INTERVAL_SECONDS)

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
        states = storage.states()
        self.assertGreaterEqual(len(states), 3)
        self.assertEqual([s["beats"] for s in states], list(range(len(states))))
        self.assertEqual(states[0]["status"], "running")
        self.assertEqual(states[-1]["status"], "finished")
        self.assertEqual(states[-1]["duration_seconds"], 0.2)
        self.assertIn("finished", states[-1]["log"][-1])
        self.assertGreaterEqual(states[-1]["elapsed_seconds"], 0.2)

    def test_unlimited_runs_until_stopped(self):
        storage = FakeStorage()

        hb = start_heartbeat("hb.json", math.inf, storage.upload, "f")
        self.assertTrue(storage.wait_for(5))
        self.assertTrue(hb.thread.is_alive())
        hb.stop("by the test")
        hb.thread.join(timeout=5)

        self.assertFalse(hb.thread.is_alive())
        last = storage.states()[-1]
        self.assertEqual(last["status"], "stopped")
        self.assertEqual(last["stop_reason"], "by the test")
        # JSON has no Infinity, so "no limit" is written as null.
        self.assertIsNone(last["duration_seconds"])
        self.assertNotIn(hb, heartbeat._active)

    def test_reports_a_pause(self):
        # A slow upload holds the thread up, like a frozen Lambda would.
        storage = FakeStorage(delay_on=(1,), delay=0.3)

        hb = start_heartbeat("hb.json", 0.4, storage.upload, "f")
        hb.thread.join(timeout=5)

        log = storage.states()[-1]["log"]
        self.assertTrue(any("the thread was paused" in line for line in log), log)

    def test_oldest_is_stopped_when_too_many_run(self):
        storage = FakeStorage()

        with patch.object(heartbeat, "MAX_RUNNING", 1):
            first = start_heartbeat("first.json", math.inf, storage.upload, "f")
            second = start_heartbeat("second.json", math.inf, storage.upload, "f")
            first.thread.join(timeout=5)

        self.assertFalse(first.thread.is_alive())
        self.assertTrue(second.thread.is_alive())
        last_first = storage.states("first.json")[-1]
        self.assertEqual(last_first["status"], "stopped")
        self.assertIn("to make room for a newer heartbeat", last_first["stop_reason"])

        second.stop("by the test")
        second.thread.join(timeout=5)

    def test_failed_first_upload_starts_nothing(self):
        active_before = list(heartbeat._active)

        def failing_upload(*args):
            raise RuntimeError("no bucket")

        with self.assertRaisesRegex(RuntimeError, "no bucket"):
            start_heartbeat("hb.json", 1, failing_upload, "f")
        self.assertEqual(heartbeat._active, active_before)

    def test_logs_to_stderr_not_stdout(self):
        # With the stdio transport, stdout carries the worker's messages.
        hb = heartbeat.Heartbeat("hb.json", 1, FakeStorage().upload, "f")
        with patch("sys.stdout", new_callable=io.StringIO) as out, \
                patch("sys.stderr", new_callable=io.StringIO) as err:
            hb.record("running", "started")
        self.assertEqual(out.getvalue(), "")
        self.assertIn("[heartbeat hb.json]", err.getvalue())
