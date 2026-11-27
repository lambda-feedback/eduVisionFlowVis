"""Runs the keep-alive extension against a fake Lambda API for extensions."""
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from . import keepalive_extension as keepalive


class FakeLambdaApi:
    """The parts of the Lambda Extensions and Telemetry APIs the extension uses."""

    def __init__(self):
        self.events = queue.Queue()
        self.next_calls = []
        self.changed = threading.Condition()
        self.registered_name = None
        self.subscription = None
        api = self

        class Handler(BaseHTTPRequestHandler):
            def _body(self):
                return self.rfile.read(int(self.headers.get("Content-Length") or 0))

            def _reply(self, body=b"{}", headers=()):
                self.send_response(200)
                for name, value in headers:
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self._body()
                api.registered_name = self.headers.get("Lambda-Extension-Name")
                self._reply(headers=[("Lambda-Extension-Identifier", "extension-1")])

            def do_PUT(self):
                api.subscription = json.loads(self._body())
                self._reply(b"OK")

            def do_GET(self):
                with api.changed:
                    api.next_calls.append(time.time())
                    api.changed.notify_all()
                self._reply(json.dumps(api.events.get()).encode("utf-8"))

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.address = f"127.0.0.1:{self.server.server_address[1]}"

    def wait_for_next_call(self, count, timeout=10):
        with self.changed:
            self.changed.wait_for(lambda: len(self.next_calls) >= count, timeout)
            return self.next_calls[count - 1] if len(self.next_calls) >= count else None

    def send_telemetry(self, event_type, request_id):
        url = self.subscription["destination"]["URI"]
        body = json.dumps([{"time": "2026-11-27T12:00:00.000Z", "type": event_type,
                            "record": {"requestId": request_id}}]).encode("utf-8")
        urllib.request.urlopen(urllib.request.Request(url, data=body, method="POST"), timeout=5).read()


@patch.object(keepalive, "LISTENER_HOST", "127.0.0.1")
@patch.object(keepalive, "LISTENER_PORT", 0)
@patch.object(keepalive, "TELEMETRY_WAIT_SECONDS", 0.5)
@patch.object(keepalive, "SAFETY_MARGIN_SECONDS", 0.5)
class TestKeepaliveExtension(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.api = FakeLambdaApi()
        self.addCleanup(self.api.server.shutdown)
        env = patch.dict(os.environ, {"AWS_LAMBDA_RUNTIME_API": self.api.address, "EDUVISION_KEEPALIVE_DIR": folder.name})
        env.start()
        self.addCleanup(env.stop)
        self.requests = 0

    def start_extension(self):
        thread = threading.Thread(target=keepalive.main, daemon=True)
        thread.start()
        self.assertIsNotNone(self.api.wait_for_next_call(1), "the extension did not finish starting")
        return thread

    def invoke(self, deadline_in=30, keep_awake_for=None, telemetry=True, runtime_seconds=0.1):
        """Send one request; return how long the extension held the instance after the runtime finished."""
        self.requests += 1
        request_id = f"request-{self.requests}"
        if keep_awake_for is not None:
            keepalive.write_json_atomic(keepalive.keepalive_file(),
                                        {"until": time.time() + keep_awake_for, "pid": os.getpid()})
        self.api.events.put({"eventType": "INVOKE", "requestId": request_id,
                             "deadlineMs": int((time.time() + deadline_in) * 1000)})
        if telemetry:
            self.api.send_telemetry("platform.start", request_id)
            time.sleep(runtime_seconds)
            self.api.send_telemetry("platform.runtimeDone", request_id)
        runtime_done_at = time.time()
        released_at = self.api.wait_for_next_call(self.requests + 1)
        self.assertIsNotNone(released_at, "the extension never asked for the next event")
        return released_at - runtime_done_at

    def shut_down(self, thread):
        self.api.events.put({"eventType": "SHUTDOWN", "shutdownReason": "SPINDOWN"})
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())

    def test_registers_and_subscribes_to_platform_telemetry(self):
        thread = self.start_extension()

        self.assertEqual(self.api.registered_name, "eduvision-keepalive")
        self.assertEqual(self.api.subscription["types"], ["platform"])
        self.assertEqual(self.api.subscription["buffering"]["timeoutMs"], 25)
        self.assertEqual(self.api.subscription["destination"]["protocol"], "HTTP")
        self.shut_down(thread)

    def test_releases_at_once_without_a_heartbeat(self):
        thread = self.start_extension()

        self.assertLess(self.invoke(), 0.5)
        self.shut_down(thread)

    def test_keeps_awake_while_a_heartbeat_asks(self):
        thread = self.start_extension()
        status_seen = []

        def look_at_status():
            time.sleep(0.8)
            status_seen.append(keepalive.read_json(keepalive.status_file()))

        threading.Thread(target=look_at_status).start()
        held = self.invoke(keep_awake_for=2)

        self.assertGreater(held, 1.4)
        self.assertLess(held, 3)
        self.assertIsNotNone(status_seen[0], "no status file while holding")
        self.assertGreater(status_seen[0]["holding_until"], time.time() - 2)
        self.assertFalse(os.path.exists(keepalive.status_file()), "status file left after releasing")
        self.shut_down(thread)

    def test_never_holds_past_the_request_deadline(self):
        thread = self.start_extension()

        # Deadline in 2 s and a 0.5 s margin: released about 1.4 s after the runtime finished.
        held = self.invoke(deadline_in=2, keep_awake_for=60)

        self.assertGreater(held, 1)
        self.assertLess(held, 2)
        self.shut_down(thread)

    def test_without_telemetry_never_holds_again(self):
        thread = self.start_extension()

        self.assertLess(self.invoke(telemetry=False), 2)
        self.assertLess(self.invoke(keep_awake_for=60), 0.5)
        self.shut_down(thread)

    @unittest.skipUnless(os.path.isdir("/proc"), "needs /proc to see whether a process is alive")
    def test_ignores_requests_from_a_worker_that_has_gone(self):
        finished = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                                  capture_output=True, text=True, check=True)
        keepalive.write_json_atomic(keepalive.keepalive_file(),
                                    {"until": time.time() + 60, "pid": int(finished.stdout)})

        self.assertIsNone(keepalive.requested_until())


class TestRequestDeadline(unittest.TestCase):
    def test_unix_milliseconds(self):
        self.assertAlmostEqual(keepalive.request_deadline({"deadlineMs": 1795807405793}, 0), 1795807405.793)

    def test_milliseconds_from_now(self):
        self.assertEqual(keepalive.request_deadline({"deadlineMs": 676051}, 1000.0), 1676.051)

    def test_missing(self):
        self.assertIsNone(keepalive.request_deadline({}, 1000.0))
