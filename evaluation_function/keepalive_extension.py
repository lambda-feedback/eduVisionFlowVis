"""AWS Lambda extension that keeps the instance awake while a heartbeat asks for it.

Lambda freezes an instance once the function has answered and every extension
has said it is done, by asking for its next event. It sends the answer to the
caller straight away, without waiting for extensions. So by holding back its
"done", this extension keeps the instance running after the student already
has their feedback, which lets the background heartbeat thread (heartbeat.py)
keep working.

For every request:

1. Lambda sends this extension an INVOKE event.
2. The extension waits until the Telemetry API reports that the function has
   finished the request (`platform.runtimeDone`).
3. If heartbeat.py has written a keep-alive file asking to stay awake until a
   later time, the extension holds the instance until then, but never past the
   request's deadline (the function timeout) minus a safety margin. Lambda
   resets an instance that runs past its deadline.
4. It asks for the next event, and Lambda may freeze the instance.

If anything goes wrong, the extension stops holding and behaves as if it were
not there. In particular, if no telemetry arrives for a request within a few
seconds, it never holds again for the rest of the instance's life.

Lambda starts it from /opt/extensions (see the Dockerfile). It uses only the
standard library, so it starts fast and does not depend on the function's
packages. heartbeat.py imports it for the shared file names and helpers.
"""
import json
import os
import tempfile
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

# Must be the file name of the executable in /opt/extensions.
EXTENSION_NAME = "eduvision-keepalive"
# Telemetry is sent to this host inside the instance. Port 9001 is reserved by Lambda.
LISTENER_HOST = "sandbox.localdomain"
LISTENER_PORT = 4243
# Release the instance this long before the request's deadline.
SAFETY_MARGIN_SECONDS = 3
# Telemetry counts as broken if a request's `platform.start` takes longer than this.
TELEMETRY_WAIT_SECONDS = 3

KEEPALIVE_FILE_NAME = "eduvision-keepalive.json"
STATUS_FILE_NAME = "eduvision-keepalive-status.json"


# ---------- files shared with heartbeat.py ----------

def keepalive_dir():
    # /tmp on Lambda: the only writable folder, shared by the function and its extensions.
    return os.environ.get("EDUVISION_KEEPALIVE_DIR") or tempfile.gettempdir()


def keepalive_file():
    """Written by heartbeat.py: {"until": <Unix time>, "pid": <worker process>}."""
    return os.path.join(keepalive_dir(), KEEPALIVE_FILE_NAME)


def status_file():
    """Written by this extension while it holds the instance: {"holding_until": <Unix time>}."""
    return os.path.join(keepalive_dir(), STATUS_FILE_NAME)


def write_json_atomic(path, data):
    temporary = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(temporary, path)


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def remove_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%H:%M:%S UTC")


def process_alive(pid):
    if not os.path.isdir("/proc"):
        return True  # No way to check, e.g. when testing on Windows.
    return os.path.exists(f"/proc/{pid}")


def requested_until(now=None):
    """The time a running heartbeat wants the instance kept awake until, or None."""
    data = read_json(keepalive_file())
    try:
        until = float(data["until"])
        pid = int(data.get("pid") or 0)
    except (TypeError, KeyError, ValueError):
        return None
    if pid and not process_alive(pid):
        return None  # The worker that wrote it has gone, so nothing is running.
    return until if until > (time.time() if now is None else now) else None


# ---------- Lambda APIs ----------

def log(message):
    # An extension's output goes to the function's logs (CloudWatch).
    print(f"[{EXTENSION_NAME}] {message}", flush=True)


def call_api(method, path, body=None, headers=None, timeout=None):
    """Call the Lambda API for extensions. `timeout=None` waits for as long as it takes."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"http://{os.environ['AWS_LAMBDA_RUNTIME_API']}{path}", data=data, method=method, headers=headers or {}
    )
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.headers, response.read()


class Telemetry:
    """Remembers which requests have started and finished, from Telemetry API events."""

    def __init__(self):
        self._changed = threading.Condition()
        self._started = set()
        self._done = set()

    def add(self, events):
        with self._changed:
            for event in events:
                if not isinstance(event, dict) or not isinstance(event.get("record"), dict):
                    continue
                request_id = event["record"].get("requestId")
                if event.get("type") == "platform.start":
                    self._started.add(request_id)
                elif event.get("type") == "platform.runtimeDone":
                    self._done.add(request_id)
            self._changed.notify_all()

    def wait_started(self, request_id, timeout):
        return self._wait(self._started, request_id, timeout)

    def wait_done(self, request_id, timeout):
        return self._wait(self._done, request_id, timeout)

    def _wait(self, seen, request_id, timeout):
        with self._changed:
            return self._changed.wait_for(lambda: request_id in seen, timeout=max(0.0, timeout))

    def forget(self, request_id):
        with self._changed:
            self._started.discard(request_id)
            self._done.discard(request_id)


def start_listener(telemetry):
    """Start the HTTP server that Lambda posts telemetry events to, and return its port."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.end_headers()
            try:
                events = json.loads(body)
            except ValueError:
                return
            if isinstance(events, list):
                telemetry.add(events)

        def log_message(self, *args):
            pass  # Logging every batch would only add noise to the function's logs.

    last_error = None
    for host in (LISTENER_HOST, "0.0.0.0"):
        try:
            server = HTTPServer((host, LISTENER_PORT), Handler)
            break
        except OSError as e:
            last_error = e
    else:
        raise last_error
    threading.Thread(target=server.serve_forever, name="telemetry-listener", daemon=True).start()
    return server.server_address[1]


def subscribe(extension_id, port):
    call_api(
        "PUT",
        "/2022-07-01/telemetry",
        {
            "schemaVersion": "2022-12-13",
            "types": ["platform"],
            # The smallest buffer and delay allowed, so `runtimeDone` arrives quickly.
            "buffering": {"maxItems": 1000, "maxBytes": 262144, "timeoutMs": 25},
            "destination": {"protocol": "HTTP", "URI": f"http://{LISTENER_HOST}:{port}"},
        },
        {"Lambda-Extension-Identifier": extension_id},
        timeout=5,
    )


# ---------- holding the instance ----------

def request_deadline(event, received_at):
    """The request's deadline as Unix time, or None if the event has none."""
    try:
        ms = float(event["deadlineMs"])
    except (KeyError, TypeError, ValueError):
        return None
    if ms > 1e12:
        return ms / 1000  # Unix time in milliseconds, as Lambda sends it.
    if ms > 0:
        return received_at + ms / 1000  # Milliseconds from now, as in one AWS docs example.
    return None


def hold_while_requested(release_by):
    """Keep this extension busy, and so the instance awake, while a heartbeat asks for it."""
    held_since = None
    written = None
    while True:
        now = time.time()
        until = requested_until(now)
        if until is None or now >= release_by:
            break
        hold_until = min(until, release_by)
        if held_since is None:
            held_since = now
            log(f"keeping the instance awake for the heartbeat until {iso(hold_until)}")
        if hold_until != written:
            write_json_atomic(status_file(), {"holding_until": hold_until})
            written = hold_until
        time.sleep(min(1.0, max(0.0, hold_until - now)))
    if held_since is not None:
        log(f"released the instance after keeping it awake for {time.time() - held_since:.0f} s")


def handle_invoke(event, telemetry):
    """Hold the instance after this request if a heartbeat asks for it.

    Returns False when telemetry seems not to work, so holding is unsafe.
    """
    received_at = time.time()
    request_id = event.get("requestId")
    deadline = request_deadline(event, received_at)
    try:
        if not telemetry.wait_started(request_id, TELEMETRY_WAIT_SECONDS):
            return False
        if deadline is None:
            return True
        release_by = deadline - SAFETY_MARGIN_SECONDS
        # The student's answer has been sent once the runtime is done.
        if telemetry.wait_done(request_id, release_by - time.time()):
            hold_while_requested(release_by)
        return True
    finally:
        telemetry.forget(request_id)
        remove_file(status_file())


def main():
    if "AWS_LAMBDA_RUNTIME_API" not in os.environ:
        log("not running on AWS Lambda, so there is nothing to do")
        return

    headers, _ = call_api(
        "POST",
        "/2020-01-01/extension/register",
        {"events": ["INVOKE", "SHUTDOWN"]},
        {"Lambda-Extension-Name": EXTENSION_NAME},
        timeout=5,
    )
    extension_id = headers["Lambda-Extension-Identifier"]

    telemetry = Telemetry()
    try:
        subscribe(extension_id, start_listener(telemetry))
    except Exception as e:
        log(f"telemetry is unavailable, so the instance is never kept awake: {e}")
        telemetry = None

    while True:
        _, body = call_api(
            "GET", "/2020-01-01/extension/event/next", headers={"Lambda-Extension-Identifier": extension_id}
        )
        event = json.loads(body)
        if event.get("eventType") == "SHUTDOWN":
            remove_file(status_file())
            return
        if telemetry is None:
            continue
        try:
            if not handle_invoke(event, telemetry):
                log("no telemetry arrived for a request, so the instance is never kept awake from now on")
                telemetry = None
        except Exception as e:
            log(f"error while keeping the instance awake, so it was released: {e}")


if __name__ == "__main__":
    main()
