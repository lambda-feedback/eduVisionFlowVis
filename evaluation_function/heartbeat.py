"""Background heartbeat: a thread that keeps reporting how long it has run.

It tests what happens to work that is still running after
`evaluation_function` returns. Every `INTERVAL_SECONDS` the thread writes a
small JSON file saying how long it has been running and uploads it under the
same name, overwriting the previous version. The HTML report reads that file
again and again and shows it live.

On AWS Lambda the whole execution environment is frozen once the response is
sent, so the thread is expected to stop there and only continue if a later
request reaches the same instance. The file makes that visible: its
`updated_at` stops moving, and when the thread resumes, its next log line
reports how long it was paused.
"""
import json
import sys
import threading
import time
from datetime import datetime, timezone

INTERVAL_SECONDS = 10
DEFAULT_DURATION_SECONDS = 120
# The longest an AWS Lambda invocation can run.
MAX_DURATION_SECONDS = 900
# Each heartbeat uploads a file every INTERVAL_SECONDS, so only a few may run
# at once in one process.
MAX_RUNNING = 3
MIME_TYPE = "application/json"
LOG_LINES = 50

_lock = threading.Lock()
_running = 0


class HeartbeatError(Exception):
    """Raised when a heartbeat is set up wrongly or cannot start."""


def heartbeat_duration(value):
    """Turn the `heartbeat_seconds` parameter into seconds, or None when it is off.

    `true` means DEFAULT_DURATION_SECONDS. Numbers are kept between one
    interval and MAX_DURATION_SECONDS.
    """
    if value is True:
        return DEFAULT_DURATION_SECONDS
    if value is None or value is False:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise HeartbeatError(f"`heartbeat_seconds` must be a number of seconds, not `{value}`.")
    # Written this way round so that NaN also counts as "off".
    if not seconds > 0:
        return None
    return min(max(seconds, INTERVAL_SECONDS), MAX_DURATION_SECONDS)


def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _clock(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _release():
    global _running
    with _lock:
        _running -= 1


class Heartbeat:
    def __init__(self, filename, duration, upload, folder):
        self.filename = filename
        self.folder = folder
        self.duration = duration
        self.interval = INTERVAL_SECONDS
        self._upload = upload
        self.started_at = time.time()
        self.last_beat_at = self.started_at
        self.beats = 0
        self.log = []
        self.url = None
        self.thread = None

    def state(self, status, now):
        return {
            "status": status,
            "started_at": _iso(self.started_at),
            "updated_at": _iso(now),
            "elapsed_seconds": round(now - self.started_at, 1),
            "beats": self.beats,
            "interval_seconds": self.interval,
            "duration_seconds": self.duration,
            "log": self.log,
        }

    def record(self, status, message, now=None):
        """Log one line and upload the current state, returning the file's URL."""
        now = time.time() if now is None else now
        line = f"{_clock(now - self.started_at)}  {message}"
        self.log.append(line)
        del self.log[:-LOG_LINES]
        # stderr, not stdout: with the stdio transport, stdout carries the
        # messages between Shimmy and this worker, and stray text there would
        # break them. Shimmy forwards stderr to the logs (CloudWatch on AWS).
        print(f"[heartbeat {self.filename}] {line}", file=sys.stderr, flush=True)
        data = json.dumps(self.state(status, now)).encode("utf-8")
        return self._upload(data, self.filename, MIME_TYPE, self.folder)

    def _run(self):
        try:
            end = self.started_at + self.duration
            next_beat = min(self.started_at + self.interval, end)
            while True:
                time.sleep(max(0.0, next_beat - time.time()))
                now = time.time()
                gap = now - self.last_beat_at
                finished = now >= end
                self.beats += 1

                message = f"running for {now - self.started_at:.0f} s"
                if gap > self.interval * 1.5:
                    message += f" (no update for {gap:.0f} s, so the thread was paused)"
                if finished:
                    message += ", finished"
                try:
                    self.record("finished" if finished else "running", message, now)
                except Exception as e:
                    print(f"[heartbeat {self.filename}] upload failed: {e}", file=sys.stderr, flush=True)
                self.last_beat_at = now

                if finished:
                    return
                next_beat += self.interval
                if next_beat <= now:
                    # Beats were missed while paused: carry on from now.
                    next_beat = now + self.interval
                next_beat = min(next_beat, end)
        finally:
            _release()


def start_heartbeat(filename, duration, upload, folder):
    """Upload the first state, then start the thread that keeps updating it.

    `upload` is called as `upload(data, filename, mime_type, folder)` and must
    return the file's URL. The first upload happens before this returns, so
    the file already exists when the report first reads it. Raises
    HeartbeatError when too many heartbeats are running, and lets the upload's
    own exception through when the first upload fails.
    """
    global _running
    with _lock:
        if _running >= MAX_RUNNING:
            raise HeartbeatError(
                f"{MAX_RUNNING} background heartbeats are already running, so no new one was started."
            )
        _running += 1

    try:
        heartbeat = Heartbeat(filename, duration, upload, folder)
        heartbeat.url = heartbeat.record("running", "started")
        heartbeat.thread = threading.Thread(target=heartbeat._run, name=f"heartbeat-{filename}", daemon=True)
        heartbeat.thread.start()
    except BaseException:
        _release()
        raise
    return heartbeat
