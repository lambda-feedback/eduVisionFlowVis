"""Background heartbeat: a thread that keeps reporting how long it has run.

It tests what happens to work that is still running after
`evaluation_function` returns. Every `INTERVAL_SECONDS` the thread writes a
small JSON file saying how long it has been running and uploads it under the
same name, overwriting the previous version. The HTML report reads that file
again and again and shows it live.

By default the thread has no time limit and runs for as long as its process
does. On AWS Lambda the whole execution environment is frozen once the
response is sent, so the thread is expected to pause there and only continue
if a later request reaches the same instance; when AWS shuts the instance
down, the thread ends without a final update. The file makes this visible:
its `updated_at` stops moving, and when the thread resumes, its next log line
reports how long it was paused.
"""
import json
import math
import sys
import threading
import time
from datetime import datetime, timezone

INTERVAL_SECONDS = 10
# `heartbeat_seconds: true`: run until the process ends.
UNLIMITED = math.inf
# Each heartbeat uploads a file every INTERVAL_SECONDS. When a new one would
# exceed this, the oldest running one is stopped to make room.
MAX_RUNNING = 3
MIME_TYPE = "application/json"
LOG_LINES = 50

_lock = threading.Lock()
_active = []  # running heartbeats, oldest first


class HeartbeatError(Exception):
    """Raised when the `heartbeat_seconds` parameter is invalid."""


def heartbeat_duration(value):
    """Turn the `heartbeat_seconds` parameter into seconds, or None when it is off.

    `true` means UNLIMITED. Numbers shorter than one interval are raised to
    one interval.
    """
    if value is True:
        return UNLIMITED
    if value is None or value is False:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise HeartbeatError(
            f"`heartbeat_seconds` must be `true` or a number of seconds, not `{value}`."
        )
    # Written this way round so that NaN also counts as "off".
    if not seconds > 0:
        return None
    return max(seconds, INTERVAL_SECONDS)


def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _clock(seconds):
    hours, rest = divmod(int(seconds), 3600)
    clock = f"{rest // 60:02d}:{rest % 60:02d}"
    return f"{hours}:{clock}" if hours else clock


class Heartbeat:
    def __init__(self, filename, duration, upload, folder):
        self.filename = filename
        self.folder = folder
        self.duration = duration
        self.interval = INTERVAL_SECONDS
        self._upload = upload
        self._stop = threading.Event()
        self.stop_reason = None
        self.started_at = time.time()
        self.last_beat_at = self.started_at
        self.beats = 0
        self.log = []
        self.url = None
        self.thread = None

    @property
    def duration_seconds(self):
        """The time limit in seconds, or None when there is none (JSON has no Infinity)."""
        return None if math.isinf(self.duration) else self.duration

    def stop(self, reason):
        """Ask the thread to write a last update saying `reason`, then end."""
        self.stop_reason = reason
        self._stop.set()

    def state(self, status, now):
        return {
            "status": status,
            "started_at": _iso(self.started_at),
            "updated_at": _iso(now),
            "elapsed_seconds": round(now - self.started_at, 1),
            "beats": self.beats,
            "interval_seconds": self.interval,
            "duration_seconds": self.duration_seconds,
            "stop_reason": self.stop_reason,
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

    def _beat(self, status, message, now):
        try:
            self.record(status, message, now)
        except Exception as e:
            print(f"[heartbeat {self.filename}] upload failed: {e}", file=sys.stderr, flush=True)
        self.last_beat_at = now

    def _run(self):
        try:
            end = self.started_at + self.duration
            next_beat = min(self.started_at + self.interval, end)
            while True:
                stopped = self._stop.wait(max(0.0, next_beat - time.time()))
                now = time.time()
                gap = now - self.last_beat_at
                self.beats += 1

                if stopped:
                    self._beat("stopped", f"running for {now - self.started_at:.0f} s, stopped {self.stop_reason}", now)
                    return

                finished = now >= end
                message = f"running for {now - self.started_at:.0f} s"
                if gap > self.interval * 1.5:
                    message += f" (no update for {gap:.0f} s, so the thread was paused)"
                if finished:
                    message += ", finished"
                self._beat("finished" if finished else "running", message, now)

                if finished:
                    return
                next_beat += self.interval
                if next_beat <= now:
                    # Beats were missed while paused: carry on from now.
                    next_beat = now + self.interval
                next_beat = min(next_beat, end)
        finally:
            with _lock:
                if self in _active:
                    _active.remove(self)


def start_heartbeat(filename, duration, upload, folder):
    """Upload the first state, then start the thread that keeps updating it.

    `upload` is called as `upload(data, filename, mime_type, folder)` and must
    return the file's URL. The first upload happens before this returns, so
    the file already exists when the report first reads it; if it fails, its
    exception is raised and no thread is started. When MAX_RUNNING heartbeats
    are already running, the oldest is stopped.
    """
    heartbeat = Heartbeat(filename, duration, upload, folder)
    heartbeat.url = heartbeat.record("running", "started")

    with _lock:
        while len(_active) >= MAX_RUNNING:
            _active.pop(0).stop(f"to make room for a newer heartbeat (at most {MAX_RUNNING} run at once)")
        _active.append(heartbeat)

    heartbeat.thread = threading.Thread(target=heartbeat._run, name=f"heartbeat-{filename}", daemon=True)
    try:
        heartbeat.thread.start()
    except BaseException:
        with _lock:
            _active.remove(heartbeat)
        raise
    return heartbeat
