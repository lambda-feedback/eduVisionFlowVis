"""Background heartbeat: a thread that keeps reporting how long it has run.

It tests whether work can keep running after `evaluation_function` returns.
Every `INTERVAL_SECONDS` the thread writes a small JSON file saying how long
it has been running and uploads it under the same name, overwriting the
previous version. The HTML report reads that file again and again and shows
it live.

On AWS Lambda the instance is frozen once the response is sent, which pauses
the thread. To prevent that, the thread asks the keep-alive extension
(keepalive_extension.py) to keep the instance awake until the heartbeat's end,
by writing that time to a file in /tmp. The extension can only do so until the
function's timeout for the current request (at most 15 minutes); after that
the instance is frozen, and the thread continues only when a later request
reaches the same instance, where the extension keeps it awake again. The log
shows how long the thread was paused whenever that happens.
"""
import json
import os
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

from . import keepalive_extension as keepalive

INTERVAL_SECONDS = 10
MAX_DURATION_SECONDS = 3600
# Each heartbeat uploads a file every INTERVAL_SECONDS. When a new one would
# exceed this, the oldest running one is stopped to make room.
MAX_RUNNING = 3
MIME_TYPE = "application/json"
LOG_LINES = 50

_lock = threading.Lock()
_active = []  # running heartbeats, oldest first
_recent = deque(maxlen=10)  # heartbeats started in this process, for status_report()
PROCESS_STARTED_AT = time.time()


class HeartbeatError(Exception):
    """Raised when the `heartbeat_seconds` parameter is invalid."""


def heartbeat_duration(value):
    """Turn the `heartbeat_seconds` parameter into seconds, or None when it is off.

    `true` means MAX_DURATION_SECONDS. Numbers are kept between one interval
    and MAX_DURATION_SECONDS.
    """
    if value is True:
        return MAX_DURATION_SECONDS
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
    return min(max(seconds, INTERVAL_SECONDS), MAX_DURATION_SECONDS)


def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _clock(seconds):
    hours, rest = divmod(int(seconds), 3600)
    clock = f"{rest // 60:02d}:{rest % 60:02d}"
    return f"{hours}:{clock}" if hours else clock


def _log_error(message):
    print(f"[heartbeat] {message}", file=sys.stderr, flush=True)


def _update_keepalive():
    """Tell the keep-alive extension until when to keep the instance awake. Needs _lock."""
    try:
        if _active:
            until = max(hb.started_at + hb.duration for hb in _active)
            keepalive.write_json_atomic(keepalive.keepalive_file(), {"until": until, "pid": os.getpid()})
        else:
            keepalive.remove_file(keepalive.keepalive_file())
    except OSError as e:
        _log_error(f"could not update the keep-alive file: {e}")


def _kept_awake_until(now):
    """Until when the keep-alive extension is holding the instance, or None."""
    status = keepalive.read_json(keepalive.status_file())
    try:
        until = float(status["holding_until"])
    except (TypeError, KeyError, ValueError):
        return None
    return until if until > now else None


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
        self.kept_awake_until = None
        self.status = "running"
        self.url = None
        self.thread = None

    @property
    def duration_seconds(self):
        return self.duration

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
            "kept_awake_until": _iso(self.kept_awake_until) if self.kept_awake_until else None,
            "stop_reason": self.stop_reason,
            "log": self.log,
        }

    def snapshot(self, now):
        """The heartbeat's state as seen from this process, for status_report()."""
        return {
            "file_name": self.filename,
            "status": self.status,
            "started_at": _iso(self.started_at),
            "last_beat_at": _iso(self.last_beat_at),
            "seconds_since_last_beat": round(now - self.last_beat_at, 1),
            "beats": self.beats,
            "duration_seconds": self.duration,
            "log": self.log[-15:],
        }

    def record(self, status, message, now=None):
        """Log one line and upload the current state, returning the file's URL.

        Without an upload function nothing is uploaded and None is returned.
        """
        now = time.time() if now is None else now
        self.status = status
        line = f"{_clock(now - self.started_at)}  {message}"
        self.log.append(line)
        del self.log[:-LOG_LINES]
        # stderr, not stdout: with the stdio transport, stdout carries the
        # messages between Shimmy and this worker, and stray text there would
        # break them. Shimmy forwards stderr to the logs (CloudWatch on AWS).
        print(f"[heartbeat {self.filename}] {line}", file=sys.stderr, flush=True)
        if self._upload is None:
            return None
        data = json.dumps(self.state(status, now)).encode("utf-8")
        return self._upload(data, self.filename, MIME_TYPE, self.folder)

    def _beat(self, status, message, now):
        held = _kept_awake_until(now)
        if held and held != self.kept_awake_until:
            message += f" (the keep-alive extension keeps the instance awake until {keepalive.iso(held)})"
        self.kept_awake_until = held
        try:
            self.record(status, message, now)
        except Exception as e:
            _log_error(f"upload of {self.filename} failed: {e}")
        self.last_beat_at = now

    def _wait_until(self, moment):
        """Sleep until `moment` (Unix time). Returns True if stopped first.

        Checks the wall clock every second instead of sleeping in one go, so
        that after the instance was frozen the thread notices at once that the
        time has passed, whichever clock the sleep itself counts on.
        """
        while True:
            remaining = moment - time.time()
            if remaining <= 0:
                return False
            if self._stop.wait(min(1.0, remaining)):
                return True

    def _run(self):
        try:
            end = self.started_at + self.duration
            next_beat = min(self.started_at + self.interval, end)
            while True:
                stopped = self._wait_until(next_beat)
                now = time.time()
                gap = now - self.last_beat_at
                self.beats += 1

                if stopped:
                    self._beat("stopped", f"running for {now - self.started_at:.0f} s, stopped {self.stop_reason}", now)
                    return

                finished = now >= end
                message = f"running for {now - self.started_at:.0f} s"
                if gap > self.interval * 1.5:
                    message += f", no update for {gap:.0f} s, so the thread was paused"
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
                _update_keepalive()


def start_heartbeat(filename, duration, upload, folder):
    """Upload the first state, then start the thread that keeps updating it.

    `upload` is called as `upload(data, filename, mime_type, folder)` and must
    return the file's URL. With `upload=None` nothing is uploaded, and the
    state is only visible through status_report(). The first upload happens before this returns, so
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
        _recent.append(heartbeat)
        _update_keepalive()

    heartbeat.thread = threading.Thread(target=heartbeat._run, name=f"heartbeat-{filename}", daemon=True)
    try:
        heartbeat.thread.start()
    except BaseException:
        with _lock:
            _active.remove(heartbeat)
            _update_keepalive()
        raise
    return heartbeat


def status_report():
    """Markdown with the heartbeats started in this process and the process itself.

    Two calls answered by the same process show whether its threads kept
    running between them.
    """
    now = time.time()
    with _lock:
        heartbeats = list(_recent)
    data = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "process_running_for_seconds": round(now - PROCESS_STARTED_AT, 1),
        "now": _iso(now),
        "heartbeats": [hb.snapshot(now) for hb in heartbeats],
    }
    return "### Heartbeat status\n\n```json\n" + json.dumps(data, indent=2) + "\n```\n"
