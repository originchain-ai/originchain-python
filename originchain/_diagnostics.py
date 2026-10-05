"""Request correlation and opt-in client diagnostics (off by default).

Every call carries ``X-OC-Logical-Request-Id`` (one UUID per call, kept across
this client's retries) and ``X-OC-Attempt`` (1, 2, ...). The engine records both
next to its own request id (``X-OC-Request-Id``), so an error a customer sees
joins the engine's record.

With ``diagnostics=True`` the client also reports what it saw of each attempt -
method, path, outcome, duration, the status it received or that nothing was sent
or received, and the request ids - to the customer's own engine
(``POST /v1/tenants/:tenant/diagnostics``, with the client's bearer). The engine
matches the path to its route template and forwards only the template, so table,
key and index names stay on that engine. Nothing else is sent: no SQL, no
parameters, no row data, no search text, no error messages, no keys. Query strings
are removed before anything is queued.

Reporting never slows or fails a call: events go into a bounded queue (the oldest
are dropped when it is full) and are sent in the background in batches of up to
32, at most once a second. A batch that cannot be sent is dropped.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable
from typing import Any, Callable

import httpx

#: The SDK version reported with diagnostics; kept equal to pyproject.toml by a test.
SDK_VERSION = "0.7.0"

LOGICAL_REQUEST_ID_HEADER = "X-OC-Logical-Request-Id"
ATTEMPT_HEADER = "X-OC-Attempt"
MAX_QUEUE = 256
MAX_BATCH = 32
FLUSH_INTERVAL_S = 1.0
SEND_TIMEOUT_S = 5.0

# The engine refuses a whole diagnostics batch for one invalid event, so an event
# that would break any of these rules is never queued.
_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})
_MAX_PATH = 512
_MAX_ATTEMPT = 100
_MAX_DURATION_MS = 3_600_000.0

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)
_ENGINE_ID_RE = re.compile(r"^[0-9a-f]{28}$")
_SQLSTATE_RE = re.compile(r"^[0-9A-Z]{5}$")
_LOWER_CODE_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+){0,3}$")
_UPPER_CODE_RE = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+){0,3}$")


def correlate(headers: dict[str, str] | None) -> tuple[str, dict[str, str]]:
    """The call's logical request id, and the headers to send.

    A caller's own ``X-OC-Logical-Request-Id`` is kept when it is a UUID. A
    non-UUID value is left as the caller sent it (the engine ignores it) and the
    call is reported under a fresh UUID instead."""
    out = dict(headers or {})
    supplied = next(
        (v for k, v in out.items() if k.lower() == LOGICAL_REQUEST_ID_HEADER.lower()), None
    )
    if supplied is not None and _UUID_RE.match(supplied):
        return supplied.lower(), out
    logical_id = str(uuid.uuid4())
    if supplied is None:
        out[LOGICAL_REQUEST_ID_HEADER] = logical_id
    return logical_id, out


def with_attempt(headers: dict[str, str], attempt: int) -> dict[str, str]:
    """``headers`` with this client's attempt number (1-based) set."""
    out = {k: v for k, v in headers.items() if k.lower() != ATTEMPT_HEADER.lower()}
    if attempt <= _MAX_ATTEMPT:
        out[ATTEMPT_HEADER] = str(attempt)
    return out


def engine_request_id(value: str | None) -> str | None:
    """The engine's request id when it is one the contract accepts."""
    if value and (_ENGINE_ID_RE.match(value) or _UUID_RE.match(value)):
        return value
    return None


def error_code(body: Any) -> str | None:
    """The engine's machine error code from an error body, when it is a SQLSTATE
    or a short snake-case / upper-case code - never a message."""
    code: Any = None
    if isinstance(body, dict):
        err = body.get("error")
        code = err.get("code") if isinstance(err, dict) else err
    if not isinstance(code, str):
        return None
    if _SQLSTATE_RE.match(code):
        return code
    # A word code: at most 32 characters and 4 digits (a SQLSTATE has 5).
    if len(code) > 32 or sum(c.isdigit() for c in code) > 4:
        return None
    if _LOWER_CODE_RE.match(code) or _UPPER_CODE_RE.match(code):
        return code
    return None


def _classify(status: int) -> tuple[str, str | None]:
    if status < 400:
        return "success", None
    known = {
        400: ("error", "validation"),
        401: ("denied", "auth"),
        403: ("denied", "permission"),
        404: ("error", "not_found"),
        408: ("timeout", "timeout"),
        409: ("error", "conflict"),
        413: ("error", "capacity"),
        422: ("error", "validation"),
        429: ("error", "rate_limited"),
        502: ("error", "unavailable"),
        503: ("error", "unavailable"),
        504: ("error", "unavailable"),
    }
    if status in known:
        return known[status]
    return ("error", "internal") if status >= 500 else ("error", "unknown")


def _failure(exc: BaseException) -> tuple[str, str, str]:
    """(outcome, error_category, transport) for a request that got no response."""
    # Never connected (or never got a pooled connection): the request was not sent.
    if isinstance(exc, (httpx.ConnectTimeout, httpx.PoolTimeout)):
        return "timeout", "timeout", "not_sent"
    if isinstance(exc, httpx.ConnectError):
        return "error", "network", "not_sent"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", "timeout", "no_response"
    return "error", "network", "no_response"


def event(
    *,
    method: str,
    path: str,
    started: float,
    logical_request_id: str,
    attempt: int,
    status: int | None = None,
    request_id: str | None = None,
    code: str | None = None,
    failure: BaseException | None = None,
) -> dict[str, Any] | None:
    """The diagnostics event for one attempt, or ``None`` when it cannot be
    reported. ``started`` is a ``time.perf_counter()`` reading."""
    method = method.upper()
    path = re.split(r"[?#]", path, maxsplit=1)[0]
    if method not in _METHODS or not path.startswith("/v1/tenants/") or len(path) > _MAX_PATH:
        return None
    duration = (time.perf_counter() - started) * 1000.0
    e: dict[str, Any] = {
        "client": "python",
        "client_version": SDK_VERSION,
        "method": method,
        "path": path,
        "duration_ms": round(min(max(duration, 0.0), _MAX_DURATION_MS), 3),
        "logical_request_id": logical_request_id,
    }
    if attempt <= _MAX_ATTEMPT:
        e["attempt"] = attempt
    if failure is not None:
        e["outcome"], e["error_category"], e["transport"] = _failure(failure)
        return e
    assert status is not None
    e["outcome"], category = _classify(status)
    e["http_status"] = status
    if request_id:
        e["request_id"] = request_id
    if category:
        e["error_category"] = category
        if code:
            e["error_code"] = code
    return e


class _Queue:
    """A bounded queue of events; the oldest are dropped when it is full."""

    def __init__(self) -> None:
        self._events: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()
        #: Events dropped because the queue was full.
        self.dropped = 0

    def _append(self, e: dict[str, Any]) -> bool:
        """Queue ``e``; True when a full batch is waiting."""
        with self._lock:
            if len(self._events) >= MAX_QUEUE:
                self._events.popleft()
                self.dropped += 1
            self._events.append(e)
            return len(self._events) >= MAX_BATCH

    def _take(self) -> list[dict[str, Any]]:
        with self._lock:
            n = min(MAX_BATCH, len(self._events))
            return [self._events.popleft() for _ in range(n)]

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


class SyncReporter(_Queue):
    """Sends queued events from one background daemon thread."""

    def __init__(self, send: Callable[[list[dict[str, Any]]], None]) -> None:
        super().__init__()
        self._send = send
        self._queued = threading.Event()
        self._full = threading.Event()
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    def push(self, e: dict[str, Any] | None) -> None:
        if e is None or self._stopped:
            return
        if self._append(e):
            self._full.set()
        self._queued.set()
        if self._thread is None:
            with self._start_lock:
                if self._thread is None:
                    self._thread = threading.Thread(
                        target=self._run, name="originchain-diagnostics", daemon=True
                    )
                    self._thread.start()

    def _run(self) -> None:
        while not self._stopped:
            self._queued.wait()
            # Gather for up to a second, or until a batch fills.
            self._full.wait(FLUSH_INTERVAL_S)
            self._queued.clear()
            self._full.clear()
            self.flush()

    def flush(self) -> None:
        """Send everything queued now. Never raises."""
        while True:
            batch = self._take()
            if not batch:
                return
            try:
                self._send(batch)
            except Exception:  # noqa: BLE001, S110 - a report must never raise into the caller
                pass  # best-effort: a failed report is dropped, never retried

    def close(self) -> None:
        """Stop the thread after sending what is queued."""
        self._stopped = True
        self._queued.set()
        self._full.set()
        if self._thread is not None:
            self._thread.join(timeout=SEND_TIMEOUT_S + 1.0)
        self.flush()


class AsyncReporter(_Queue):
    """Sends queued events from a task on the running event loop."""

    def __init__(self, send: Callable[[list[dict[str, Any]]], Awaitable[None]]) -> None:
        super().__init__()
        self._send = send
        self._task: asyncio.Task[None] | None = None
        self._full: asyncio.Event | None = None

    def push(self, e: dict[str, Any] | None) -> None:
        if e is None:
            return
        full = self._append(e)
        if self._full is None:
            self._full = asyncio.Event()  # created inside the loop (Python 3.9)
        if full:
            self._full.set()
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._later())

    async def _later(self) -> None:
        assert self._full is not None
        try:
            await asyncio.wait_for(self._full.wait(), FLUSH_INTERVAL_S)
        except asyncio.TimeoutError:
            pass
        self._full.clear()
        await self.flush()

    async def flush(self) -> None:
        """Send everything queued now. Never raises."""
        while True:
            batch = self._take()
            if not batch:
                return
            try:
                await self._send(batch)
            except Exception:  # noqa: BLE001, S110 - a report must never raise into the caller
                pass  # best-effort: a failed report is dropped, never retried

    async def close(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        await self.flush()
