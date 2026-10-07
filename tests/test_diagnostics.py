"""Request correlation and opt-in diagnostics.

Every test routes the client through ``httpx.MockTransport``: engine calls are
answered by the test's handler, and posts to ``/diagnostics`` are recorded
separately so the reports can be inspected.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from originchain import AsyncOriginChain, OCError, OCServerError, OriginChain
from originchain import _diagnostics as diag

TENANT = "01HX1TESTTENANTXXXXXXXXXX1"
ENGINE_ID = "3f2a9c1b7e5d4a60000000000042"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
ALLOWED = {
    "client", "client_version", "method", "path", "outcome", "duration_ms", "http_status",
    "error_category", "error_code", "request_id", "logical_request_id", "attempt", "transport",
}

Handler = Callable[[httpx.Request], httpx.Response]


class Engine:
    """Answers engine calls with ``answers`` in turn (the last one repeats); a
    callable answer is raised or returned. Records calls and diagnostics posts."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers) or [(200, {"kind": "select", "rows": []})]
        self.calls: list[httpx.Request] = []
        self.reports: list[list[dict[str, Any]]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/diagnostics"):
            self.reports.append(json.loads(req.content)["events"])
            return httpx.Response(202, json={"accepted": 1})
        self.calls.append(req)
        answer = self.answers[min(len(self.calls), len(self.answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        return httpx.Response(status, json=body, headers={"X-OC-Request-Id": ENGINE_ID})

    @property
    def events(self) -> list[dict[str, Any]]:
        return [e for batch in self.reports for e in batch]


def client(engine: Handler, *, diagnostics: bool = False, retries: int = 0) -> OriginChain:
    db = OriginChain(
        base_url="http://test.invalid", bearer="b", tenant=TENANT,
        max_retries=retries, diagnostics=diagnostics,
    )
    db._client = httpx.Client(base_url="http://test.invalid", transport=httpx.MockTransport(engine))
    db._backoff = lambda attempt: 0.0  # type: ignore[method-assign]
    return db


def test_every_call_sends_a_logical_request_id_kept_across_retries() -> None:
    engine = Engine((503, {"error": "busy"}), (200, {"kind": "select", "rows": []}))
    db = client(engine, retries=2)
    db.sql("SELECT 1")
    db.sql("SELECT 2")
    first, retry, second = engine.calls
    lrid = first.headers["X-OC-Logical-Request-Id"]
    assert UUID_RE.match(lrid), lrid
    assert retry.headers["X-OC-Logical-Request-Id"] == lrid, "the same id on a retry"
    assert (first.headers["X-OC-Attempt"], retry.headers["X-OC-Attempt"]) == ("1", "2")
    assert second.headers["X-OC-Logical-Request-Id"] != lrid
    assert second.headers["X-OC-Attempt"] == "1"


def test_errors_carry_both_ids() -> None:
    engine = Engine((500, {"error": {"code": "internal", "message": "boom"}}))
    with pytest.raises(OCServerError) as raised:
        client(engine).sql("SELECT 1")
    assert raised.value.request_id == ENGINE_ID
    assert raised.value.logical_request_id == engine.calls[0].headers["X-OC-Logical-Request-Id"]

    lost = Engine(httpx.ConnectError("refused"))
    with pytest.raises(OCError) as raised:
        client(lost).sql("SELECT 1")
    assert raised.value.request_id is None
    assert UUID_RE.match(raised.value.logical_request_id or "")


def test_diagnostics_are_off_by_default() -> None:
    engine = Engine()
    db = client(engine)
    db.sql("SELECT 1")
    db.flush_diagnostics()
    db.close()
    assert engine.reports == []


def test_reports_carry_only_contract_fields_and_no_query_text() -> None:
    engine = Engine((503, {"error": "engine is busy"}), (200, {"kind": "select", "rows": []}))
    db = client(engine, diagnostics=True, retries=1)
    db.sql("SELECT secret FROM private_table")
    db._request("GET", f"/v1/tenants/{TENANT}/fts/articles/body?q=confidential+search")
    db.flush_diagnostics()
    events = engine.events
    assert len(events) == 3, "one per attempt, the retried 503 included"
    for e in events:
        assert set(e) <= ALLOWED, set(e) - ALLOWED
        assert not re.search("secret|private_table|confidential", json.dumps(e))
    busy, ok, fts = events
    lrid = engine.calls[0].headers["X-OC-Logical-Request-Id"]
    assert busy == {
        **busy,
        "client": "python", "client_version": diag.SDK_VERSION, "method": "POST",
        "path": f"/v1/tenants/{TENANT}/sql", "outcome": "error",
        "error_category": "unavailable", "http_status": 503, "request_id": ENGINE_ID,
        "logical_request_id": lrid, "attempt": 1,
    }
    assert "error_code" not in busy, "a message is never sent as a code"
    assert (ok["outcome"], ok["attempt"], ok["logical_request_id"]) == ("success", 2, lrid)
    assert fts["path"] == f"/v1/tenants/{TENANT}/fts/articles/body"


def test_error_codes_are_kept_only_when_code_shaped() -> None:
    assert diag.error_code({"error": {"code": "rate_limited"}}) == "rate_limited"
    assert diag.error_code({"error": "40001"}) == "40001"
    assert diag.error_code({"error": "busy"}) == "busy"
    assert diag.error_code({"error": "E12345"}) is None, "a word code has at most 4 digits"
    assert diag.error_code({"error": "connection reset by peer"}) is None
    assert diag.error_code({"error": {"message": "x"}}) is None
    assert diag.error_code("plain text") is None


def test_a_request_that_got_no_response_is_reported_without_a_status() -> None:
    for exc, outcome, transport in [
        (httpx.ConnectError("refused"), "error", "not_sent"),
        (httpx.ConnectTimeout("slow"), "timeout", "not_sent"),
        (httpx.ReadTimeout("slow"), "timeout", "no_response"),
        (httpx.RemoteProtocolError("reset"), "error", "no_response"),
    ]:
        engine = Engine(exc)
        db = client(engine, diagnostics=True)
        with pytest.raises(OCError):
            db.sql("SELECT 1")
        db.flush_diagnostics()
        (e,) = engine.events
        assert (e["outcome"], e["transport"]) == (outcome, transport), type(exc).__name__
        assert "http_status" not in e


def test_a_callers_uuid_is_kept_and_a_non_uuid_never_reported() -> None:
    engine = Engine()
    db = client(engine, diagnostics=True)
    mine = "0B9A6C1E-2F4D-4C8B-9E7A-1D2C3B4A5F60"
    db._request("POST", f"/v1/tenants/{TENANT}/sql", json={}, headers={"x-oc-logical-request-id": mine})
    db._request("POST", f"/v1/tenants/{TENANT}/sql", json={}, headers={"X-OC-Logical-Request-Id": "order-42"})
    db.flush_diagnostics()
    assert engine.calls[0].headers["X-OC-Logical-Request-Id"] == mine
    assert engine.calls[1].headers["X-OC-Logical-Request-Id"] == "order-42"
    first, second = engine.events
    assert first["logical_request_id"] == mine.lower()
    assert UUID_RE.match(second["logical_request_id"])


def test_events_that_would_poison_a_batch_are_never_queued() -> None:
    """The engine refuses a whole batch for one invalid event."""
    base: dict[str, Any] = {
        "path": f"/v1/tenants/{TENANT}/sql", "started": time.perf_counter(),
        "logical_request_id": "8fb1dce6-72c0-4aa5-8d46-50a4e0b47ba5", "attempt": 1, "status": 200,
    }
    assert diag.event(method="HEAD", **base) is None
    assert diag.event(method="GET", **{**base, "path": "/v1/version"}) is None
    assert diag.event(method="GET", **{**base, "path": "/v1/tenants/" + "x" * 600}) is None
    old = diag.event(method="GET", **{**base, "started": time.perf_counter() - 7200})
    assert old is not None and old["duration_ms"] == 3_600_000.0
    late = diag.event(method="GET", **{**base, "attempt": 101})
    assert late is not None and "attempt" not in late
    assert "X-OC-Attempt" not in diag.with_attempt({}, 101)


def test_a_report_that_cannot_be_sent_never_fails_a_call() -> None:
    def engine(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/diagnostics"):
            raise httpx.ConnectError("offline")
        return httpx.Response(200, json={"kind": "select", "rows": []})

    db = client(engine, diagnostics=True)
    db.sql("SELECT 1")
    db.flush_diagnostics()
    db.close()


def test_close_sends_what_the_background_thread_has_not() -> None:
    engine = Engine()
    db = client(engine, diagnostics=True)
    db.sql("SELECT 1")
    db.close()
    assert len(engine.events) == 1


def test_the_background_thread_sends_within_a_second_or_two() -> None:
    engine = Engine()
    db = client(engine, diagnostics=True)
    db.sql("SELECT 1")
    deadline = time.monotonic() + 5
    while not engine.reports and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(engine.events) == 1
    db.close()


def test_the_queue_is_bounded_and_drops_the_oldest() -> None:
    q = diag.SyncReporter(lambda batch: None)
    q._stopped = True  # no thread: push() still queues through _append
    for i in range(300):
        q._append({"n": i})
    assert len(q) == 256 and q.dropped == 44
    assert q._take()[0] == {"n": 44}


def test_async_client_correlates_and_reports() -> None:
    engine = Engine((503, {"error": "busy"}), (200, {"kind": "select", "rows": []}))

    async def run() -> OCError | None:
        db = AsyncOriginChain(
            base_url="http://test.invalid", bearer="b", tenant=TENANT,
            max_retries=1, diagnostics=True,
        )
        db._client = httpx.AsyncClient(
            base_url="http://test.invalid", transport=httpx.MockTransport(engine)
        )
        db._backoff = lambda attempt: 0.0  # type: ignore[method-assign]
        await db._request("POST", f"/v1/tenants/{TENANT}/sql", json={"sql": "SELECT 1"})
        await db.flush_diagnostics()
        await db.aclose()
        return None

    asyncio.run(run())
    first, retry = engine.calls
    assert first.headers["X-OC-Logical-Request-Id"] == retry.headers["X-OC-Logical-Request-Id"]
    assert [e["attempt"] for e in engine.events] == [1, 2]
    assert [e["outcome"] for e in engine.events] == ["error", "success"]


def test_reported_version_matches_pyproject() -> None:
    text = (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert version and version.group(1) == diag.SDK_VERSION
