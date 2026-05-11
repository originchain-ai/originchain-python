"""Auto-Idempotency-Key wiring.

Regression guard: every mutating SDK call must attach an Idempotency-Key
header so a network retry of the same logical call dedupes on the
engine. The engine's idempotency cache is LRU-bounded (10k entries +
24h TTL) so fresh-per-call UUIDs are safe.
"""
from __future__ import annotations

import re

import httpx

from originchain.client import _MUTATING_METHODS, _new_idempotency_key


UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def test_new_idempotency_key_is_unique_hex32() -> None:
    a = _new_idempotency_key()
    b = _new_idempotency_key()
    assert UUID_HEX_RE.match(a), a
    assert UUID_HEX_RE.match(b), b
    assert a != b


def test_mutating_methods_set_includes_post_put_patch_delete() -> None:
    assert _MUTATING_METHODS == frozenset({"POST", "PUT", "PATCH", "DELETE"})


def test_vector_put_auto_attaches_idempotency_key(mock_client) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(204)

    db = mock_client(handler)
    db.vector_put(
        "embeds", id="doc-1", embedding=[0.1, 0.2, 0.3], dim=3, metric="cosine"
    )
    assert len(seen) == 1
    got = seen[0].headers.get("Idempotency-Key")
    assert got, "Idempotency-Key missing on vector_put"
    assert UUID_HEX_RE.match(got), got


def test_schemas_register_auto_attaches_idempotency_key(mock_client) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"id": "demo.users", "tenant": "t"})

    db = mock_client(handler)
    db.schemas.register("namespace = 'demo'\ntable = 'users'\nprimary_key = ['id']\n")
    assert len(seen) == 1
    assert seen[0].headers.get("Idempotency-Key"), "missing on schemas.register"


def test_rows_put_caller_key_wins(mock_client) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"ok": True})

    db = mock_client(handler)
    db.rows.put("demo.users", {"id": "u1"}, idempotency_key="caller-stable-key")
    assert seen[0].headers.get("Idempotency-Key") == "caller-stable-key"


def test_rows_put_auto_gens_when_caller_passes_none(mock_client) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"ok": True})

    db = mock_client(handler)
    db.rows.put("demo.users", {"id": "u1"})
    got = seen[0].headers.get("Idempotency-Key")
    assert got and UUID_HEX_RE.match(got), got


def test_rows_put_batch_per_chunk_keys_share_base(mock_client) -> None:
    """A single ``put_batch`` call must derive each chunk's key from a
    single base UUID — that way a partial-retry of the same call hits the
    same cache slots and the engine dedupes correctly. A regression that
    minted fresh keys per chunk would break this guarantee silently."""
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"inserted": 1})

    db = mock_client(handler)
    rows = [{"id": f"u{i}"} for i in range(3)]
    db.rows.put_batch("demo.users", rows, chunk=1)
    assert len(seen) == 3

    keys = [r.headers.get("Idempotency-Key", "") for r in seen]
    # Each chunk gets a suffix like "-c0", "-c1", ...; the prefix is the
    # shared base UUID.
    bases = {k.rsplit("-c", 1)[0] for k in keys}
    suffixes = {k.rsplit("-c", 1)[1] for k in keys}
    assert len(bases) == 1, f"chunks did not share a base: {keys}"
    assert suffixes == {"0", "1", "2"}, suffixes


def test_get_does_not_send_idempotency_key(mock_client) -> None:
    """Reads must not consume an idempotency cache slot."""
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=[])

    db = mock_client(handler)
    db.schemas.list()
    assert seen[0].method == "GET"
    assert "Idempotency-Key" not in seen[0].headers
