"""Tests for ``client.sql`` / ``client.sql_one`` / ``client.sql.query`` /
``client.sql.execute``.

The wire shape is the discriminated union from
``oc-http/src/preview_endpoints.rs::SqlResp``: ``{kind: "select" |
"insert" | "delete", ...}``. We assert the SDK decodes each branch into
the right dataclass and that ``sql_one`` errors clearly for non-SELECT.
"""

from __future__ import annotations

import json
import warnings

import httpx
import pytest

from originchain import (
    OCValidationError,
    OriginChainBadRequest,
    OriginChainServerError,
    SqlDelete,
    SqlExecResult,
    SqlInsert,
    SqlResult,
    SqlSelect,
)


def test_sql_select_returns_dataclass(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path == "/v1/tenants/01HX1TESTTENANTXXXXXXXXXX1/sql"
        body = json.loads(req.content)
        assert body == {"sql": "SELECT * FROM t"}
        return httpx.Response(200, json={"kind": "select", "rows": [{"a": 1}, {"a": 2}]})

    client = mock_client(handler)
    resp = client.sql("SELECT * FROM t")
    assert isinstance(resp, SqlSelect)
    assert resp.rows == ({"a": 1}, {"a": 2})


def test_sql_insert_returns_translation(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"kind": "insert", "schema": "trading.orders", "rows": [{"order_id": "o1"}]},
        )

    client = mock_client(handler)
    resp = client.sql("INSERT INTO trading.orders ...")
    assert isinstance(resp, SqlInsert)
    assert resp.schema == "trading.orders"
    assert resp.rows == ({"order_id": "o1"},)


def test_sql_delete_returns_pk(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"kind": "delete", "schema": "trading.orders", "pk": "o1"},
        )

    client = mock_client(handler)
    resp = client.sql("DELETE FROM trading.orders WHERE order_id = 'o1'")
    assert isinstance(resp, SqlDelete)
    assert resp.schema == "trading.orders"
    assert resp.pk == "o1"


def test_sql_one_first_row(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"kind": "select", "rows": [{"a": 1}, {"a": 2}]})

    client = mock_client(handler)
    row = client.sql_one("SELECT * FROM t LIMIT 1")
    assert row == {"a": 1}


def test_sql_one_empty(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"kind": "select", "rows": []})

    client = mock_client(handler)
    assert client.sql_one("SELECT * FROM t WHERE 1=0") is None


def test_sql_one_rejects_non_select(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"kind": "delete", "schema": "t", "pk": "x"}
        )

    client = mock_client(handler)
    with pytest.raises(OCValidationError):
        client.sql_one("DELETE FROM t WHERE id='x'")


# ─────────────────────── Typed-namespace v1 surface ───────────────────────
# `client.sql.query` / `client.sql.execute`. The legacy callable
# `client.sql("...")` still works (covered by tests above); the new
# methods decode the response into the richer `SqlResult` /
# `SqlExecResult` shapes that surface columns + rows_affected for the
# spec-shape API.


def test_sql_query_returns_sql_result(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        # `params` should NOT be in the body when caller omits it.
        assert "params" not in body
        return httpx.Response(
            200,
            json={
                "kind": "select",
                "rows": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
            },
        )

    client = mock_client(handler)
    out = client.sql.query("SELECT a, b FROM t")
    assert isinstance(out, SqlResult)
    assert out.rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    # When the server didn't emit a `columns` array, SDK derives one
    # from the first row's key order so callers always get something.
    assert out.columns == ["a", "b"]


# ── Bind parameters ───────────────────────────────────────────────────
# The `/sql` handler binds POSITIONALLY — `params` is a JSON array that
# `$1` / `$2` index into. 0.5.x sent a named map, which the engine has no
# binder for, so every parameterised query failed. Source of truth: the
# TypeScript client (`params?: unknown[]`) and the Go client (variadic
# `params ...any` marshalled straight into `body["params"]`).


def test_sql_query_forwards_params_as_positional_array(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"kind": "select", "rows": []})

    client = mock_client(handler)
    client.sql.query(
        "SELECT * FROM t WHERE a = $1 AND b = $2",
        params=["AAPL", 50],
    )
    assert seen["body"] == {
        "sql": "SELECT * FROM t WHERE a = $1 AND b = $2",
        "params": ["AAPL", 50],
    }
    # Specifically a JSON array, not an object — a map is what broke.
    assert isinstance(seen["body"]["params"], list)


def test_sql_query_params_preserve_order_from_tuple(mock_client) -> None:
    """Any sequence works, and $N ordering is the caller's order."""
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"kind": "select", "rows": []})

    client = mock_client(handler)
    client.sql.query("SELECT * FROM t WHERE a = $1 AND b = $2", params=(1, "x"))
    assert seen["body"]["params"] == [1, "x"]


def test_sql_query_empty_params_still_sends_array(mock_client) -> None:
    """`params=[]` is not the same as omitting it — an explicit empty
    array must still reach the wire so the caller's intent survives."""
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"kind": "select", "rows": []})

    client = mock_client(handler)
    client.sql.query("SELECT 1", params=[])
    assert seen["body"] == {"sql": "SELECT 1", "params": []}


def test_sql_query_rejects_named_param_mapping(mock_client) -> None:
    """A dict is refused client-side with an actionable message rather
    than being flattened into a guessed positional order."""
    called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"kind": "select", "rows": []})

    client = mock_client(handler)
    with pytest.raises(OCValidationError) as exc:
        client.sql.query("SELECT * FROM t WHERE a = :a", params={"a": 1})
    msg = str(exc.value)
    assert "positional" in msg
    assert "$1" in msg
    # No request should have been issued.
    assert not called


def test_sql_query_rejects_bare_string_params(mock_client) -> None:
    """`params="AAPL"` would bind four one-character parameters if the
    str were treated as a sequence."""
    client = mock_client(
        lambda req: httpx.Response(200, json={"kind": "select", "rows": []})
    )
    with pytest.raises(OCValidationError) as exc:
        client.sql.query("SELECT * FROM t WHERE s = $1", params="AAPL")
    assert "list or tuple" in str(exc.value)


def test_sql_query_rejects_non_select(mock_client) -> None:
    # query() expects SELECT; an insert-translation kind should surface
    # a typed validation error rather than silently returning empty rows.
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"kind": "insert", "schema": "t", "rows": [{"a": 1}]}
        )

    client = mock_client(handler)
    with pytest.raises(OCValidationError):
        client.sql.query("INSERT INTO t (a) VALUES (1)")


def test_sql_query_error_400(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(400, json={"error": "bad sql"}))
    with pytest.raises(OriginChainBadRequest):
        client.sql.query("nonsense")


def test_sql_query_error_500(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(OriginChainServerError):
        client.sql.query("SELECT 1")


def test_sql_execute_insert(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"kind": "insert", "schema": "t", "rows": [{"a": 1}]},
        )

    client = mock_client(handler)
    out = client.sql.execute("INSERT INTO t (a) VALUES (1)")
    assert isinstance(out, SqlExecResult)
    assert out.kind == "insert"
    assert out.schema == "t"
    assert out.rows_affected == 1


def test_sql_execute_delete(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"kind": "delete", "schema": "t", "pk": "row-1"}
        )

    client = mock_client(handler)
    out = client.sql.execute("DELETE FROM t WHERE id='row-1'")
    assert out.kind == "delete"
    assert out.schema == "t"
    assert out.rows_affected == 1


def test_sql_execute_error_400(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(400, json={"error": "bad sql"}))
    with pytest.raises(OriginChainBadRequest):
        client.sql.execute("nonsense")


# ─────────────────────── 0.5 additions (2026-06-08) ───────────────────────
# Materialized views: install / refresh / read.


from originchain import (
    MaterializedViewInstallResult,
    MaterializedViewRefreshResult,
    MaterializedViewRows,
)


def test_sql_install_materialized_view(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path.endswith("/sql/materialized-views")
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "name": "daily_orders",
                "rows_materialized": 4321,
                "bytes_written": 819200,
                "refresh_ts": 1717804800,
            },
        )

    client = mock_client(handler)
    out = client.sql.install_materialized_view(
        "daily_orders",
        "SELECT order_id, qty FROM trading.orders",
        refresh_mode="on_demand",
    )
    assert isinstance(out, MaterializedViewInstallResult)
    assert out.name == "daily_orders"
    assert out.rows_materialized == 4321
    assert out.bytes_written == 819200
    assert out.refresh_ts == 1717804800
    assert seen["body"]["name"] == "daily_orders"
    assert seen["body"]["query"].startswith("SELECT")
    assert seen["body"]["refresh_mode"] == "on_demand"
    # `source_schema` is omitted from the body when caller doesn't
    # supply it — server derives it from the plan.
    assert "source_schema" not in seen["body"]


def test_sql_install_materialized_view_with_source_schema(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "name": "v1",
                "rows_materialized": 0,
                "bytes_written": 0,
                "refresh_ts": 1717804800,
            },
        )

    client = mock_client(handler)
    client.sql.install_materialized_view(
        "v1",
        "SELECT * FROM trading.orders",
        source_schema="trading.orders",
    )
    assert seen["body"]["source_schema"] == "trading.orders"


# ── refresh_mode wire values ──────────────────────────────────────────
# The engine's `RefreshMode` serde enum is `#[serde(rename_all =
# "snake_case")]` over `OnDemand | Incremental`, so the only two values
# it accepts are `on_demand` and `incremental`; unknown values are a hard
# 400, not a silent fallback. 0.5.x sent `manual` / `on_write`, so every
# install failed. Source of truth: `oc-query/src/materialized_view.rs`.


def _mv_body_capturing_client(mock_client, seen: dict):
    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "name": "v1",
                "rows_materialized": 0,
                "bytes_written": 0,
                "refresh_ts": 1717804800,
            },
        )

    return mock_client(handler)


def test_sql_install_materialized_view_defaults_to_on_demand(mock_client) -> None:
    seen: dict = {}
    client = _mv_body_capturing_client(mock_client, seen)
    client.sql.install_materialized_view("v1", "SELECT * FROM trading.orders")
    assert seen["body"]["refresh_mode"] == "on_demand"


def test_sql_install_materialized_view_incremental(mock_client) -> None:
    seen: dict = {}
    client = _mv_body_capturing_client(mock_client, seen)
    client.sql.install_materialized_view(
        "v1", "SELECT * FROM trading.orders", refresh_mode="incremental"
    )
    assert seen["body"]["refresh_mode"] == "incremental"


@pytest.mark.parametrize(
    ("legacy", "wire"),
    [("manual", "on_demand"), ("on_write", "incremental")],
)
def test_sql_install_materialized_view_legacy_mode_alias(
    mock_client, legacy: str, wire: str
) -> None:
    """The dead 0.5.x names map onto the real wire values and warn."""
    seen: dict = {}
    client = _mv_body_capturing_client(mock_client, seen)
    with pytest.warns(DeprecationWarning, match=legacy):
        client.sql.install_materialized_view(
            "v1", "SELECT * FROM trading.orders", refresh_mode=legacy
        )
    assert seen["body"]["refresh_mode"] == wire


def test_sql_install_materialized_view_rejects_unknown_mode(mock_client) -> None:
    """Unknown modes are refused locally — the engine 400s on them, and
    a client-side error names the valid values."""
    called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = mock_client(handler)
    with pytest.raises(OCValidationError) as exc:
        client.sql.install_materialized_view(
            "v1", "SELECT * FROM trading.orders", refresh_mode="on_read"
        )
    msg = str(exc.value)
    assert "on_demand" in msg and "incremental" in msg
    assert not called


def test_sql_install_materialized_view_never_sends_dead_values(mock_client) -> None:
    """Regression guard: `manual` / `on_write` must never reach the wire
    under any input — that's the bug this release fixes."""
    for supplied in ("on_demand", "incremental", "manual", "on_write"):
        seen: dict = {}
        client = _mv_body_capturing_client(mock_client, seen)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            client.sql.install_materialized_view(
                "v1", "SELECT * FROM trading.orders", refresh_mode=supplied
            )
        assert seen["body"]["refresh_mode"] in ("on_demand", "incremental")


def test_sql_install_materialized_view_error_400(mock_client) -> None:
    client = mock_client(
        lambda req: httpx.Response(
            400, json={"error": "materialized view query: parse error"}
        )
    )
    with pytest.raises(OriginChainBadRequest):
        client.sql.install_materialized_view("bad", "NOT SQL")


def test_sql_refresh_materialized_view(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path.endswith(
            "/sql/materialized-views/daily_orders/refresh"
        )
        return httpx.Response(
            200,
            json={
                "name": "daily_orders",
                "rows_materialized": 4500,
                "bytes_written": 856000,
                "refresh_ts": 1717891200,
            },
        )

    client = mock_client(handler)
    out = client.sql.refresh_materialized_view("daily_orders")
    assert isinstance(out, MaterializedViewRefreshResult)
    assert out.name == "daily_orders"
    assert out.rows_materialized == 4500


def test_sql_refresh_materialized_view_not_found(mock_client) -> None:
    # 404 when the view isn't installed. Surfaces as OCNotFoundError;
    # the test just verifies a 4xx-class error propagates.
    from originchain import OCNotFoundError

    client = mock_client(
        lambda req: httpx.Response(
            404, json={"error": "materialized view not found"}
        )
    )
    with pytest.raises(OCNotFoundError):
        client.sql.refresh_materialized_view("missing")


def test_sql_read_materialized_view(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path.endswith("/sql/materialized-views/daily_orders")
        return httpx.Response(
            200,
            json={
                "name": "daily_orders",
                "rows": [
                    {"order_id": "o1", "qty": 100},
                    {"order_id": "o2", "qty": 200},
                ],
            },
        )

    client = mock_client(handler)
    out = client.sql.read_materialized_view("daily_orders")
    assert isinstance(out, MaterializedViewRows)
    assert out.name == "daily_orders"
    assert len(out.rows) == 2
    assert out.rows[0] == {"order_id": "o1", "qty": 100}


def test_sql_read_materialized_view_error_500(mock_client) -> None:
    client = mock_client(
        lambda req: httpx.Response(500, json={"error": "snapshot decode failed"})
    )
    with pytest.raises(OriginChainServerError):
        client.sql.read_materialized_view("daily_orders")


def test_sql_callable_back_compat(mock_client) -> None:
    # The pre-namespace API: `client.sql("SELECT ...")` returning the
    # tagged-union dataclass. Must keep working after `client.sql`
    # became a namespace instance.
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"kind": "select", "rows": [{"a": 1}]})

    client = mock_client(handler)
    resp = client.sql("SELECT * FROM t")
    assert isinstance(resp, SqlSelect)
    assert resp.rows == ({"a": 1},)
