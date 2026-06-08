"""Tests for ``client.sql`` / ``client.sql_one`` / ``client.sql.query`` /
``client.sql.execute``.

The wire shape is the discriminated union from
``oc-http/src/preview_endpoints.rs::SqlResp``: ``{kind: "select" |
"insert" | "delete", ...}``. We assert the SDK decodes each branch into
the right dataclass and that ``sql_one`` errors clearly for non-SELECT.
"""

from __future__ import annotations

import json

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


def test_sql_query_forwards_params(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"kind": "select", "rows": []})

    client = mock_client(handler)
    client.sql.query("SELECT * FROM t WHERE a = :a", params={"a": 1})
    assert seen["body"] == {"sql": "SELECT * FROM t WHERE a = :a", "params": {"a": 1}}


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
