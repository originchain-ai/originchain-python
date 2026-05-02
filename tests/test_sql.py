"""Tests for ``client.sql`` / ``client.sql_one``.

The wire shape is the discriminated union from
``oc-http/src/preview_endpoints.rs::SqlResp``: ``{kind: "select" |
"insert" | "delete", ...}``. We assert the SDK decodes each branch into
the right dataclass and that ``sql_one`` errors clearly for non-SELECT.
"""

from __future__ import annotations

import json

import httpx
import pytest

from originchain import OCValidationError, SqlDelete, SqlInsert, SqlSelect


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
