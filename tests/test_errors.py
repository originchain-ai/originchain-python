"""Error-mapping tests. Verify each HTTP status raises the right
exception subclass with the expected metadata. 402 is the new branch
introduced alongside the typed surfaces — assert the canonical
addon-required body shape is unpacked onto attributes.
"""

from __future__ import annotations

import httpx
import pytest

from originchain import (
    OCAuthError,
    OCNotFoundError,
    OCPaymentRequiredError,
    OCRateLimitedError,
    OCServerError,
    OCValidationError,
)


def _resp(status: int, body=None, headers=None) -> httpx.Response:
    return httpx.Response(status, json=body or {}, headers=headers or {})


def test_401_raises_auth_error(mock_client) -> None:
    client = mock_client(lambda req: _resp(401, {"error": "bad bearer"}))
    with pytest.raises(OCAuthError):
        client.sql("SELECT 1")


def test_402_raises_payment_required_with_fields(mock_client) -> None:
    body = {
        "error": "addon_required",
        "addon": "vector-search",
        "name": "Vector Search",
        "monthly_usd": 49,
        "preview": False,
        "enterprise_only": False,
        "purchase_url": "https://app.originchain.ai/billing/addons?enable=vector-search",
        "msg": "This endpoint requires the Vector Search add-on.",
    }
    client = mock_client(lambda req: _resp(402, body))
    with pytest.raises(OCPaymentRequiredError) as exc:
        client.vector_topk(
            "embeddings", query=[0.0] * 3, k=1, dim=3
        )
    assert exc.value.addon == "vector-search"
    assert exc.value.name == "Vector Search"
    assert exc.value.monthly_usd == 49
    assert exc.value.preview is False
    assert exc.value.enterprise_only is False
    assert "originchain.ai/billing" in (exc.value.purchase_url or "")


def test_404_raises_not_found(mock_client) -> None:
    client = mock_client(lambda req: _resp(404, {"error": "no such schema"}))
    with pytest.raises(OCNotFoundError):
        client.graph.neighbors("missing", rel="r", pk="x")


def test_400_raises_validation(mock_client) -> None:
    client = mock_client(lambda req: _resp(400, {"error": "bad sql"}))
    with pytest.raises(OCValidationError):
        client.sql("nonsense")


def test_429_raises_rate_limited_with_retry_after(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _resp(429, {"error": "slow down"}, headers={"Retry-After": "3"})

    client = mock_client(handler)
    with pytest.raises(OCRateLimitedError) as exc:
        client.sql("SELECT 1")
    assert exc.value.retry_after == 3.0


def test_500_raises_server_error(mock_client) -> None:
    client = mock_client(lambda req: _resp(500, {"error": "boom"}))
    with pytest.raises(OCServerError):
        client.fts_search("t", "f", q="x")
