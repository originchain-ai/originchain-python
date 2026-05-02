"""Tests for ``client.vector_put`` / ``client.vector_topk``.

Wire shapes come from ``oc-http/src/preview_endpoints.rs::VecPutReq /
VecTopkReq / VecHit``. ``put`` is a 201 No-Content; ``topk`` returns a
JSON array of ``{id, score}``.
"""

from __future__ import annotations

import json

import httpx

from originchain import VectorHit


def test_vector_put_serializes_body(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert (
            req.url.path
            == "/v1/tenants/01HX1TESTTENANTXXXXXXXXXX1/vector/embeddings/put"
        )
        seen["body"] = json.loads(req.content)
        return httpx.Response(201)

    client = mock_client(handler)
    client.vector_put(
        "embeddings",
        id="doc-1",
        embedding=[0.1, 0.2, 0.3],
        dim=3,
        metric="cosine",
        metadata={"tag": "alpha"},
    )
    assert seen["body"] == {
        "id": "doc-1",
        "embedding": [0.1, 0.2, 0.3],
        "dim": 3,
        "metric": "cosine",
        "metadata": {"tag": "alpha"},
    }


def test_vector_topk_decodes_hits(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        # `mode` is optional on the wire; absent = server default
        # ("high_recall"). The kwarg is None by default, so the field
        # must not appear in the request body.
        assert "mode" not in body
        assert body["k"] == 5
        assert body["dim"] == 3
        return httpx.Response(
            200,
            json=[{"id": "a", "score": 0.95}, {"id": "b", "score": 0.71}],
        )

    client = mock_client(handler)
    hits = client.vector_topk(
        "embeddings",
        query=[0.1, 0.2, 0.3],
        k=5,
        dim=3,
        metric="cosine",
    )
    assert hits == [VectorHit(id="a", score=0.95), VectorHit(id="b", score=0.71)]


def test_vector_topk_with_filter(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["filter"] == {"tag": "alpha"}
        return httpx.Response(200, json=[{"id": "a", "score": 0.9}])

    client = mock_client(handler)
    hits = client.vector_topk(
        "embeddings",
        query=[0.0, 1.0, 0.0],
        k=3,
        dim=3,
        filter={"tag": "alpha"},
    )
    assert len(hits) == 1


def test_vector_topk_mode_fast(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["mode"] == "fast"
        return httpx.Response(200, json=[{"id": "a", "score": 0.9}])

    client = mock_client(handler)
    hits = client.vector_topk(
        "embeddings",
        query=[0.1, 0.2, 0.3],
        k=5,
        dim=3,
        mode="fast",
    )
    assert len(hits) == 1


def test_vector_topk_mode_high_recall(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["mode"] == "high_recall"
        return httpx.Response(200, json=[{"id": "a", "score": 0.9}])

    client = mock_client(handler)
    hits = client.vector_topk(
        "embeddings",
        query=[0.1, 0.2, 0.3],
        k=5,
        dim=3,
        mode="high_recall",
    )
    assert len(hits) == 1
