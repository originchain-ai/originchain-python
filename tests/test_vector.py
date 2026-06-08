"""Tests for ``client.vector_put`` / ``client.vector_topk`` (legacy)
and the typed-namespace v1 surface ``client.vector.put`` /
``client.vector.topk`` / ``client.vector.delete`` /
``client.vector.install_centroids``.

Wire shapes come from ``oc-http/src/preview_endpoints.rs::VecPutReq /
VecTopkReq / VecHit``. ``put`` is a 201 No-Content; ``topk`` returns a
JSON array of ``{id, score, metadata?}``.
"""

from __future__ import annotations

import json

import httpx
import pytest

from originchain import (
    InstallCentroidsResult,
    OriginChainBadRequest,
    OriginChainServerError,
    VectorHit,
    VectorHitV2,
)


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


# ─────────────────────── Typed-namespace v1 surface ───────────────────────
# `client.vector.put` / `.topk` / `.delete` / `.install_centroids`. The
# new shape derives `dim` from the embedding/query length so callers
# don't have to pass it twice, and `.topk` decodes a richer `VectorHitV2`
# that carries server-returned metadata.


def test_vector_ns_put_derives_dim(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path.endswith("/vector/embeddings/put")
        seen["body"] = json.loads(req.content)
        return httpx.Response(201)

    client = mock_client(handler)
    client.vector.put("embeddings", "doc-1", [0.1, 0.2, 0.3, 0.4])
    assert seen["body"]["id"] == "doc-1"
    assert seen["body"]["embedding"] == [0.1, 0.2, 0.3, 0.4]
    # `dim` is derived from len(embedding) — caller never passes it.
    assert seen["body"]["dim"] == 4


def test_vector_ns_put_with_metadata(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(201)

    client = mock_client(handler)
    client.vector.put("embeddings", "doc-1", [0.0, 1.0], metadata={"tag": "alpha"})
    assert seen["body"]["metadata"] == {"tag": "alpha"}


def test_vector_ns_topk_decodes_v2_hits(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["dim"] == 3
        assert body["k"] == 5
        assert body["metric"] == "cosine"
        return httpx.Response(
            200,
            json=[
                {"id": "a", "score": 0.95, "metadata": {"src": "wiki"}},
                {"id": "b", "score": 0.71},
            ],
        )

    client = mock_client(handler)
    hits = client.vector.topk("embeddings", [0.1, 0.2, 0.3], k=5)
    assert len(hits) == 2
    assert isinstance(hits[0], VectorHitV2)
    assert hits[0].vec_id == "a"
    assert hits[0].id == "a"  # backward-compat alias
    assert hits[0].score == 0.95
    assert hits[0].metadata == {"src": "wiki"}
    assert hits[1].metadata is None


def test_vector_ns_topk_with_filter_and_nprobe(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["filter"] == {"tag": "alpha"}
        assert body["ivf_nprobe"] == 8
        return httpx.Response(200, json=[])

    client = mock_client(handler)
    out = client.vector.topk(
        "embeddings",
        [0.1, 0.2, 0.3],
        k=3,
        metric="dot",
        filter={"tag": "alpha"},
        nprobe=8,
    )
    assert out == []


def test_vector_ns_delete(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["path"] = req.url.path
        return httpx.Response(204)

    client = mock_client(handler)
    client.vector.delete("embeddings", "doc-1")
    assert seen["method"] == "DELETE"
    assert seen["path"].endswith("/vector/embeddings/doc-1")


def test_vector_ns_install_centroids(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/vector/embeddings/install_centroids")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"installed": True, "partitions": 2, "dim": 3})

    client = mock_client(handler)
    out = client.vector.install_centroids(
        "embeddings", [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    )
    assert isinstance(out, InstallCentroidsResult)
    assert out.installed is True
    assert out.partitions == 2
    assert out.dim == 3
    assert seen["body"]["centroids"] == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


def test_vector_ns_put_error_400(mock_client) -> None:
    client = mock_client(
        lambda req: httpx.Response(400, json={"error": "dim mismatch"})
    )
    with pytest.raises(OriginChainBadRequest):
        client.vector.put("embeddings", "doc-1", [0.0, 1.0])


def test_vector_ns_topk_error_500(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(OriginChainServerError):
        client.vector.topk("embeddings", [0.0, 1.0], k=1)


def test_vector_ns_delete_error_400(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(400, json={"error": "bad id"}))
    with pytest.raises(OriginChainBadRequest):
        client.vector.delete("embeddings", "")


def test_vector_ns_install_centroids_error_400(mock_client) -> None:
    client = mock_client(
        lambda req: httpx.Response(400, json={"error": "dim mismatch"})
    )
    with pytest.raises(OriginChainBadRequest):
        client.vector.install_centroids("embeddings", [[0.0, 0.0], [1.0]])
