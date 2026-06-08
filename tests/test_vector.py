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
        return httpx.Response(200, json={"deleted": True})

    client = mock_client(handler)
    out = client.vector.delete("embeddings", "doc-1")
    assert seen["method"] == "DELETE"
    assert seen["path"].endswith("/vector/embeddings/doc-1")
    assert out.deleted is True


def test_vector_ns_install_centroids(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        # 0.5: URL is `install-centroids` (hyphen) to match the
        # engine's admin-route convention. The underscore path the
        # 0.4 SDK used returned 404 against the deployed engine.
        assert req.url.path.endswith("/vector/embeddings/install-centroids")
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


# ─────────────────────── 0.5 additions (2026-06-08) ───────────────────────
# `delete_bulk`, `train_and_install_centroids`, `centroids`,
# `rebalance_status`, and the post-handler-shipped `delete` end-to-end
# verification (handler was missing in 0.4).


from originchain import (
    CentroidsPreview,
    IvfRebalanceStatus,
    TrainAndInstallCentroidsResult,
    VectorDeleteBulkResult,
    VectorDeleteResult,
)


def test_vector_ns_delete_missing_row_is_idempotent(mock_client) -> None:
    # Missing-row case returns 200 + {"deleted": false}, NOT 404.
    # Lets cleanup paths call delete unconditionally without try/except.
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"deleted": False})

    client = mock_client(handler)
    out = client.vector.delete("embeddings", "never-existed")
    assert isinstance(out, VectorDeleteResult)
    assert out.deleted is False


def test_vector_ns_delete_with_index_and_repair(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params.get("index") == "ivf"
        assert req.url.params.get("repair") == "true"
        return httpx.Response(200, json={"deleted": True})

    client = mock_client(handler)
    out = client.vector.delete("embeddings", "doc-1", index="ivf", repair=True)
    assert out.deleted is True


def test_vector_ns_delete_bulk(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path.endswith("/vector/embeddings/delete-bulk")
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200, json={"deleted_count": 2, "missing_count": 1}
        )

    client = mock_client(handler)
    out = client.vector.delete_bulk("embeddings", ["a", "b", "c"])
    assert isinstance(out, VectorDeleteBulkResult)
    assert out.deleted_count == 2
    assert out.missing_count == 1
    assert seen["body"]["ids"] == ["a", "b", "c"]
    # Defaults: no `index` / no `repair` keys when caller omits them.
    assert "index" not in seen["body"]
    assert "repair" not in seen["body"]


def test_vector_ns_delete_bulk_with_repair(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"deleted_count": 1, "missing_count": 0})

    client = mock_client(handler)
    client.vector.delete_bulk(
        "embeddings", ["doc-1"], index="hnsw", repair=True
    )
    assert seen["body"]["index"] == "hnsw"
    assert seen["body"]["repair"] is True


def test_vector_ns_delete_bulk_error_400(mock_client) -> None:
    client = mock_client(
        lambda req: httpx.Response(400, json={"error": "too many ids"})
    )
    with pytest.raises(OriginChainBadRequest):
        client.vector.delete_bulk("embeddings", ["x"] * 99999)


def test_vector_ns_train_and_install_centroids(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path.endswith(
            "/vector/embeddings/train-and-install-centroids"
        )
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "trained": True,
                "installed": True,
                "partitions": 8,
                "dim": 128,
                "iterations": 42,
                "converged": True,
                "last_max_shift": 7.3e-5,
                "training_corpus_size": 10000,
            },
        )

    client = mock_client(handler)
    out = client.vector.train_and_install_centroids(
        "embeddings",
        partitions=8,
        init="kmeans_plus_plus",
        max_iterations=100,
        seed=42,
    )
    assert isinstance(out, TrainAndInstallCentroidsResult)
    assert out.trained is True
    assert out.installed is True
    assert out.partitions == 8
    assert out.iterations == 42
    assert out.converged is True
    assert out.training_corpus_size == 10000
    assert seen["body"]["partitions"] == 8
    assert seen["body"]["init"] == "kmeans_plus_plus"
    assert seen["body"]["max_iterations"] == 100
    assert seen["body"]["seed"] == 42


def test_vector_ns_train_and_install_centroids_minimal_body(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "trained": True,
                "installed": True,
                "partitions": 4,
                "dim": 3,
                "iterations": 5,
                "converged": False,
                "last_max_shift": 0.1,
                "training_corpus_size": 100,
            },
        )

    client = mock_client(handler)
    client.vector.train_and_install_centroids("embeddings", partitions=4)
    # Only `partitions` should land in the body when no optional knob
    # is supplied — keeps the wire request minimal and lets server-side
    # defaults govern training.
    assert seen["body"] == {"partitions": 4}


def test_vector_ns_train_and_install_centroids_error_400(mock_client) -> None:
    # Under-population guard: server returns 400 when count < partitions*4.
    client = mock_client(
        lambda req: httpx.Response(
            400, json={"error": "not enough vectors to train 8 partitions"}
        )
    )
    with pytest.raises(OriginChainBadRequest):
        client.vector.train_and_install_centroids("embeddings", partitions=8)


def test_vector_ns_centroids_installed(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path.endswith("/vector/embeddings/centroids")
        return httpx.Response(
            200,
            json={
                "installed": True,
                "partitions": 2,
                "dim": 3,
                "centroids_preview": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
            },
        )

    client = mock_client(handler)
    out = client.vector.centroids("embeddings")
    assert isinstance(out, CentroidsPreview)
    assert out.installed is True
    assert out.partitions == 2
    assert out.dim == 3
    assert out.centroids_preview == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_vector_ns_centroids_not_installed(mock_client) -> None:
    # `installed=False` is a 200, not a 404 — the route is meaningful
    # for any table; centroids are optional state.
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "installed": False,
                "partitions": 0,
                "dim": 0,
                "centroids_preview": [],
            },
        )

    client = mock_client(handler)
    out = client.vector.centroids("embeddings")
    assert out.installed is False
    assert out.centroids_preview == []


def test_vector_ns_rebalance_status(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path.endswith(
            "/vector/embeddings/ivf-rebalance-status"
        )
        return httpx.Response(
            200,
            json={
                "total_live": 1000,
                "partitions": 4,
                "live_per_cell": [250, 250, 250, 250],
                "skew": 1.0,
                "action": "None",
            },
        )

    client = mock_client(handler)
    out = client.vector.rebalance_status("embeddings")
    assert isinstance(out, IvfRebalanceStatus)
    assert out.total_live == 1000
    assert out.partitions == 4
    assert out.live_per_cell == [250, 250, 250, 250]
    assert out.skew == 1.0
    # SDK normalises the serde-tagged action variant to lowercase.
    assert out.action == "none"


def test_vector_ns_rebalance_status_recommended(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "total_live": 1000,
                "partitions": 4,
                "live_per_cell": [700, 100, 100, 100],
                "skew": 2.8,
                "action": "Recommended",
            },
        )

    client = mock_client(handler)
    out = client.vector.rebalance_status("embeddings")
    assert out.skew > 2.0
    assert out.action == "recommended"


def test_vector_ns_rebalance_status_503(mock_client) -> None:
    # 503 when no centroids have been installed (table isn't IVF /
    # hasn't been bootstrapped). Surfaces as OriginChainServerError.
    client = mock_client(
        lambda req: httpx.Response(
            503, json={"error": "no IVF centroids installed"}
        )
    )
    with pytest.raises(OriginChainServerError):
        client.vector.rebalance_status("embeddings")
