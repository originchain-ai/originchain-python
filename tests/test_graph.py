"""Tests for the ``client.graph.*`` namespace.

Wire shapes from ``oc-http/src/preview_endpoints.rs``:
- ``neighbors`` / ``reverse``: GET → ``List[str]`` of neighbour PKs.
- ``bfs``: GET → ``List[{pk, depth}]``.
- ``path``: GET → ``{reachable: bool}``.
- ``dijkstra``: GET → ``{cost: float | null}`` with ``weights_json`` as
  a JSON-stringified query parameter.
- ``k-shortest``: GET → ``{paths: [{nodes, cost}]}``.
- ``random-walk``: GET → ``{start, walk}``.
- ``louvain``: GET → ``{communities: [{pk, community}]}``.
- ``pagerank``: GET → ``[{pk, score}]``.
- ``label_propagation``: GET → ``[{pk, label}]`` (Plan-row shape).
- ``betweenness``: GET → ``[{pk, betweenness}]`` (Plan-row shape).
"""

from __future__ import annotations

import json

import httpx
import pytest

from originchain import (
    DijkstraResult,
    GraphBfsHit,
    GraphPath,
    Neighbor,
    OriginChainBadRequest,
    OriginChainServerError,
    Path,
)


def test_graph_neighbors(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert (
            req.url.path
            == "/v1/tenants/01HX1TESTTENANTXXXXXXXXXX1/graph/users/neighbors"
        )
        assert req.url.params["rel"] == "follows"
        assert req.url.params["pk"] == "u1"
        return httpx.Response(200, json=["u2", "u3"])

    client = mock_client(handler)
    out = client.graph.neighbors("users", rel="follows", pk="u1")
    assert out == [Neighbor(pk="u2", depth=1), Neighbor(pk="u3", depth=1)]


def test_graph_reverse_neighbors(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/reverse")
        return httpx.Response(200, json=["u4"])

    client = mock_client(handler)
    out = client.graph.reverse_neighbors("users", rel="follows", pk="u1")
    assert out == [Neighbor(pk="u4", depth=1)]


def test_graph_bfs_decodes_depth(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["max_depth"] == "2"
        return httpx.Response(
            200,
            json=[{"pk": "u2", "depth": 1}, {"pk": "u3", "depth": 2}],
        )

    client = mock_client(handler)
    out = client.graph.bfs("users", rel="follows", pk="u1", max_depth=2)
    assert out == [GraphBfsHit(pk="u2", depth=1), GraphBfsHit(pk="u3", depth=2)]


def test_graph_path_reachable(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["src"] == "u1"
        assert req.url.params["dst"] == "u9"
        return httpx.Response(200, json={"reachable": True})

    client = mock_client(handler)
    out = client.graph.path("users", rel="follows", src="u1", dst="u9")
    assert out == GraphPath(reachable=True)


def test_graph_dijkstra_serializes_weights(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/dijkstra")
        # weights_json is a JSON-stringified query parameter.
        weights = json.loads(req.url.params["weights_json"])
        assert weights == {"u1|u2": 1.5, "u2|u3": 0.5}
        return httpx.Response(200, json={"cost": 2.0})

    client = mock_client(handler)
    out = client.graph.dijkstra(
        "users",
        rel="follows",
        src="u1",
        dst="u3",
        weights={"u1|u2": 1.5, "u2|u3": 0.5},
    )
    assert out == DijkstraResult(cost=2.0)


def test_graph_dijkstra_unreachable(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"cost": None})

    client = mock_client(handler)
    out = client.graph.dijkstra(
        "users", rel="follows", src="u1", dst="u9", weights={}
    )
    assert out.cost is None


# ─────────────────────── Typed-namespace v1 surface ───────────────────────
# Positional-arg helpers + the algorithm methods (k_shortest, random_walk,
# louvain, pagerank, label_propagation, betweenness, shortest_path).


def test_graph_neighbors_of_positional(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["rel"] == "follows"
        assert req.url.params["pk"] == "u1"
        return httpx.Response(200, json=["u2", "u3"])

    client = mock_client(handler)
    out = client.graph.neighbors_of("users", "u1", "follows")
    assert out == ["u2", "u3"]


def test_graph_bfs_of_positional(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["max_depth"] == "5"
        return httpx.Response(
            200, json=[{"pk": "u2", "depth": 1}, {"pk": "u3", "depth": 2}]
        )

    client = mock_client(handler)
    out = client.graph.bfs_of("users", "u1", "follows")
    assert out == ["u2", "u3"]


def test_graph_k_shortest(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/graph/users/k-shortest")
        assert req.url.params["source"] == "u1"
        assert req.url.params["target"] == "u3"
        assert req.url.params["k"] == "3"
        return httpx.Response(
            200,
            json={
                "paths": [
                    {"nodes": ["u1", "u2", "u3"], "cost": 2.0},
                    {"nodes": ["u1", "u4", "u3"], "cost": 3.5},
                ]
            },
        )

    client = mock_client(handler)
    paths = client.graph.k_shortest("users", "u1", "u3", "follows", k=3)
    assert paths == [
        Path(nodes=["u1", "u2", "u3"], cost=2.0),
        Path(nodes=["u1", "u4", "u3"], cost=3.5),
    ]


def test_graph_k_shortest_with_weight_col(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["weight_col"] == "edge_w"
        return httpx.Response(200, json={"paths": []})

    client = mock_client(handler)
    out = client.graph.k_shortest(
        "users", "u1", "u9", "follows", k=5, weight_col="edge_w"
    )
    assert out == []


def test_graph_shortest_path(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        # shortest_path delegates to k_shortest(k=1).
        assert req.url.params["k"] == "1"
        return httpx.Response(
            200, json={"paths": [{"nodes": ["u1", "u2", "u3"], "cost": 2.0}]}
        )

    client = mock_client(handler)
    out = client.graph.shortest_path("users", "u1", "u3", "follows")
    assert out == ["u1", "u2", "u3"]


def test_graph_shortest_path_unreachable(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"paths": []})

    client = mock_client(handler)
    out = client.graph.shortest_path("users", "u1", "u9", "follows")
    assert out is None


def test_graph_random_walk_unbiased(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/graph/users/random-walk")
        assert req.url.params["steps"] == "5"
        assert req.url.params["seed"] == "42"
        # p=q=1.0 is the unbiased identity; SDK omits both from the wire.
        assert "p" not in req.url.params
        assert "q" not in req.url.params
        return httpx.Response(200, json={"start": "u1", "walk": ["u1", "u2", "u3"]})

    client = mock_client(handler)
    out = client.graph.random_walk("users", "u1", "follows", steps=5, seed=42)
    assert out == ["u1", "u2", "u3"]


def test_graph_random_walk_biased(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["p"] == "0.5"
        assert req.url.params["q"] == "2.0"
        return httpx.Response(200, json={"start": "u1", "walk": ["u1", "u2"]})

    client = mock_client(handler)
    out = client.graph.random_walk(
        "users", "u1", "follows", steps=2, seed=7, p=0.5, q=2.0
    )
    assert out == ["u1", "u2"]


def test_graph_louvain(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/graph/users/louvain")
        return httpx.Response(
            200,
            json={
                "communities": [
                    {"pk": "u1", "community": 0},
                    {"pk": "u2", "community": 0},
                    {"pk": "u3", "community": 1},
                ]
            },
        )

    client = mock_client(handler)
    out = client.graph.louvain("users", "follows")
    assert out == {"u1": 0, "u2": 0, "u3": 1}


def test_graph_pagerank(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/graph/users/pagerank")
        assert req.url.params["damping"] == "0.85"
        assert req.url.params["nodes"] == "u1,u2,u3"
        return httpx.Response(
            200,
            json=[
                {"pk": "u1", "score": 0.5},
                {"pk": "u2", "score": 0.3},
                {"pk": "u3", "score": 0.2},
            ],
        )

    client = mock_client(handler)
    out = client.graph.pagerank("users", "follows", nodes=["u1", "u2", "u3"])
    assert out == {"u1": 0.5, "u2": 0.3, "u3": 0.2}


def test_graph_label_propagation(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["seed"] == "7"
        return httpx.Response(
            200,
            json=[
                {"pk": "u1", "label": 0},
                {"pk": "u2", "label": 0},
                {"pk": "u3", "label": 1},
            ],
        )

    client = mock_client(handler)
    out = client.graph.label_propagation("users", "follows", seed=7)
    assert out == {"u1": 0, "u2": 0, "u3": 1}


def test_graph_label_propagation_array_pk(mock_client) -> None:
    # The Plan-row variant may emit `pk` as a list (the substrate's
    # PK array shape). Verify the SDK stringifies it for the dict key.
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=[{"pk": ["u1", "alpha"], "label": 2}]
        )

    client = mock_client(handler)
    out = client.graph.label_propagation("users", "follows", seed=1)
    assert json.loads(next(iter(out))) == ["u1", "alpha"]
    assert next(iter(out.values())) == 2


def test_graph_betweenness(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/graph/users/betweenness")
        return httpx.Response(
            200,
            json=[
                {"pk": "u2", "betweenness": 1.5},
                {"pk": "u1", "betweenness": 0.5},
            ],
        )

    client = mock_client(handler)
    out = client.graph.betweenness("users", "follows")
    assert out == {"u2": 1.5, "u1": 0.5}


def test_graph_betweenness_with_max_nodes(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["max_nodes"] == "100"
        return httpx.Response(200, json=[])

    client = mock_client(handler)
    out = client.graph.betweenness("users", "follows", max_nodes=100)
    assert out == {}


def test_graph_ns_error_400(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(400, json={"error": "no rel"}))
    with pytest.raises(OriginChainBadRequest):
        client.graph.k_shortest("users", "u1", "u3", "missing-rel", k=1)


def test_graph_ns_error_500(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(OriginChainServerError):
        client.graph.pagerank("users", "follows", nodes=["u1"])


def test_graph_ns_louvain_error_400(mock_client) -> None:
    client = mock_client(
        lambda req: httpx.Response(400, json={"error": "too many nodes"})
    )
    with pytest.raises(OriginChainBadRequest):
        client.graph.louvain("users", "follows")


def test_graph_ns_random_walk_error_400(mock_client) -> None:
    client = mock_client(
        lambda req: httpx.Response(400, json={"error": "steps too high"})
    )
    with pytest.raises(OriginChainBadRequest):
        client.graph.random_walk("users", "u1", "follows", steps=99999, seed=1)


def test_graph_ns_label_propagation_error_500(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(OriginChainServerError):
        client.graph.label_propagation("users", "follows", seed=1)


def test_graph_ns_betweenness_error_400(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(400, json={"error": "graph too big"}))
    with pytest.raises(OriginChainBadRequest):
        client.graph.betweenness("users", "follows")


def test_graph_ns_neighbors_of_error_400(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(400, json={"error": "no rel"}))
    with pytest.raises(OriginChainBadRequest):
        client.graph.neighbors_of("users", "u1", "no-such-rel")


def test_graph_ns_bfs_of_error_400(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(400, json={"error": "no rel"}))
    with pytest.raises(OriginChainBadRequest):
        client.graph.bfs_of("users", "u1", "no-such-rel")
