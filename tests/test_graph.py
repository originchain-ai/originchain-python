"""Tests for the ``client.graph.*`` namespace.

Wire shapes from ``oc-http/src/preview_endpoints.rs``:
- ``neighbors`` / ``reverse``: GET → ``List[str]`` of neighbour PKs.
- ``bfs``: GET → ``List[{pk, depth}]``.
- ``path``: GET → ``{reachable: bool}``.
- ``dijkstra``: GET → ``{cost: float | null}`` with ``weights_json`` as
  a JSON-stringified query parameter.
"""

from __future__ import annotations

import json

import httpx

from originchain import DijkstraResult, GraphBfsHit, GraphPath, Neighbor


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
