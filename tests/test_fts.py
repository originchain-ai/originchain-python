"""Tests for ``client.fts_index`` / ``client.fts_search``.

Boolean / phrase modes return ``List[str]`` of doc_ids; BM25 returns
``List[{doc_id, score}]``. The SDK normalises both into ``FtsHit``
(score=0.0 for boolean / phrase).
"""

from __future__ import annotations

import json

import httpx

from originchain import FtsHit


def test_fts_index_posts_doc(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert (
            req.url.path
            == "/v1/tenants/01HX1TESTTENANTXXXXXXXXXX1/fts/articles/body"
        )
        body = json.loads(req.content)
        assert body == {"doc_id": "d1", "text": "the quick brown fox"}
        return httpx.Response(201)

    client = mock_client(handler)
    client.fts_index("articles", "body", doc_id="d1", text="the quick brown fox")


def test_fts_search_boolean_returns_doc_ids(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.params["q"] == "quick brown"
        assert req.url.params["mode"] == "boolean"
        return httpx.Response(200, json=["d1", "d2"])

    client = mock_client(handler)
    hits = client.fts_search("articles", "body", q="quick brown", mode="boolean")
    assert hits == [FtsHit(doc_id="d1"), FtsHit(doc_id="d2")]
    assert all(h.score == 0.0 for h in hits)


def test_fts_search_bm25_returns_ranked(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["mode"] == "bm25"
        assert req.url.params["k"] == "3"
        return httpx.Response(
            200, json=[{"doc_id": "d1", "score": 4.21}, {"doc_id": "d2", "score": 2.10}]
        )

    client = mock_client(handler)
    hits = client.fts_search("articles", "body", q="quick", mode="bm25", k=3)
    assert hits == [FtsHit(doc_id="d1", score=4.21), FtsHit(doc_id="d2", score=2.10)]
