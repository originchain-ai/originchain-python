"""Tests for ``client.fts_index`` / ``client.fts_search`` (legacy) and
``client.fts.index`` / ``client.fts.search`` / ``client.fts.install_synonyms``
/ ``client.fts.install_stopwords`` (typed-namespace v1).

Boolean / phrase modes return ``List[str]`` of doc_ids; BM25 returns
``List[{doc_id, score}]`` or the enriched
``{hits: [...], facets: {...}}`` envelope when ``highlight=True`` or
``facets=[...]`` is supplied. The SDK normalises every shape into
:class:`FtsResult` for the namespace API and :class:`FtsHit` for the
legacy methods.
"""

from __future__ import annotations

import json

import httpx
import pytest

from originchain import (
    FacetBucket,
    FtsHit,
    FtsHitWithHighlights,
    FtsResult,
    OriginChainBadRequest,
    OriginChainServerError,
)


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


# ─────────────────────── Typed-namespace v1 surface ───────────────────────
# `client.fts.index` / `.search` / `.install_synonyms` / `.install_stopwords`.
# `.search` decodes into the uniform :class:`FtsResult`, with optional
# highlight snippets + facet buckets when the caller asked for them.


def test_fts_ns_index(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(201)

    client = mock_client(handler)
    client.fts.index("articles", "body", "d1", "the quick brown fox")
    assert seen["path"].endswith("/fts/articles/body")
    assert seen["body"] == {"doc_id": "d1", "text": "the quick brown fox"}


def test_fts_ns_search_bm25_plain(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["mode"] == "bm25"
        assert req.url.params["k"] == "10"
        return httpx.Response(
            200,
            json=[{"doc_id": "d1", "score": 3.14}, {"doc_id": "d2", "score": 1.41}],
        )

    client = mock_client(handler)
    out = client.fts.search("articles", "body", "quick")
    assert isinstance(out, FtsResult)
    assert out.hits == [
        FtsHitWithHighlights(doc_id="d1", score=3.14, highlights=None),
        FtsHitWithHighlights(doc_id="d2", score=1.41, highlights=None),
    ]
    assert out.facets is None


def test_fts_ns_search_boolean_doc_ids_only(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["mode"] == "boolean"
        # boolean mode shouldn't carry a `k` parameter; it's bm25-only.
        assert "k" not in req.url.params
        return httpx.Response(200, json=["d1", "d2"])

    client = mock_client(handler)
    out = client.fts.search("articles", "body", "quick brown", mode="boolean")
    assert [h.doc_id for h in out.hits] == ["d1", "d2"]
    assert all(h.score == 0.0 for h in out.hits)


def test_fts_ns_search_phrase(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["mode"] == "phrase"
        return httpx.Response(200, json=["d1"])

    client = mock_client(handler)
    out = client.fts.search("articles", "body", "quick brown fox", mode="phrase")
    assert [h.doc_id for h in out.hits] == ["d1"]


def test_fts_ns_search_with_highlight(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["highlight"] == "true"
        return httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "doc_id": "d1",
                        "score": 4.5,
                        "highlights": {"body": ["the <em>quick</em> brown fox"]},
                    }
                ]
            },
        )

    client = mock_client(handler)
    out = client.fts.search(
        "articles", "body", "quick", mode="bm25", highlight=True
    )
    assert out.hits[0].highlights == {"body": ["the <em>quick</em> brown fox"]}
    assert out.facets is None


def test_fts_ns_search_with_facets(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["facets"] == "color,brand"
        return httpx.Response(
            200,
            json={
                "hits": [{"doc_id": "d1", "score": 1.0, "highlights": {}}],
                "facets": {
                    "color": [
                        {"value": "red", "count": 3},
                        {"value": "blue", "count": 1},
                    ]
                },
            },
        )

    client = mock_client(handler)
    out = client.fts.search(
        "articles", "body", "quick", mode="bm25", facets=["color", "brand"]
    )
    assert out.facets is not None
    assert out.facets["color"] == [
        FacetBucket(value="red", count=3),
        FacetBucket(value="blue", count=1),
    ]


def test_fts_ns_search_with_fuzzy(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["fuzzy"] == "2"
        return httpx.Response(200, json=[{"doc_id": "d1", "score": 2.0}])

    client = mock_client(handler)
    out = client.fts.search(
        "articles", "body", "quik", mode="bm25", fuzzy=2
    )
    assert out.hits[0].doc_id == "d1"


def test_fts_ns_install_synonyms(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/fts/articles/body/synonyms")
        seen["body"] = json.loads(req.content)
        return httpx.Response(201)

    client = mock_client(handler)
    client.fts.install_synonyms(
        "articles", "body", {"car": ["auto", "vehicle"]}
    )
    assert seen["body"] == {"synonyms": {"car": ["auto", "vehicle"]}}


def test_fts_ns_install_stopwords(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/fts/articles/body/stopwords")
        seen["body"] = json.loads(req.content)
        return httpx.Response(201)

    client = mock_client(handler)
    client.fts.install_stopwords("articles", "body", ["the", "a", "an"])
    assert seen["body"] == {"stopwords": ["the", "a", "an"]}


def test_fts_ns_search_error_400(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(400, json={"error": "bad q"}))
    with pytest.raises(OriginChainBadRequest):
        client.fts.search("articles", "body", "")


def test_fts_ns_index_error_500(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(OriginChainServerError):
        client.fts.index("articles", "body", "d1", "text")


def test_fts_ns_install_synonyms_error_400(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(400, json={"error": "too many"}))
    with pytest.raises(OriginChainBadRequest):
        client.fts.install_synonyms("articles", "body", {"x": ["y"]})


def test_fts_ns_install_stopwords_error_400(mock_client) -> None:
    client = mock_client(lambda req: httpx.Response(400, json={"error": "bad list"}))
    with pytest.raises(OriginChainBadRequest):
        client.fts.install_stopwords("articles", "body", ["the"])
