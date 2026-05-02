"""End-to-end smoke against a live engine on ``localhost:8080``.

Skipped unless ``OC_E2E_TEST=1`` so CI doesn't try to spin up a server.
For manual validation:

    OC_BASE_URL=http://localhost:8080 \
    OC_BEARER=devtoken \
    OC_TENANT=01HX1MYTESTTENANTXXXXXXXXXX \
    OC_E2E_TEST=1 \
    pytest sdk/python/tests/test_e2e_live.py -v

Caveats:
- The tests assume a schema named ``trading.orders`` already exists with
  a string PK ``order_id``. If it doesn't, the SQL test is skipped.
- The vector / FTS / graph tests are best-effort — they don't fail the
  suite if the relevant add-on isn't entitled (402 is treated as
  "skip, not fail" so the engine doesn't need every preview surface
  enabled to run a smoke).
"""

from __future__ import annotations

import os

import pytest

from originchain import (
    OCNotFoundError,
    OCPaymentRequiredError,
    OCValidationError,
    OriginChain,
    SqlSelect,
)

E2E = os.environ.get("OC_E2E_TEST") == "1"
pytestmark = pytest.mark.skipif(not E2E, reason="set OC_E2E_TEST=1 to run live tests")


@pytest.fixture(scope="module")
def live_client() -> OriginChain:
    return OriginChain.from_env()


def test_live_sql_select_one(live_client: OriginChain) -> None:
    try:
        resp = live_client.sql("SELECT * FROM trading.orders LIMIT 1")
    except OCNotFoundError:
        pytest.skip("schema trading.orders not registered on this engine")
    except OCValidationError:
        pytest.skip("engine doesn't accept this SQL shape; adjust for your schema")
    assert isinstance(resp, SqlSelect)


def test_live_vector_topk(live_client: OriginChain) -> None:
    try:
        live_client.vector_topk(
            "embeddings",
            query=[0.0, 0.0, 0.0],
            k=1,
            dim=3,
        )
    except OCPaymentRequiredError:
        pytest.skip("vector add-on not entitled on this tenant")
    except OCNotFoundError:
        pytest.skip("vector table 'embeddings' not present")


def test_live_fts_search(live_client: OriginChain) -> None:
    try:
        live_client.fts_search("articles", "body", q="the", mode="boolean")
    except OCPaymentRequiredError:
        pytest.skip("fts add-on not entitled on this tenant")
    except OCNotFoundError:
        pytest.skip("fts index for articles.body not present")


def test_live_graph_neighbors(live_client: OriginChain) -> None:
    try:
        live_client.graph.neighbors("users", rel="follows", pk="u1")
    except OCPaymentRequiredError:
        pytest.skip("graph add-on not entitled on this tenant")
    except OCNotFoundError:
        pytest.skip("graph schema 'users' not present")
