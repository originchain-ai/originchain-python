"""Shared test fixtures.

Tests use ``httpx.MockTransport`` to intercept requests at the transport
level — no real socket, no live engine. The ``mock_client`` fixture
returns an ``OriginChain`` whose underlying ``httpx.Client`` routes
through a caller-supplied handler. The handler is the assertion
surface: it inspects the incoming request and returns whatever
``httpx.Response`` the test wants.
"""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from originchain import OriginChain


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> OriginChain:
    """Build a sync client with a MockTransport-backed httpx.Client.

    We can't reuse ``OriginChain.__init__`` directly because it builds
    its own httpx.Client. Instead we construct, then swap. ``max_retries
    =0`` keeps tests deterministic — retries would re-call the handler
    and confuse assertion counts."""
    client = OriginChain(
        base_url="http://test.invalid",
        bearer="test-bearer",
        tenant="01HX1TESTTENANTXXXXXXXXXX1",
        max_retries=0,
    )
    client._client = httpx.Client(
        base_url="http://test.invalid",
        transport=httpx.MockTransport(handler),
        headers={
            "Authorization": "Bearer test-bearer",
            "User-Agent": "originchain-python-test/0.3.0",
        },
    )
    return client


@pytest.fixture
def mock_client() -> Callable[[Callable[[httpx.Request], httpx.Response]], OriginChain]:
    """Returns a builder so each test can supply its own handler."""
    return make_client
