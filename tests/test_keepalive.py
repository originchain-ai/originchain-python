"""Pooled connections outlive httpx's 5 s default idle expiry."""

from __future__ import annotations

from originchain import AsyncOriginChain, OriginChain
from originchain.client import (
    DEFAULT_KEEPALIVE_EXPIRY_S,
    MAX_KEEPALIVE_EXPIRY_S,
    _pool_limits,
)


def test_default_keepalive_is_300s_and_pool_sizes_stay_bounded() -> None:
    limits = _pool_limits(DEFAULT_KEEPALIVE_EXPIRY_S)
    assert limits.keepalive_expiry == 300.0
    # httpx.Limits(keepalive_expiry=...) alone would make these None (unbounded).
    assert limits.max_connections == 100
    assert limits.max_keepalive_connections == 20


def test_keepalive_is_clamped_below_the_network_idle_timeout() -> None:
    assert _pool_limits(10_000).keepalive_expiry == MAX_KEEPALIVE_EXPIRY_S
    assert MAX_KEEPALIVE_EXPIRY_S < 350.0
    assert _pool_limits(-1).keepalive_expiry == 0.0
    assert _pool_limits(60).keepalive_expiry == 60.0


def test_clients_accept_the_parameter() -> None:
    sync = OriginChain(base_url="https://example.invalid", bearer="x", tenant="t", keepalive_expiry=120)
    sync.close()
    OriginChain(base_url="https://example.invalid", bearer="x", tenant="t").close()
    AsyncOriginChain(base_url="https://example.invalid", bearer="x", tenant="t", keepalive_expiry=120)
