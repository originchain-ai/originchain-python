"""Tests for ``client.usage`` - the engine ``GET /v1/tenants/:t/usage``
surface.

The response ``tier`` carries the neutral configuration slug
(``entry`` / ``standard`` / ``advanced`` / ``custom``) - never the
internal weather codename - and a richer ``configuration`` object. See
``oc-http/src/handlers/usage.rs``.
"""

from __future__ import annotations

import json

import httpx

from originchain import TenantConfiguration, TenantUsage


def test_usage_decodes_neutral_configuration(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path == "/v1/tenants/01HX1TESTTENANTXXXXXXXXXX1/usage"
        return httpx.Response(
            200,
            json={
                "tenant": "01HX1TESTTENANTXXXXXXXXXX1",
                "tier": "standard",
                "configuration": {
                    "slug": "standard",
                    "label": "4 vCPU / 16 GB, HA",
                    "vcpu": 4,
                    "ram_gb": 16,
                    "storage_gb": 100,
                    "ha": True,
                    "monthly_price": 699,
                },
                "used": {"store_keys": 42},
                "schemas": [],
            },
        )

    client = mock_client(handler)
    u = client.usage()
    assert isinstance(u, TenantUsage)
    # Neutral slug, never the weather codename.
    assert u.tier == "standard"
    assert isinstance(u.configuration, TenantConfiguration)
    assert u.configuration.slug == "standard"
    assert u.configuration.vcpu == 4
    assert u.configuration.ha is True
    assert u.configuration.monthly_price == 699
    assert "storm" not in json.dumps(
        {"tier": u.tier, "label": u.configuration.label}
    ).lower()


def test_usage_legacy_mode_has_no_configuration(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        # Legacy per-addon mode: engine omits tier + configuration.
        return httpx.Response(
            200,
            json={
                "tenant": "01HX1TESTTENANTXXXXXXXXXX1",
                "used": {"store_keys": 7},
                "schemas": [],
            },
        )

    client = mock_client(handler)
    u = client.usage()
    assert u.tier is None
    assert u.configuration is None
    assert u.used["store_keys"] == 7
