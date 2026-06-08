"""Tests for ``client.admin.*`` — the per-tenant replication-mode
config admin surface (0.5 addition).

Wire shape from ``oc-http/src/admin.rs::TenantConfigPutBody /
TenantConfigSnapshot``:
- PUT body: ``{"replication_mode": "active_passive" | "raft_quorum"}``
- Response: ``{"replication_mode": "...", "installed": true | null}``
  (``installed`` is `true` on the put response, omitted on the read).
"""

from __future__ import annotations

import json

import httpx
import pytest

from originchain import (
    OriginChainBadRequest,
    OriginChainServerError,
    TenantConfigSnapshot,
)


TENANT = "01HX1TESTTENANTXXXXXXXXXX1"


def test_admin_install_tenant_config_active_passive(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path == f"/v1/admin/tenants/{TENANT}/config"
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={"replication_mode": "active_passive", "installed": True},
        )

    client = mock_client(handler)
    out = client.admin.install_tenant_config(TENANT)
    assert isinstance(out, TenantConfigSnapshot)
    assert out.replication_mode == "active_passive"
    assert out.installed is True
    assert seen["body"] == {"replication_mode": "active_passive"}


def test_admin_install_tenant_config_raft_quorum(mock_client) -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={"replication_mode": "raft_quorum", "installed": True},
        )

    client = mock_client(handler)
    out = client.admin.install_tenant_config(
        TENANT, replication_mode="raft_quorum"
    )
    assert out.replication_mode == "raft_quorum"
    assert out.installed is True
    assert seen["body"]["replication_mode"] == "raft_quorum"


def test_admin_install_tenant_config_rejects_unimplemented(mock_client) -> None:
    # Server rejects with 400 when the requested mode's
    # `is_implemented` returns false — the SDK surfaces it as
    # OriginChainBadRequest (alias of OCValidationError).
    client = mock_client(
        lambda req: httpx.Response(
            400,
            json={
                "error": (
                    "tenant config replication_mode=multi_writer not "
                    "available yet (eta 2026-09)"
                )
            },
        )
    )
    with pytest.raises(OriginChainBadRequest):
        # The literal type bound on the kwarg covers only the two
        # shipped variants; the test passes the unsupported variant as
        # a raw string to mimic an operator using an old SDK against a
        # server that has tightened the validation list.
        client.admin.install_tenant_config(
            TENANT, replication_mode="multi_writer"  # type: ignore[arg-type]
        )


def test_admin_get_tenant_config_installed(mock_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path == f"/v1/admin/tenants/{TENANT}/config"
        # `installed` is None on the read response (the server omits
        # the field via `skip_serializing_if = Option::is_none`); the
        # SDK preserves that None so the read-vs-install discriminator
        # stays meaningful.
        return httpx.Response(
            200, json={"replication_mode": "raft_quorum"}
        )

    client = mock_client(handler)
    out = client.admin.get_tenant_config(TENANT)
    assert out.replication_mode == "raft_quorum"
    assert out.installed is None


def test_admin_get_tenant_config_implicit_default(mock_client) -> None:
    # A tenant whose config has never been installed reads back as
    # active_passive — the server emits the implicit default rather
    # than 404'ing, so callers always see a truthful non-404 shape.
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"replication_mode": "active_passive"}
        )

    client = mock_client(handler)
    out = client.admin.get_tenant_config(TENANT)
    assert out.replication_mode == "active_passive"
    assert out.installed is None


def test_admin_get_tenant_config_error_500(mock_client) -> None:
    client = mock_client(
        lambda req: httpx.Response(500, json={"error": "registry corrupt"})
    )
    with pytest.raises(OriginChainServerError):
        client.admin.get_tenant_config(TENANT)
