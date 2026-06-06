"""Async variant of :class:`OriginChain`. Mirrors the sync surface.

Both clients are kept side-by-side rather than sharing a common base
because httpx's sync/async clients have subtly different timeout +
context-manager semantics. Duplication is the lesser evil; pyright
catches drift.
"""

# Same rationale as the sync client: httpx.Response.json() returns
# `Any`, which `mypy --strict` flags at every `return await
# resp.json()` even when the method's declared return type matches
# what the engine actually emits. Disable the single check at file
# scope.
# mypy: disable-error-code="no-any-return"

from __future__ import annotations

import asyncio
import json as _json
import os
import warnings
from typing import Any, List, Literal, Mapping, Optional

import httpx

from .client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_S,
    RETRYABLE_STATUSES,
    _HTTP2_AVAILABLE,
    _MUTATING_METHODS,
    _new_idempotency_key,
)
from .errors import (
    OCAuthError,
    OCError,
    OCNotFoundError,
    OCPaymentRequiredError,
    OCRateLimitedError,
    OCReplicationDegraded,
    OCServerError,
    OCValidationError,
)
from .models import (
    DijkstraResult,
    FtsHit,
    GraphBfsHit,
    GraphPath,
    Neighbor,
    SqlResponse,
    SqlSelect,
    VectorHit,
    _decode_sql_response,
)


class _AsyncSchemas:
    def __init__(self, parent: "AsyncOriginChain") -> None:
        self._p = parent

    async def list(self) -> list[str]:
        r = await self._p._request("GET", f"/v1/tenants/{self._p.tenant}/schemas")
        return r.json()

    async def get(self, schema: str) -> str:
        r = await self._p._request("GET", f"/v1/tenants/{self._p.tenant}/schemas/{schema}")
        return r.text

    async def register(self, toml_source: str) -> dict[str, Any]:
        r = await self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/schemas",
            content=toml_source.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
        )
        return r.json()


class _AsyncRows:
    def __init__(self, parent: "AsyncOriginChain") -> None:
        self._p = parent

    async def get(self, schema: str, pk: str) -> dict[str, Any]:
        r = await self._p._request("GET", f"/v1/tenants/{self._p.tenant}/rows/{schema}/{pk}")
        return r.json()

    async def put(
        self,
        schema: str,
        row: Mapping[str, Any],
        *,
        expect_insert: bool = False,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        params = {"expect": "insert"} if expect_insert else None
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
        r = await self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/rows/{schema}",
            json=row,
            params=params,
            headers=headers,
        )
        return r.json()


class _AsyncGraph:
    """Async ``db.graph.*`` namespace. Mirrors the sync ``_Graph``."""

    def __init__(self, parent: "AsyncOriginChain") -> None:
        self._p = parent

    async def neighbors(self, schema: str, *, rel: str, pk: str) -> List[Neighbor]:
        params = {"rel": rel, "pk": pk}
        r = await self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/neighbors",
            params=params,
        )
        return [Neighbor(pk=str(p), depth=1) for p in r.json()]

    async def reverse_neighbors(
        self, schema: str, *, rel: str, pk: str
    ) -> List[Neighbor]:
        params = {"rel": rel, "pk": pk}
        r = await self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/reverse",
            params=params,
        )
        return [Neighbor(pk=str(p), depth=1) for p in r.json()]

    async def bfs(
        self,
        schema: str,
        *,
        rel: str,
        pk: str,
        max_depth: int = 3,
    ) -> List[GraphBfsHit]:
        params = {"rel": rel, "pk": pk, "max_depth": str(max_depth)}
        r = await self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/bfs",
            params=params,
        )
        return [GraphBfsHit._from_payload(h) for h in r.json()]

    async def path(
        self,
        schema: str,
        *,
        rel: str,
        src: str,
        dst: str,
        max_depth: int = 3,
    ) -> GraphPath:
        params = {"rel": rel, "src": src, "dst": dst, "max_depth": str(max_depth)}
        r = await self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/path",
            params=params,
        )
        body = r.json()
        return GraphPath(reachable=bool(body.get("reachable", False)))

    async def dijkstra(
        self,
        schema: str,
        *,
        rel: str,
        src: str,
        dst: str,
        weights: Mapping[str, float],
    ) -> DijkstraResult:
        params = {
            "rel": rel,
            "src": src,
            "dst": dst,
            "weights_json": _json.dumps(dict(weights)),
        }
        r = await self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/dijkstra",
            params=params,
        )
        body = r.json()
        cost = body.get("cost")
        return DijkstraResult(cost=None if cost is None else float(cost))


class AsyncOriginChain:
    """asyncio-native client. ``async with`` for resource cleanup."""

    def __init__(
        self,
        *,
        base_url: str,
        bearer: str,
        tenant: str,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        verify: bool | str = True,
        user_agent: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.bearer = bearer
        self.tenant = tenant
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            verify=verify,
            http2=_HTTP2_AVAILABLE,
            headers={
                "Authorization": f"Bearer {bearer}",
                "User-Agent": user_agent or "originchain-python/0.3.0",
            },
        )
        self.schemas = _AsyncSchemas(self)
        self.rows = _AsyncRows(self)
        self.graph = _AsyncGraph(self)

    @classmethod
    def from_env(cls, **kwargs: Any) -> "AsyncOriginChain":
        try:
            base_url = os.environ["OC_BASE_URL"]
            bearer = os.environ["OC_BEARER"]
            tenant = os.environ["OC_TENANT"]
        except KeyError as e:
            raise OCError(f"missing required env var: {e.args[0]}") from None
        return cls(base_url=base_url, bearer=bearer, tenant=tenant, **kwargs)

    async def health(self) -> dict[str, Any]:
        r = await self._request("GET", "/health")
        return r.json()

    async def ask(
        self, nl: str, *, schemas: Optional[list[str]] = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"nl": nl}
        if schemas is not None:
            body["schemas"] = schemas
        r = await self._request("POST", f"/v1/tenants/{self.tenant}/ask", json=body)
        return r.json()

    async def query(self, plan: dict[str, Any]) -> list[Any]:
        r = await self._request(
            "POST", f"/v1/tenants/{self.tenant}/query", json=plan
        )
        return r.json()

    # ── SQL ───────────────────────────────────────────────────────────

    async def sql(self, query: str) -> SqlResponse:
        """See :meth:`OriginChain.sql` for the contract."""
        r = await self._request(
            "POST",
            f"/v1/tenants/{self.tenant}/sql",
            json={"sql": query},
        )
        return _decode_sql_response(r.json())

    async def sql_one(self, query: str) -> Optional[dict[str, Any]]:
        resp = await self.sql(query)
        if not isinstance(resp, SqlSelect):
            raise OCValidationError(
                f"sql_one expected SELECT, got {type(resp).__name__}"
            )
        if not resp.rows:
            return None
        first = resp.rows[0]
        if not isinstance(first, dict):
            return {"value": first}
        return dict(first)

    # ── Vector ────────────────────────────────────────────────────────

    async def vector_put(
        self,
        table: str,
        *,
        id: str,
        embedding: list[float],
        dim: int,
        metric: str = "cosine",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        body: dict[str, Any] = {
            "id": id,
            "embedding": list(embedding),
            "dim": dim,
            "metric": metric,
        }
        if metadata is not None:
            body["metadata"] = dict(metadata)
        await self._request(
            "POST",
            f"/v1/tenants/{self.tenant}/vector/{table}/put",
            json=body,
        )

    async def vector_topk(
        self,
        table: str,
        *,
        query: list[float],
        k: int = 10,
        dim: int,
        metric: str = "cosine",
        filter: Optional[Mapping[str, Any]] = None,
        mode: Optional[Literal["fast", "high_recall"]] = None,
    ) -> list[VectorHit]:
        body: dict[str, Any] = {
            "query": list(query),
            "k": k,
            "dim": dim,
            "metric": metric,
        }
        if mode is not None:
            body["mode"] = mode
        if filter is not None:
            body["filter"] = dict(filter)
        r = await self._request(
            "POST",
            f"/v1/tenants/{self.tenant}/vector/{table}/topk",
            json=body,
        )
        return [VectorHit._from_payload(h) for h in r.json()]

    # ── Full-text ─────────────────────────────────────────────────────

    async def fts_index(
        self,
        table: str,
        field: str,
        *,
        doc_id: str,
        text: str,
    ) -> None:
        await self._request(
            "POST",
            f"/v1/tenants/{self.tenant}/fts/{table}/{field}",
            json={"doc_id": doc_id, "text": text},
        )

    async def fts_search(
        self,
        table: str,
        field: str,
        *,
        q: str,
        mode: str = "boolean",
        k: int = 10,
    ) -> list[FtsHit]:
        params: dict[str, str] = {"q": q, "mode": mode}
        if mode == "bm25":
            params["k"] = str(k)
        r = await self._request(
            "GET",
            f"/v1/tenants/{self.tenant}/fts/{table}/{field}",
            params=params,
        )
        body = r.json()
        if mode == "bm25":
            return [FtsHit._from_ranked(h) for h in body]
        return [FtsHit._from_doc_id(str(d)) for d in body]

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncOriginChain":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json: Any = None,
        content: Optional[bytes] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> httpx.Response:
        # Auto-Idempotency-Key on mutating calls. See the sync client's
        # `_request` for the rationale; engine cache is LRU-bounded so
        # fresh-per-call is safe, and callers retain override semantics
        # by passing the header explicitly.
        if method.upper() in _MUTATING_METHODS:
            if headers is None:
                headers = {"Idempotency-Key": _new_idempotency_key()}
            elif not any(k.lower() == "idempotency-key" for k in headers):
                headers = {**headers, "Idempotency-Key": _new_idempotency_key()}
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    content=content,
                    headers=headers,
                )
            except httpx.RequestError as e:
                last_exc = e
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                raise OCError(f"transport error: {e}") from e

            if resp.status_code < 400:
                if resp.headers.get("X-OC-Replication", "").lower() == "degraded":
                    warnings.warn(
                        "leader returned 200 but follower(s) didn't ack within"
                        " --sync-timeout-ms; write is durable but RPO=0 not met",
                        OCReplicationDegraded,
                        stacklevel=3,
                    )
                return resp

            if resp.status_code in RETRYABLE_STATUSES and attempt < self.max_retries:
                wait = self._retry_after(resp) or self._backoff(attempt)
                await asyncio.sleep(wait)
                continue
            self._raise_for(resp)
        raise OCError(f"request failed after retries: {last_exc}")

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(4.0, 0.25 * (2 ** attempt))

    @staticmethod
    def _retry_after(resp: httpx.Response) -> Optional[float]:
        ra = resp.headers.get("Retry-After")
        if ra is None:
            return None
        try:
            return float(ra)
        except ValueError:
            return None

    @staticmethod
    def _raise_for(resp: httpx.Response) -> None:
        body: Any
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        msg = body.get("error") if isinstance(body, dict) else str(body)
        status = resp.status_code
        if status in (401, 403):
            raise OCAuthError(msg or "unauthorized", status=status, body=body)
        if status == 402:
            addon_msg = (
                body.get("msg") if isinstance(body, dict) else None
            ) or msg or "payment required (add-on)"
            raise OCPaymentRequiredError(addon_msg, status=status, body=body)
        if status == 404:
            raise OCNotFoundError(msg or "not found", status=status, body=body)
        if status == 400:
            raise OCValidationError(msg or "validation failed", status=status, body=body)
        if status == 429:
            raise OCRateLimitedError(
                msg or "rate limited",
                retry_after=AsyncOriginChain._retry_after(resp) or 1.0,
                status=status,
                body=body,
            )
        if 500 <= status < 600:
            raise OCServerError(msg or f"server error {status}", status=status, body=body)
        raise OCError(msg or f"unexpected status {status}", status=status, body=body)


__all__ = ["AsyncOriginChain"]
