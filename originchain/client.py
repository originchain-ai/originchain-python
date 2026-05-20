"""Synchronous client. The async variant in :mod:`async_client` mirrors
this surface exactly — keep both in sync when adding methods."""

from __future__ import annotations

import json as _json
import os
import time
import uuid
import warnings
from typing import Any, Iterable, List, Literal, Mapping, Optional

import httpx
from importlib.util import find_spec

# Auto-enable HTTP/2 if the optional `h2` package is installed. With ALPN
# now advertised by the engine, a persistent client can multiplex many
# requests over one TCP+TLS connection instead of paying a fresh
# handshake (~300 ms from outside the engine's region) on every call.
# `pip install originchain[http2]` pulls in `h2`; bare `pip install
# originchain` keeps HTTP/1.1 so existing installs don't break.
_HTTP2_AVAILABLE = find_spec("h2") is not None

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


# Default behaviour — override per-call if needed.
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 3
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _new_idempotency_key() -> str:
    """Generate a fresh Idempotency-Key. The engine's server-side cache is
    bounded (LRU 10k entries + 24h TTL), so fresh-per-call is safe and
    makes retries on the same request object idempotent without the caller
    having to think about it."""
    return uuid.uuid4().hex


class _Schemas:
    """``db.schemas.*`` namespace."""

    def __init__(self, parent: "OriginChain") -> None:
        self._p = parent

    def list(self) -> list[str]:
        return self._p._request("GET", f"/v1/tenants/{self._p.tenant}/schemas").json()

    def get(self, schema: str) -> str:
        return self._p._request("GET", f"/v1/tenants/{self._p.tenant}/schemas/{schema}").text

    def register(self, toml_source: str) -> dict[str, Any]:
        return self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/schemas",
            content=toml_source.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
        ).json()


class _Rows:
    """``db.rows.*`` namespace."""

    def __init__(self, parent: "OriginChain") -> None:
        self._p = parent

    def get(self, schema: str, pk: str) -> dict[str, Any]:
        return self._p._request("GET", f"/v1/tenants/{self._p.tenant}/rows/{schema}/{pk}").json()

    def put(
        self,
        schema: str,
        row: Mapping[str, Any],
        *,
        expect_insert: bool = False,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        params = {"expect": "insert"} if expect_insert else None
        # Explicit caller key wins; otherwise `_request` auto-fills.
        headers = (
            {"Idempotency-Key": idempotency_key} if idempotency_key else None
        )
        return self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/rows/{schema}",
            json=row,
            params=params,
            headers=headers,
        ).json()

    def put_batch(
        self,
        schema: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        expect_insert: bool = False,
        idempotency_key: Optional[str] = None,
        chunk: int = 1000,
    ) -> int:
        """Upload `rows` in chunks of `chunk`. Returns total rows accepted.

        Each chunk gets a deterministic per-chunk key derived from a single
        base UUID — so a partial-retry of the same `put_batch` call hits
        the same cache slots and the engine dedupes correctly. The caller
        can pass a stable `idempotency_key` to extend dedup across process
        restarts; otherwise we mint a fresh base for this single call."""
        params = {"expect": "insert"} if expect_insert else None
        base_idem = idempotency_key or _new_idempotency_key()
        total = 0
        chunk_buf: list[Mapping[str, Any]] = []
        for i, r in enumerate(rows):
            chunk_buf.append(r)
            if len(chunk_buf) >= chunk:
                total += self._send_batch(schema, chunk_buf, params, base_idem, i // chunk)
                chunk_buf = []
        if chunk_buf:
            total += self._send_batch(
                schema, chunk_buf, params, base_idem, total // chunk
            )
        return total

    def _send_batch(
        self,
        schema: str,
        rows: list[Mapping[str, Any]],
        params: Optional[dict],
        base_idem: str,
        chunk_no: int,
    ) -> int:
        resp = self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/rows/{schema}/_batch",
            json=list(rows),
            params=params,
            headers={"Idempotency-Key": f"{base_idem}-c{chunk_no}"},
        )
        return resp.json().get("inserted", 0)


class _Graph:
    """``db.graph.*`` namespace. Wraps the five graph endpoints under
    ``/v1/tenants/:t/graph/:schema/{neighbors,reverse,bfs,path,dijkstra}``.
    Each method returns a typed dataclass instead of raw dicts."""

    def __init__(self, parent: "OriginChain") -> None:
        self._p = parent

    def neighbors(self, schema: str, *, rel: str, pk: str) -> List[Neighbor]:
        params = {"rel": rel, "pk": pk}
        pks = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/neighbors",
            params=params,
        ).json()
        return [Neighbor(pk=str(p), depth=1) for p in pks]

    def reverse_neighbors(self, schema: str, *, rel: str, pk: str) -> List[Neighbor]:
        params = {"rel": rel, "pk": pk}
        pks = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/reverse",
            params=params,
        ).json()
        return [Neighbor(pk=str(p), depth=1) for p in pks]

    def bfs(
        self,
        schema: str,
        *,
        rel: str,
        pk: str,
        max_depth: int = 3,
    ) -> List[GraphBfsHit]:
        params = {"rel": rel, "pk": pk, "max_depth": str(max_depth)}
        hits = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/bfs",
            params=params,
        ).json()
        return [GraphBfsHit._from_payload(h) for h in hits]

    def path(
        self,
        schema: str,
        *,
        rel: str,
        src: str,
        dst: str,
        max_depth: int = 3,
    ) -> GraphPath:
        params = {"rel": rel, "src": src, "dst": dst, "max_depth": str(max_depth)}
        body = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/path",
            params=params,
        ).json()
        return GraphPath(reachable=bool(body.get("reachable", False)))

    def dijkstra(
        self,
        schema: str,
        *,
        rel: str,
        src: str,
        dst: str,
        weights: Mapping[str, float],
    ) -> DijkstraResult:
        # Backend reads `q.weights_json` as a query parameter (NOT a body).
        # See preview_endpoints.rs::graph_dijkstra.
        params = {
            "rel": rel,
            "src": src,
            "dst": dst,
            "weights_json": _json.dumps(dict(weights)),
        }
        body = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/dijkstra",
            params=params,
        ).json()
        cost = body.get("cost")
        return DijkstraResult(cost=None if cost is None else float(cost))


class OriginChain:
    """Synchronous OriginChain client.

    Use :meth:`from_env` for the standard env-var bootstrap, or pass
    ``base_url`` / ``bearer`` / ``tenant`` directly for tests."""

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
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            verify=verify,
            http2=_HTTP2_AVAILABLE,
            headers={
                "Authorization": f"Bearer {bearer}",
                "User-Agent": user_agent or "originchain-python/0.3.0",
            },
        )
        self.schemas = _Schemas(self)
        self.rows = _Rows(self)
        self.graph = _Graph(self)

    @classmethod
    def from_env(cls, **kwargs: Any) -> "OriginChain":
        """Build a client from ``OC_BASE_URL``, ``OC_BEARER``, ``OC_TENANT``.

        Raises :class:`OCError` if any are missing — callers should set
        them via their secrets manager, not commit them."""
        try:
            base_url = os.environ["OC_BASE_URL"]
            bearer = os.environ["OC_BEARER"]
            tenant = os.environ["OC_TENANT"]
        except KeyError as e:
            raise OCError(f"missing required env var: {e.args[0]}") from None
        return cls(base_url=base_url, bearer=bearer, tenant=tenant, **kwargs)

    # ── Top-level convenience ─────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health").json()

    def ask(self, nl: str, *, schemas: Optional[list[str]] = None) -> dict[str, Any]:
        """Natural-language query. Returns ``{"rows": [...], "cache": "..."}``."""
        body: dict[str, Any] = {"nl": nl}
        if schemas is not None:
            body["schemas"] = schemas
        return self._request(
            "POST", f"/v1/tenants/{self.tenant}/ask", json=body
        ).json()

    def query(self, plan: dict[str, Any]) -> list[Any]:
        """Structured plan execution. The plan format is in the engine docs."""
        return self._request(
            "POST", f"/v1/tenants/{self.tenant}/query", json=plan
        ).json()

    # ── SQL ───────────────────────────────────────────────────────────

    def sql(self, query: str) -> SqlResponse:
        """Execute a SQL statement against the substrate.

        Returns a tagged-union dataclass discriminated on ``kind``:

        - ``SqlSelect``: server ran the SELECT and returned rows.
        - ``SqlInsert``: server translated the INSERT into the typed
          ``/rows/:schema`` payload — caller re-issues with idempotency.
        - ``SqlDelete``: server translated the DELETE into a typed PK
          for ``/rows/:schema/:pk``.

        See ``backend/crates/oc-http/src/preview_endpoints.rs::sql_exec``
        for the full contract. The split exists because writes from
        ``/sql`` need the idempotency-key plumbing the typed row
        endpoints already have."""
        body = self._request(
            "POST",
            f"/v1/tenants/{self.tenant}/sql",
            json={"sql": query},
        ).json()
        return _decode_sql_response(body)

    def sql_one(self, query: str) -> Optional[dict[str, Any]]:
        """Convenience: run a SELECT and return the first row (or
        ``None`` if no rows). Raises :class:`OCValidationError` if the
        statement isn't a SELECT — there's no "first" of an INSERT or
        DELETE translation."""
        resp = self.sql(query)
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

    def vector_put(
        self,
        table: str,
        *,
        id: str,
        embedding: list[float],
        dim: int,
        metric: str = "cosine",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Upsert a single vector. ``dim`` is required because the
        substrate validates the embedding length matches the table's
        configured dimensionality on every put. ``metric`` is one of
        ``"cosine"`` / ``"dot"`` / ``"l2"`` and is checked against the
        first put's choice for the table — changing it after the index
        is built returns 400."""
        body: dict[str, Any] = {
            "id": id,
            "embedding": list(embedding),
            "dim": dim,
            "metric": metric,
        }
        if metadata is not None:
            body["metadata"] = dict(metadata)
        self._request(
            "POST",
            f"/v1/tenants/{self.tenant}/vector/{table}/put",
            json=body,
        )

    def vector_topk(
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
        """Approximate-NN topk over a vector table.

        ``mode`` selects the recall/latency profile: ``"fast"`` favours
        latency, ``"high_recall"`` favours recall. When omitted the
        server defaults to ``"high_recall"``. ``filter`` is a metadata
        equality filter — non-empty filters force an HNSW + post-filter
        codepath server-side."""
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
        hits = self._request(
            "POST",
            f"/v1/tenants/{self.tenant}/vector/{table}/topk",
            json=body,
        ).json()
        return [VectorHit._from_payload(h) for h in hits]

    # ── Full-text ─────────────────────────────────────────────────────

    def fts_index(
        self,
        table: str,
        field: str,
        *,
        doc_id: str,
        text: str,
    ) -> None:
        """Index ``text`` under ``(table, field, doc_id)``. Per-tenant
        per-field inverted index; subsequent calls with the same
        ``doc_id`` overwrite."""
        self._request(
            "POST",
            f"/v1/tenants/{self.tenant}/fts/{table}/{field}",
            json={"doc_id": doc_id, "text": text},
        )

    def fts_search(
        self,
        table: str,
        field: str,
        *,
        q: str,
        mode: str = "boolean",
        k: int = 10,
    ) -> list[FtsHit]:
        """Full-text search.

        ``mode="boolean"`` (default) AND-matches all tokens and returns
        unranked doc_ids. ``mode="bm25"`` returns the top-``k`` hits
        ranked by BM25. ``mode="phrase"`` requires the tokens in order.

        All three modes return :class:`FtsHit` for a uniform shape;
        boolean / phrase set ``score=0.0`` since they don't rank."""
        params: dict[str, str] = {"q": q, "mode": mode}
        if mode == "bm25":
            params["k"] = str(k)
        body = self._request(
            "GET",
            f"/v1/tenants/{self.tenant}/fts/{table}/{field}",
            params=params,
        ).json()
        if mode == "bm25":
            return [FtsHit._from_ranked(h) for h in body]
        # Boolean / phrase return List[str] of doc_ids.
        return [FtsHit._from_doc_id(str(d)) for d in body]

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OriginChain":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── Internal: retry + error mapping ───────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json: Any = None,
        content: Optional[bytes] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> httpx.Response:
        # Auto-add Idempotency-Key on every mutating call so a network
        # blip / 502 retry deduplicates. The engine's idempotency cache
        # is LRU-bounded (10k entries) so a fresh UUID per call is safe;
        # callers who want cross-process retry semantics can still pass
        # a stable key in `headers`.
        if method.upper() in _MUTATING_METHODS:
            if headers is None:
                headers = {"Idempotency-Key": _new_idempotency_key()}
            elif not any(k.lower() == "idempotency-key" for k in headers):
                headers = {**headers, "Idempotency-Key": _new_idempotency_key()}
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.request(
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
                    time.sleep(self._backoff(attempt))
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
                time.sleep(wait)
                continue

            self._raise_for(resp)
        # _raise_for never returns; loop exhaustion only via transport.
        raise OCError(f"request failed after retries: {last_exc}")

    @staticmethod
    def _backoff(attempt: int) -> float:
        # Exponential with jitter cap. 0.25, 0.5, 1.0, 2.0, capped at 4 s.
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
            # 402 body shape is the canonical addon-required envelope —
            # see oc-http::entitlements::build_402_body.
            addon_msg = (
                body.get("msg")
                if isinstance(body, dict)
                else None
            ) or msg or "payment required (add-on)"
            raise OCPaymentRequiredError(addon_msg, status=status, body=body)
        if status == 404:
            raise OCNotFoundError(msg or "not found", status=status, body=body)
        if status == 400:
            raise OCValidationError(msg or "validation failed", status=status, body=body)
        if status == 429:
            raise OCRateLimitedError(
                msg or "rate limited",
                retry_after=OriginChain._retry_after(resp) or 1.0,
                status=status,
                body=body,
            )
        if 500 <= status < 600:
            raise OCServerError(msg or f"server error {status}", status=status, body=body)
        raise OCError(msg or f"unexpected status {status}", status=status, body=body)


__all__ = ["OriginChain"]
