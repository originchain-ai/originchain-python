"""Typed namespaces for the SQL / vector / FTS / graph surfaces.

These hang off the sync :class:`OriginChain` client as
``client.sql`` / ``client.vector`` / ``client.fts`` / ``client.graph``.
They are first-class typed wrappers around the same ``/v1`` HTTP routes
that the legacy methods on the client (``client.sql(...)``,
``client.vector_put(...)``, etc.) call. Both surfaces share one HTTP
client and one auth header — neither is a deprecation path; the
typed namespaces just match the per-family layout the spec calls for.

The wire shapes are documented inline against the Rust handlers in
``backend/crates/oc-http/src/preview_endpoints.rs`` so future schema
drifts get caught by tests that mock the documented JSON.
"""

# Same Any-return relaxation as in client.py — httpx.Response.json()
# returns Any and mypy --strict can't see through the third-party type
# to confirm our dataclass decoders return what they say.
# mypy: disable-error-code="no-any-return"

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING, Any, List, Literal, Mapping, Optional

from .models import (
    CentroidsPreview,
    FacetBucket,
    FtsHitWithHighlights,
    FtsResult,
    GraphEmbeddingHit,
    GraphSageResult,
    InstallCentroidsResult,
    IvfRebalanceStatus,
    MaterializedViewInstallResult,
    MaterializedViewRefreshResult,
    MaterializedViewRows,
    Path,
    SqlExecResult,
    SqlResult,
    TenantConfigSnapshot,
    TrainAndInstallCentroidsResult,
    VectorDeleteBulkResult,
    VectorDeleteResult,
    VectorHitV2,
    _decode_sql_response,
)

if TYPE_CHECKING:
    from .client import OriginChain


# ─────────────────────────── SQL namespace ───────────────────────────


class _SqlNamespace:
    """``client.sql`` — typed SQL methods.

    Also callable for backward compatibility: ``client.sql("SELECT ...")``
    still works exactly as it did before this namespace landed (returns
    the tagged-union :class:`SqlResponse`). Callers using the new
    surface should prefer :meth:`query` / :meth:`execute`."""

    def __init__(self, parent: "OriginChain") -> None:
        self._p = parent

    # Callable form preserves the pre-namespace `client.sql("...")` API.
    def __call__(self, query: str) -> Any:
        return self._p._sql_callable_impl(query)

    def query(
        self,
        query: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> SqlResult:
        """Run a SELECT against the substrate.

        ``params`` is forwarded as the ``params`` field on the request
        body so server-side parameterised SQL works the same way it
        does on the typed row routes. Returns a :class:`SqlResult`
        with both the row list and (when the server emits it) the
        column ordering."""
        body: dict[str, Any] = {"sql": query}
        if params is not None:
            body["params"] = dict(params)
        payload = self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/sql",
            json=body,
        ).json()
        # The preview endpoint returns the kind-tagged union; query()
        # expects SELECT shape and surfaces a clear error otherwise so
        # callers don't silently get back an "insert" envelope.
        if payload.get("kind") not in (None, "select"):
            from .errors import OCValidationError

            raise OCValidationError(
                f"sql.query expected SELECT, got kind={payload.get('kind')!r}",
                status=200,
                body=payload,
            )
        rows_raw = payload.get("rows", [])
        rows: list[dict[str, Any]] = [
            dict(r) if isinstance(r, dict) else {"value": r} for r in rows_raw
        ]
        columns_raw = payload.get("columns")
        columns: Optional[list[str]] = (
            [str(c) for c in columns_raw] if isinstance(columns_raw, list) else None
        )
        # Fall back to first-row keys for a stable column order when
        # the server didn't surface one explicitly.
        if columns is None and rows:
            columns = list(rows[0].keys())
        return SqlResult(rows=rows, columns=columns)

    def install_materialized_view(
        self,
        name: str,
        query: str,
        refresh_mode: Literal["manual", "on_write"] = "manual",
        source_schema: Optional[str] = None,
    ) -> MaterializedViewInstallResult:
        """Install a materialized view: translate ``query``, run the
        initial materialization, and stamp the snapshot under
        ``mv_snapshot_key`` in one WAL frame. ``refresh_mode="manual"``
        is the only mode shipped today (``"on_write"`` is parsed but
        falls back to manual; the on-write incremental path is a v2
        punt). ``source_schema`` is an optional hint — when omitted, the
        handler derives it from the plan's first scan target."""
        body: dict[str, Any] = {
            "name": name,
            "query": query,
            "refresh_mode": refresh_mode,
        }
        if source_schema is not None:
            body["source_schema"] = source_schema
        payload = self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/sql/materialized-views",
            json=body,
        ).json()
        return MaterializedViewInstallResult._from_payload(payload)

    def refresh_materialized_view(
        self, name: str
    ) -> MaterializedViewRefreshResult:
        """On-demand refresh of an installed materialized view. Loads
        the persisted definition, re-translates + re-executes the
        query, atomically overwrites the snapshot. 404 when ``name``
        isn't installed."""
        payload = self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/sql/materialized-views/{name}/refresh",
        ).json()
        return MaterializedViewRefreshResult._from_payload(payload)

    def read_materialized_view(self, name: str) -> MaterializedViewRows:
        """Read the current snapshot of an installed materialized
        view. ``rows`` is the rmp-decoded row bundle from the last
        refresh; the shape is whatever the original SELECT projected.
        404 when ``name`` isn't installed."""
        payload = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/sql/materialized-views/{name}",
        ).json()
        return MaterializedViewRows._from_payload(payload)

    def execute(self, stmt: str) -> SqlExecResult:
        """Run a non-SELECT (INSERT / UPDATE / DELETE).

        The preview ``/sql`` endpoint translates writes into typed row
        payloads rather than executing them inline (see
        ``oc-http/src/preview_endpoints.rs::sql_exec``). This method
        round-trips the translation as a :class:`SqlExecResult` with
        the translated ``kind`` and (when known) the affected schema.
        Customers who want auto-execution of the translation should
        re-issue against ``client.rows.*`` with an idempotency key."""
        payload = self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/sql",
            json={"sql": stmt},
        ).json()
        decoded = _decode_sql_response(payload)
        # The tagged-union has `.kind` on every variant; `.schema` only
        # on SqlInsert / SqlDelete (not SqlSelect). Read defensively
        # via getattr so mypy doesn't trip on the union access.
        kind_val = getattr(decoded, "kind", "unknown")
        kind: str = str(kind_val) if kind_val is not None else "unknown"
        schema_val = getattr(decoded, "schema", None)
        schema: Optional[str] = str(schema_val) if schema_val is not None else None
        # Translated INSERTs / DELETEs report 1 row each (preview
        # endpoint doesn't execute multi-row writes inline). UPDATEs
        # are not yet supported by the preview translator — they come
        # back as 400, which is surfaced as OCValidationError before
        # we get here.
        rows_affected = 1 if kind in ("insert", "delete") else 0
        return SqlExecResult(kind=kind, rows_affected=rows_affected, schema=schema)


# ─────────────────────────── Vector namespace ───────────────────────────


class _VectorNamespace:
    """``client.vector`` — typed vector methods. ``put`` / ``topk`` /
    ``delete`` / ``install_centroids``. The default ``dim`` for
    :meth:`put` and :meth:`topk` is the length of the embedding /
    query vector — the substrate validates this against the table's
    configured dimensionality, so misshaped vectors 400 on the server
    side rather than silently corrupting the index."""

    def __init__(self, parent: "OriginChain") -> None:
        self._p = parent

    def put(
        self,
        table: str,
        vec_id: str,
        embedding: List[float],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        body: dict[str, Any] = {
            "id": vec_id,
            "embedding": list(embedding),
            "dim": len(embedding),
        }
        if metadata is not None:
            body["metadata"] = dict(metadata)
        self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/vector/{table}/put",
            json=body,
        )

    def topk(
        self,
        table: str,
        query: List[float],
        k: int = 10,
        metric: Literal["cosine", "dot", "l2", "manhattan"] = "cosine",
        filter: Optional[Mapping[str, Any]] = None,
        nprobe: Optional[int] = None,
    ) -> list[VectorHitV2]:
        body: dict[str, Any] = {
            "query": list(query),
            "k": k,
            "dim": len(query),
            "metric": metric,
        }
        if filter is not None:
            body["filter"] = dict(filter)
        # `nprobe` is the IVF tuning knob — the server reads it under
        # `ivf_nprobe` to disambiguate from the legacy HNSW `ef_search`
        # hint. Absent → server default for the table's index kind.
        if nprobe is not None:
            body["ivf_nprobe"] = nprobe
        hits = self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/vector/{table}/topk",
            json=body,
        ).json()
        return [VectorHitV2._from_payload(h) for h in hits]

    def delete(
        self,
        table: str,
        vec_id: str,
        index: Optional[Literal["hnsw", "ivf", "ivf_pq"]] = None,
        repair: bool = False,
    ) -> VectorDeleteResult:
        """Remove a single vector from ``table``. Idempotent — deleting
        a non-existent id returns ``{"deleted": false}`` (200), not 404,
        so callers don't have to special-case the missing-row branch.

        ``index`` picks the engine dispatch (``"hnsw"`` default,
        ``"ivf"``, or ``"ivf_pq"``). ``repair=True`` opts the HNSW arm
        into the graph-repair path (re-links neighbours across the hole
        instead of tombstoning only) — ignored on the IVF arms which
        shrink cleanly without topology repair. Pre-2026-06-08 the
        handler didn't exist; 0.4 was wire-ready but errored at runtime
        against the engine."""
        params: dict[str, str] = {}
        if index is not None:
            params["index"] = index
        if repair:
            params["repair"] = "true"
        payload = self._p._request(
            "DELETE",
            f"/v1/tenants/{self._p.tenant}/vector/{table}/{vec_id}",
            params=params or None,
        ).json()
        return VectorDeleteResult._from_payload(payload)

    def delete_bulk(
        self,
        table: str,
        ids: List[str],
        index: Optional[Literal["hnsw", "ivf", "ivf_pq"]] = None,
        repair: bool = False,
    ) -> VectorDeleteBulkResult:
        """Remove up to ``oc_vector::MAX_BULK_DELETE`` (10 000) vectors
        in a single WAL frame. Duplicate ids in one call are deduped
        server-side and count once toward ``deleted_count``. ``repair``
        only applies to the HNSW arm — see :meth:`delete`. Returns
        ``{deleted_count, missing_count}`` so the caller can detect
        partial overlap with the live set."""
        body: dict[str, Any] = {"ids": list(ids)}
        if index is not None:
            body["index"] = index
        if repair:
            body["repair"] = True
        payload = self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/vector/{table}/delete-bulk",
            json=body,
        ).json()
        return VectorDeleteBulkResult._from_payload(payload)

    def install_centroids(
        self,
        table: str,
        centroids: List[List[float]],
    ) -> InstallCentroidsResult:
        """Pre-install IVF centroids for ``table``. The server validates
        every centroid is the same dim and atomically swaps the
        partitioning. The number of centroids becomes the number of
        partitions; ``dim`` is taken from the first centroid.

        0.5: URL renamed to ``install-centroids`` (hyphen) to match the
        engine's admin-route convention. The 0.4 path with an underscore
        no longer exists on the server — callers on 0.4 hit 404."""
        body: dict[str, Any] = {
            "centroids": [list(c) for c in centroids],
        }
        payload = self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/vector/{table}/install-centroids",
            json=body,
        ).json()
        return InstallCentroidsResult._from_payload(payload)

    def train_and_install_centroids(
        self,
        table: str,
        partitions: int,
        init: Optional[Literal["kmeans_plus_plus", "random_sample"]] = None,
        max_iterations: Optional[int] = None,
        batch_size: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> TrainAndInstallCentroidsResult:
        """Read up to ``MAX_TRAINING_SAMPLE_VECTORS`` existing vectors,
        run mini-batch k-means in-process on the engine, then install
        the resulting centroid matrix — the common path for an operator
        bootstrapping an IVF table after a `put_vec` row of writes.

        Refuses with 400 when ``count < partitions × 4`` (k-means
        under-population guard). Synchronous on the request thread —
        large corpora can tie the request up for several seconds; the
        async-job variant is a v2 punt."""
        body: dict[str, Any] = {"partitions": int(partitions)}
        if init is not None:
            body["init"] = init
        if max_iterations is not None:
            body["max_iterations"] = int(max_iterations)
        if batch_size is not None:
            body["batch_size"] = int(batch_size)
        if convergence_threshold is not None:
            body["convergence_threshold"] = float(convergence_threshold)
        if seed is not None:
            body["seed"] = int(seed)
        payload = self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/vector/{table}"
            f"/train-and-install-centroids",
            json=body,
        ).json()
        return TrainAndInstallCentroidsResult._from_payload(payload)

    def centroids(self, table: str) -> CentroidsPreview:
        """Read-only "is anything installed?" inspector. Returns 200
        with ``installed=False`` when nothing has been installed yet
        (not 404 — the route is meaningful for any registered table).
        ``centroids_preview`` is a truncated peek (first 4 centroids,
        first 8 dims each) so the response stays small at any
        ``partitions × dim``."""
        payload = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/vector/{table}/centroids",
        ).json()
        return CentroidsPreview._from_payload(payload)

    def rebalance_status(self, table: str) -> IvfRebalanceStatus:
        """Per-cell live-count + skew report for an IVF table. ``action``
        is one of ``"none" | "recommended" | "required"`` — a hint, not
        a trigger; nothing is auto-rebalanced. The operator decides
        whether to re-train via :meth:`train_and_install_centroids`.
        503 when no centroids have been installed (the table isn't an
        IVF table or hasn't been bootstrapped yet)."""
        payload = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/vector/{table}/ivf-rebalance-status",
        ).json()
        return IvfRebalanceStatus._from_payload(payload)


# ─────────────────────────── FTS namespace ───────────────────────────


class _FtsNamespace:
    """``client.fts`` — typed full-text methods. ``index`` / ``search``
    / ``install_synonyms`` / ``install_stopwords``. The search shape is
    the enriched envelope (hits + optional facets) — both ranked and
    boolean / phrase modes funnel through the same :class:`FtsResult`."""

    def __init__(self, parent: "OriginChain") -> None:
        self._p = parent

    def index(
        self,
        table: str,
        field: str,
        doc_id: str,
        text: str,
    ) -> None:
        self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/fts/{table}/{field}",
            json={"doc_id": doc_id, "text": text},
        )

    def search(
        self,
        table: str,
        field: str,
        query: str,
        mode: Literal["boolean", "bm25", "phrase"] = "bm25",
        fuzzy: Optional[int] = None,
        highlight: bool = False,
        facets: Optional[List[str]] = None,
        k: int = 10,
    ) -> FtsResult:
        """Run a full-text search and decode into a uniform shape.

        - ``mode="boolean"`` (AND tokens) / ``mode="phrase"`` (ordered
          tokens) return doc_ids only; ``score`` defaults to 0.0.
        - ``mode="bm25"`` returns ranked hits. When ``highlight=True``
          or ``facets=[...]`` is supplied, the server emits the
          enriched envelope with per-hit highlight snippets and / or
          per-field facet buckets.
        """
        params: dict[str, str] = {"q": query, "mode": mode}
        if mode == "bm25":
            params["k"] = str(k)
        if fuzzy is not None and fuzzy > 0:
            params["fuzzy"] = str(fuzzy)
        if highlight:
            params["highlight"] = "true"
        if facets:
            params["facets"] = ",".join(facets)
        payload = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/fts/{table}/{field}",
            params=params,
        ).json()
        # Three wire shapes:
        #  - List[str]                              → boolean / phrase
        #  - List[{doc_id, score}]                  → plain bm25
        #  - {hits: [{doc_id, score, highlights?}],
        #     facets?: {field: [{value, count}]}}   → enriched bm25
        if isinstance(payload, list):
            if not payload:
                return FtsResult(hits=[], facets=None)
            first = payload[0]
            if isinstance(first, dict) and "score" in first:
                hits = [FtsHitWithHighlights._from_ranked(h) for h in payload]
            else:
                hits = [FtsHitWithHighlights._from_doc_id(str(d)) for d in payload]
            return FtsResult(hits=hits, facets=None)
        # Enriched envelope.
        hits_raw = payload.get("hits", [])
        hits = [FtsHitWithHighlights._from_enriched(h) for h in hits_raw]
        facets_raw = payload.get("facets") or {}
        facets_out: Optional[dict[str, list[FacetBucket]]] = None
        if facets_raw:
            facets_out = {
                str(field_name): [FacetBucket._from_payload(b) for b in buckets]
                for field_name, buckets in facets_raw.items()
            }
        return FtsResult(hits=hits, facets=facets_out)

    def install_synonyms(
        self,
        table: str,
        field: str,
        synonyms: Mapping[str, List[str]],
    ) -> None:
        self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/fts/{table}/{field}/synonyms",
            json={"synonyms": {k: list(v) for k, v in synonyms.items()}},
        )

    def install_stopwords(
        self,
        table: str,
        field: str,
        stopwords: List[str],
    ) -> None:
        self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/fts/{table}/{field}/stopwords",
            json={"stopwords": list(stopwords)},
        )


# ─────────────────────────── Graph namespace extension ──────────────────
# The legacy `client.graph` namespace (`_Graph` in client.py) carries
# `neighbors` / `reverse_neighbors` / `bfs` / `path` / `dijkstra` with
# kwarg-style signatures. The spec adds nine more methods and re-shapes
# the existing ones to positional-arg style (`neighbors(schema, src_pk,
# rel)` rather than `neighbors(schema, rel=..., pk=...)`).
#
# `_GraphNamespaceExtended` is mixed into `_Graph` so call-sites that
# use either signature work. The positional methods route through the
# same HTTP request helpers and decode into the typed dataclasses below.


class _GraphNamespaceExtended:
    """Spec-shape graph methods. Mixed into the existing ``_Graph``
    namespace so ``client.graph`` carries both APIs in one place."""

    # ``self._p`` is the parent ``OriginChain`` instance (set by
    # ``_Graph.__init__`` in client.py). This class is a behaviour
    # mixin; it does NOT define its own __init__.

    _p: Any  # set by _Graph.__init__

    # ── positional-arg variants of the legacy endpoints ──────────────

    def neighbors_of(self, schema: str, src_pk: str, rel: str) -> list[str]:
        """Positional-arg neighbours lookup. Returns decoded PK strings
        directly (the legacy ``neighbors(...)`` returns ``Neighbor``
        dataclasses, which the spec doesn't require here)."""
        params = {"rel": rel, "pk": src_pk}
        pks = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/neighbors",
            params=params,
        ).json()
        return [str(p) for p in pks]

    def bfs_of(
        self,
        schema: str,
        src_pk: str,
        rel: str,
        max_depth: int = 5,
    ) -> list[str]:
        """Positional-arg BFS. Returns the visited PK list (no depth
        annotation — the legacy ``bfs(...)`` carries depths)."""
        params = {"rel": rel, "pk": src_pk, "max_depth": str(max_depth)}
        hits = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/bfs",
            params=params,
        ).json()
        return [str(h.get("pk", "")) for h in hits]

    def shortest_path(
        self,
        schema: str,
        src: str,
        target: str,
        rel: str,
    ) -> Optional[list[str]]:
        """Run unweighted shortest path. Returns the node list (source
        first) or ``None`` when ``target`` is unreachable from ``src``.

        Implemented in terms of ``k_shortest(... k=1)`` so the SDK
        doesn't duplicate the BFS plumbing — the server's k-shortest
        handler is the one source of truth for "shortest by hops"."""
        paths = self.k_shortest(schema, src, target, rel, k=1)
        if not paths:
            return None
        return paths[0].nodes

    def k_shortest(
        self,
        schema: str,
        src: str,
        target: str,
        rel: str,
        k: int,
        weight_col: Optional[str] = None,
    ) -> list[Path]:
        """Yen's K-shortest loop-free paths. Default weight = 1 per
        edge (BFS-equivalent ranking); pass ``weight_col`` to read
        the per-edge weight from the destination row's named column."""
        params: dict[str, str] = {
            "rel": rel,
            "source": src,
            "target": target,
            "k": str(k),
        }
        if weight_col is not None:
            params["weight_col"] = weight_col
        body = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/k-shortest",
            params=params,
        ).json()
        paths_raw = body.get("paths", [])
        return [Path._from_payload(p) for p in paths_raw]

    def random_walk(
        self,
        schema: str,
        start: str,
        rel: str,
        steps: int,
        seed: int,
        p: float = 1.0,
        q: float = 1.0,
    ) -> list[str]:
        """Seeded random walk. ``p=q=1.0`` is the unbiased baseline;
        either knob != 1.0 routes through the Node2Vec-biased variant."""
        params: dict[str, str] = {
            "rel": rel,
            "start": start,
            "steps": str(steps),
            "seed": str(seed),
        }
        # Only send p/q when they're not the unbiased identity — keeps
        # the wire request minimal and matches the server's "either
        # set → biased walk" behaviour.
        if p != 1.0:
            params["p"] = str(p)
        if q != 1.0:
            params["q"] = str(q)
        body = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/random-walk",
            params=params,
        ).json()
        walk = body.get("walk", [])
        return [str(n) for n in walk]

    def louvain(
        self,
        schema: str,
        rel: str,
        tolerance: float = 1e-4,
        max_levels: int = 10,
    ) -> dict[str, int]:
        """Louvain community detection. Returns ``{pk -> community_id}``.
        Community ids are dense ``[0, k)`` integers in lex-first-PK
        appearance order."""
        params = {
            "rel": rel,
            "tolerance": str(tolerance),
            "max_levels": str(max_levels),
        }
        body = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/louvain",
            params=params,
        ).json()
        communities = body.get("communities", [])
        return {str(c["pk"]): int(c["community"]) for c in communities}

    def pagerank(
        self,
        schema: str,
        rel: str,
        damping: float = 0.85,
        tolerance: float = 1e-6,
        nodes: Optional[List[str]] = None,
        max_iter: int = 100,
    ) -> dict[str, float]:
        """PageRank over ``nodes`` along forward edges of ``rel``.

        The server requires a node universe (so the caller picks the
        subgraph). When ``nodes`` is omitted the SDK reflects the
        empty list back to the server, which 400s — callers either
        pass the full node list or hit the typed error. ``tolerance``
        maps to the server's ``tol`` query param."""
        params: dict[str, str] = {
            "rel": rel,
            "damping": str(damping),
            "tol": str(tolerance),
            "max_iter": str(max_iter),
        }
        if nodes is not None:
            params["nodes"] = ",".join(nodes)
        hits = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/pagerank",
            params=params,
        ).json()
        return {str(h["pk"]): float(h["score"]) for h in hits}

    def label_propagation(
        self,
        schema: str,
        rel: str,
        seed: int,
        max_iter: int = 50,
    ) -> dict[str, int]:
        """Label-propagation community detection. Returns
        ``{pk -> label}``. ``seed`` is forwarded to the server's RNG
        so ``(seed, graph)`` produces reproducible labels — the server
        defaults to the current unix timestamp when no seed is given,
        which is non-deterministic, so the SDK requires the kwarg."""
        params = {
            "rel": rel,
            "max_iter": str(max_iter),
            "seed": str(seed),
        }
        rows = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/label_propagation",
            params=params,
        ).json()
        # Wire shape: List[{pk: [...], label: u64}]. The "pk" cell is
        # itself a list (Plan-variant row shape), so we stringify it
        # via JSON to get a stable dict key.
        out: dict[str, int] = {}
        for row in rows:
            pk_cell = row.get("pk")
            key = pk_cell if isinstance(pk_cell, str) else _json.dumps(pk_cell)
            out[key] = int(row.get("label", 0))
        return out

    def node2vec_topk(
        self,
        schema: str,
        rel: str,
        query_pk: str,
        k: int,
        metric: Literal["cosine", "dot", "l2", "manhattan"] = "cosine",
    ) -> list[GraphEmbeddingHit]:
        """Top-``k`` nodes most similar to ``query_pk`` under ``metric``,
        against the persisted Node2Vec embeddings for ``(schema, rel)``.

        Persistence is the prerequisite: call ``POST .../node2vec``
        with ``persist=true`` first. Until then the route 503s with a
        ``"POST with persist=true first"`` hint; the SDK surfaces that
        as :class:`OCServerError` so callers can detect the missing-
        prerequisite case."""
        params: dict[str, str] = {
            "query": query_pk,
            "k": str(k),
            "metric": metric,
        }
        hits = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/node2vec/{rel}/topk",
            params=params,
        ).json()
        return [GraphEmbeddingHit._from_payload(h) for h in hits]

    def graphsage(
        self,
        schema: str,
        feature_col: str,
        rel: str,
        config: Optional[Mapping[str, Any]] = None,
        persist: bool = False,
    ) -> GraphSageResult:
        """Train GraphSAGE attribute-aware node embeddings on the row
        universe under ``schema`` and return the per-node vectors.

        ``config`` is forwarded verbatim as the request body's optional
        knobs (``embedding_dim`` / ``num_layers`` / ``aggregator`` /
        ``epochs`` / ``seed`` / ...); a missing key picks the engine's
        library default. ``persist=True`` writes the embeddings under
        the sibling ``vec_graphsage`` key shape so :meth:`graphsage_topk`
        becomes callable. Server refuses with 400 when the projected
        response (``vocab_size × embedding_dim``) exceeds the 1M-float
        cap — use the CLI export for larger embedding sets."""
        body: dict[str, Any] = {
            "rel": rel,
            "feature_col": feature_col,
            "persist": bool(persist),
        }
        if config is not None:
            for key, value in config.items():
                if key in ("rel", "feature_col", "persist"):
                    # Spec-controlled fields — caller picks via the
                    # explicit kwargs above, not via the config blob.
                    continue
                body[key] = value
        payload = self._p._request(
            "POST",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/graphsage",
            json=body,
        ).json()
        return GraphSageResult._from_payload(payload)

    def graphsage_topk(
        self,
        schema: str,
        rel: str,
        query_pk: str,
        k: int,
        metric: Literal["cosine", "dot", "l2", "manhattan"] = "cosine",
    ) -> list[GraphEmbeddingHit]:
        """Top-``k`` nodes most similar to ``query_pk`` under ``metric``,
        against the persisted GraphSAGE embeddings. Same shape as
        :meth:`node2vec_topk` so SDKs only learn the wire grammar once.

        503 (surfaced as :class:`OCServerError`) when no embeddings have
        been persisted under ``(schema, rel)`` yet — call
        :meth:`graphsage` with ``persist=True`` first."""
        params: dict[str, str] = {
            "query": query_pk,
            "k": str(k),
            "metric": metric,
        }
        hits = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/graphsage/{rel}/topk",
            params=params,
        ).json()
        return [GraphEmbeddingHit._from_payload(h) for h in hits]

    def betweenness(
        self,
        schema: str,
        rel: str,
        max_nodes: Optional[int] = None,
    ) -> dict[str, float]:
        """Brandes' betweenness centrality. Returns ``{pk -> score}``,
        ordered by descending score. ``max_nodes`` clamps the node
        universe; the server hard-caps at
        ``oc_graph::BETWEENNESS_MAX_NODES`` (100k) — Brandes' is
        O(V·E) and bigger graphs won't land in an HTTP budget."""
        params: dict[str, str] = {"rel": rel}
        if max_nodes is not None:
            params["max_nodes"] = str(max_nodes)
        rows = self._p._request(
            "GET",
            f"/v1/tenants/{self._p.tenant}/graph/{schema}/betweenness",
            params=params,
        ).json()
        out: dict[str, float] = {}
        for row in rows:
            pk_cell = row.get("pk")
            key = pk_cell if isinstance(pk_cell, str) else _json.dumps(pk_cell)
            out[key] = float(row.get("betweenness", 0.0))
        return out


# ─────────────────────────── Admin namespace ───────────────────────────
#
# Hangs off ``client.admin`` and routes against the admin sub-router
# (``/v1/admin/*``). The tenant-config endpoints are gated by the
# admin-token middleware on the server, NOT the per-tenant bearer; the
# SDK forwards whatever auth header the parent client has on it and
# lets the server 401 the request when the wrong token is presented.
# An operator typically uses a dedicated client constructed with the
# admin token rather than the per-tenant bearer.


class _AdminNamespace:
    """``client.admin`` — tenant-config admin surface (0.5 addition).

    Today's only routes are the per-tenant replication-mode config
    (install + read). Both are idempotent; install replaces the prior
    config for the same tenant. A tenant whose config has never been
    installed runs as ``active_passive`` — the legacy direct-WAL path,
    same as today's behaviour."""

    def __init__(self, parent: "OriginChain") -> None:
        self._p = parent

    def install_tenant_config(
        self,
        tenant_id: str,
        replication_mode: Literal["active_passive", "raft_quorum"] = "active_passive",
    ) -> TenantConfigSnapshot:
        """``POST /v1/admin/tenants/:tenant/config`` — install (or
        replace) the replication mode for ``tenant_id``. Idempotent.

        ``"active_passive"`` is today's default and matches the legacy
        direct-WAL write path. ``"raft_quorum"`` routes writes through
        the Raft consensus path that landed in Phase D; the server
        rejects with 400 when the variant's ``is_implemented`` returns
        false (the SDK relays as :class:`OCValidationError`)."""
        body: dict[str, Any] = {"replication_mode": replication_mode}
        payload = self._p._request(
            "POST",
            f"/v1/admin/tenants/{tenant_id}/config",
            json=body,
        ).json()
        return TenantConfigSnapshot._from_payload(payload)

    def get_tenant_config(self, tenant_id: str) -> TenantConfigSnapshot:
        """``GET /v1/admin/tenants/:tenant/config`` — read the installed
        replication mode. Returns the implicit ``active_passive``
        default (with ``installed=None``) when no admin has ever
        installed a config for this tenant, so callers always see a
        truthful non-404 shape."""
        payload = self._p._request(
            "GET",
            f"/v1/admin/tenants/{tenant_id}/config",
        ).json()
        return TenantConfigSnapshot._from_payload(payload)


__all__ = [
    "_SqlNamespace",
    "_VectorNamespace",
    "_FtsNamespace",
    "_GraphNamespaceExtended",
    "_AdminNamespace",
]
