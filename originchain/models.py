"""Typed dataclass models for the SQL / vector / FTS / graph surfaces.

These mirror the JSON wire shapes emitted by ``oc-http``'s preview
endpoints (see ``backend/crates/oc-http/src/preview_endpoints.rs``).
Field names match the wire snake_case verbatim. Every dataclass is
``frozen=True`` so they're hashable and immutable - callers can stash
results in ``set`` / ``dict`` keys without worrying about mutation.

Decoders (``_from_payload`` classmethods) are kept tolerant of extra
fields so a server-side addition doesn't immediately break old clients
- unknown keys get dropped, known keys are coerced to the declared type
where it's cheap (e.g. ``int`` from ``float`` for depths, ``str`` from
non-string PKs that the substrate handed back as a JSON-stringified
fallback). If the server changes the shape in a non-backward-compatible
way, the SDK gets bumped major.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Union


# ─────────────────────────── SQL ───────────────────────────


@dataclass(frozen=True)
class SqlSelect:
    """``{"kind": "select", "rows": [...]}``. The substrate ran a SELECT
    against the row k/v store and returned the rows verbatim. ``rows``
    is the raw JSON value list - each row is whatever shape the
    schema's projection emits."""

    rows: Tuple[Any, ...]
    kind: str = "select"


@dataclass(frozen=True)
class SqlInsert:
    """``{"kind": "insert", "schema": "...", "rows": [...]}``. Translated
    INSERT payload - the caller is expected to re-issue against
    ``/v1/tenants/:t/rows/:schema`` with idempotency. We don't auto-
    execute writes from ``/sql`` in v0; see preview_endpoints.rs."""

    schema: str
    rows: Tuple[Any, ...]
    kind: str = "insert"


@dataclass(frozen=True)
class SqlDelete:
    """``{"kind": "delete", "schema": "...", "pk": "..."}``. Translated
    DELETE - caller re-issues against ``/v1/tenants/:t/rows/:schema/:pk``."""

    schema: str
    pk: str
    kind: str = "delete"


SqlResponse = Union[SqlSelect, SqlInsert, SqlDelete]


def _decode_sql_response(payload: Mapping[str, Any]) -> SqlResponse:
    """Tagged-union decode. ``kind`` discriminates on the wire."""
    kind = payload.get("kind")
    if kind == "select":
        rows = payload.get("rows", [])
        return SqlSelect(rows=tuple(rows))
    if kind == "insert":
        return SqlInsert(
            schema=str(payload["schema"]),
            rows=tuple(payload.get("rows", [])),
        )
    if kind == "delete":
        return SqlDelete(
            schema=str(payload["schema"]),
            pk=str(payload["pk"]),
        )
    raise ValueError(f"unknown SQL response kind: {kind!r}")


# ─────────────────────────── Vector ───────────────────────────


@dataclass(frozen=True)
class VectorHit:
    """One topk hit: ``{"id": "...", "score": 0.93}``. Score semantics
    depend on the metric the index was built with - cosine and dot
    return higher-is-closer, l2 returns lower-is-closer. The SDK doesn't
    re-sort or normalise; the substrate already returns hits in the
    right order for the metric."""

    id: str
    score: float

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "VectorHit":
        return cls(id=str(payload["id"]), score=float(payload["score"]))


# ─────────────────────────── Full-text ───────────────────────────


@dataclass(frozen=True)
class FtsHit:
    """One BM25-ranked hit: ``{"doc_id": "...", "score": 4.21}``.

    Boolean / phrase mode return ``doc_id`` only; ``score`` is set to
    ``0.0`` in those modes so the dataclass shape is uniform."""

    doc_id: str
    score: float = 0.0

    @classmethod
    def _from_ranked(cls, payload: Mapping[str, Any]) -> "FtsHit":
        return cls(doc_id=str(payload["doc_id"]), score=float(payload["score"]))

    @classmethod
    def _from_doc_id(cls, doc_id: str) -> "FtsHit":
        return cls(doc_id=doc_id, score=0.0)


# ─────────────────────────── Graph ───────────────────────────


@dataclass(frozen=True)
class Neighbor:
    """One neighbour PK from ``/graph/:schema/neighbors`` or
    ``/graph/:schema/reverse``. ``depth`` is always ``1`` for these
    one-hop endpoints; BFS hits use :class:`GraphBfsHit` instead."""

    pk: str
    depth: int = 1


@dataclass(frozen=True)
class GraphBfsHit:
    """``{"pk": "...", "depth": N}`` returned by ``/graph/:schema/bfs``.
    ``depth`` is the BFS distance from the source PK."""

    pk: str
    depth: int

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "GraphBfsHit":
        return cls(pk=str(payload["pk"]), depth=int(payload["depth"]))


@dataclass(frozen=True)
class GraphPath:
    """``/graph/:schema/path`` response. ``reachable`` is the only field
    the substrate returns in v0 - the actual path itself is not
    materialised. v1 will surface the edge list."""

    reachable: bool


@dataclass(frozen=True)
class DijkstraResult:
    """``/graph/:schema/dijkstra`` response. ``cost`` is ``None`` when
    the destination is unreachable from the source under the supplied
    weight function, otherwise the total weight along the cheapest
    path."""

    cost: Optional[float]


# ─────────────────────────── Typed-namespace v1 shapes ───────────────────────
# Added 2026-06-07 alongside the `client.sql` / `client.vector` /
# `client.fts` / `client.graph` typed namespace surface. The legacy
# dataclasses above are kept verbatim so existing call-sites
# (`client.sql(...)`, `client.vector_topk(...)`, `client.graph.bfs(...)`)
# don't break — these new shapes wrap richer responses (rows + columns
# for SQL, hits + facets for FTS, nodes + cost for paths) that the
# legacy methods didn't surface.


@dataclass(frozen=True)
class SqlResult:
    """SELECT-shape response with both rows and an optional column list.

    ``rows`` is the raw row list as returned by the substrate. ``columns``
    is the projection order when the server emits it (the preview
    ``/sql`` endpoint does not always include a separate columns array;
    when absent, fall back to ``list(row.keys())`` on the first row)."""

    rows: list[dict[str, Any]]
    columns: Optional[list[str]] = None


@dataclass(frozen=True)
class SqlExecResult:
    """Non-SELECT-shape response. ``rows_affected`` is best-effort — the
    preview ``/sql`` endpoint translates writes into typed row payloads
    rather than executing them inline, so for INSERT/DELETE the SDK
    re-issues against the typed row routes and surfaces the count the
    substrate confirms. ``kind`` echoes the wire discriminator."""

    kind: str
    rows_affected: int = 0
    schema: Optional[str] = None


@dataclass(frozen=True)
class FacetBucket:
    """One bucket in a facet aggregation. Wire shape: ``{"value": "...",
    "count": N}``. Mirrors ``oc_fulltext::FacetBucket`` directly."""

    value: str
    count: int

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "FacetBucket":
        return cls(value=str(payload["value"]), count=int(payload["count"]))


@dataclass(frozen=True)
class FtsHitWithHighlights:
    """One BM25 hit with optional per-field highlight snippets. Boolean /
    phrase modes produce this dataclass too with ``score=0.0`` and no
    highlights so the typed-namespace shape stays uniform.

    Distinct from the legacy :class:`FtsHit` (which doesn't carry
    highlights) so callers can keep importing the simpler shape from
    pre-typed-namespace code without picking up a wider interface."""

    doc_id: str
    score: float
    highlights: Optional[dict[str, list[str]]] = None

    @classmethod
    def _from_ranked(cls, payload: Mapping[str, Any]) -> "FtsHitWithHighlights":
        return cls(doc_id=str(payload["doc_id"]), score=float(payload["score"]))

    @classmethod
    def _from_enriched(cls, payload: Mapping[str, Any]) -> "FtsHitWithHighlights":
        hl_raw = payload.get("highlights") or {}
        hl: Optional[dict[str, list[str]]] = (
            {str(k): [str(s) for s in v] for k, v in hl_raw.items()} if hl_raw else None
        )
        return cls(
            doc_id=str(payload["doc_id"]),
            score=float(payload["score"]),
            highlights=hl,
        )

    @classmethod
    def _from_doc_id(cls, doc_id: str) -> "FtsHitWithHighlights":
        return cls(doc_id=doc_id, score=0.0, highlights=None)


@dataclass(frozen=True)
class FtsResult:
    """Container for the BM25 / boolean / phrase response. ``hits`` is
    always populated; ``facets`` is populated only when the caller
    passed ``facets=[...]`` and the server returned the enriched
    envelope (``{hits: [...], facets: {field: [...]}}``)."""

    hits: list[FtsHitWithHighlights]
    facets: Optional[dict[str, list[FacetBucket]]] = None


@dataclass(frozen=True)
class Path:
    """One ranked graph path. ``nodes`` is the decoded PK label list,
    source first; ``cost`` is the sum of edge weights along the path
    (defaults to hop count when the relation has no weight column)."""

    nodes: list[str]
    cost: float

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "Path":
        nodes = [str(n) for n in payload.get("nodes", [])]
        return cls(nodes=nodes, cost=float(payload.get("cost", 0.0)))


@dataclass(frozen=True)
class InstallCentroidsResult:
    """``/vector/:table/install-centroids`` response. Echoes the number
    of partitions installed and the vector dimensionality the server
    accepted, so callers can sanity-check against their corpus shape."""

    installed: bool
    partitions: int
    dim: int

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "InstallCentroidsResult":
        return cls(
            installed=bool(payload.get("installed", True)),
            partitions=int(payload.get("partitions", 0)),
            dim=int(payload.get("dim", 0)),
        )


# ─────────────────── 0.5 additions (2026-06-08) ───────────────────
# Wraps engine endpoints that landed after the 0.4.0 typed-namespace
# batch (commit 6b2f0046). Each dataclass mirrors the wire shape from
# the named handler in `backend/crates/oc-http/src/handlers/`; field
# names match snake_case verbatim.


@dataclass(frozen=True)
class VectorDeleteResult:
    """``DELETE /vector/:table/:vec_id`` response. The handler returns
    ``{"deleted": true}`` on a real removal and ``{"deleted": false}``
    on an idempotent missing-row no-op — never 404, so callers don't
    have to swallow not-found in cleanup paths."""

    deleted: bool

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "VectorDeleteResult":
        return cls(deleted=bool(payload.get("deleted", False)))


@dataclass(frozen=True)
class VectorDeleteBulkResult:
    """``POST /vector/:table/delete-bulk`` response. ``deleted_count``
    counts ids that were live before the call; ``missing_count`` counts
    ids that were already absent. Sum may be less than ``len(ids)``
    after dedup — the server collapses duplicate ids in one batch."""

    deleted_count: int
    missing_count: int

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "VectorDeleteBulkResult":
        return cls(
            deleted_count=int(payload.get("deleted_count", 0)),
            missing_count=int(payload.get("missing_count", 0)),
        )


@dataclass(frozen=True)
class IvfRebalanceStatus:
    """``GET /vector/:table/ivf-rebalance-status`` response. ``action``
    is one of ``"none" | "recommended" | "required"``; nothing is
    auto-triggered server-side — the operator reads this and decides
    whether to call ``train_and_install_centroids`` again."""

    total_live: int
    partitions: int
    live_per_cell: list[int]
    skew: float
    action: str

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "IvfRebalanceStatus":
        raw_action = payload.get("action", "none")
        # Server emits the action as a serde-tagged string ("None" /
        # "Recommended" / "Required" or the snake_case equivalent
        # depending on the serde-rename in oc_vector::RebalanceAction).
        # Normalise to lowercase snake_case so SDK callers see a single
        # vocabulary regardless of the server's serde-attr choices.
        action_str = str(raw_action).lower()
        return cls(
            total_live=int(payload.get("total_live", 0)),
            partitions=int(payload.get("partitions", 0)),
            live_per_cell=[int(c) for c in payload.get("live_per_cell", [])],
            skew=float(payload.get("skew", 0.0)),
            action=action_str,
        )


@dataclass(frozen=True)
class TrainAndInstallCentroidsResult:
    """``POST /vector/:table/train-and-install-centroids`` response.
    Carries both training diagnostics (``iterations`` / ``converged``
    / ``last_max_shift`` / ``training_corpus_size``) and the install
    outcome (``installed`` / ``partitions`` / ``dim``)."""

    trained: bool
    installed: bool
    partitions: int
    dim: int
    iterations: int
    converged: bool
    last_max_shift: float
    training_corpus_size: int

    @classmethod
    def _from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "TrainAndInstallCentroidsResult":
        return cls(
            trained=bool(payload.get("trained", False)),
            installed=bool(payload.get("installed", False)),
            partitions=int(payload.get("partitions", 0)),
            dim=int(payload.get("dim", 0)),
            iterations=int(payload.get("iterations", 0)),
            converged=bool(payload.get("converged", False)),
            last_max_shift=float(payload.get("last_max_shift", 0.0)),
            training_corpus_size=int(payload.get("training_corpus_size", 0)),
        )


@dataclass(frozen=True)
class CentroidsPreview:
    """``GET /vector/:table/centroids`` response. ``installed=False``
    when no centroids have been installed yet; ``centroids_preview`` is
    a truncated peek (server-side cap is 4 centroids × first 8 dims)
    so the response stays cheap even at production dimensionality."""

    installed: bool
    partitions: int
    dim: int
    centroids_preview: list[list[float]]

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "CentroidsPreview":
        raw_preview = payload.get("centroids_preview", [])
        preview: list[list[float]] = [
            [float(x) for x in row] for row in raw_preview
        ]
        return cls(
            installed=bool(payload.get("installed", False)),
            partitions=int(payload.get("partitions", 0)),
            dim=int(payload.get("dim", 0)),
            centroids_preview=preview,
        )


@dataclass(frozen=True)
class GraphEmbeddingHit:
    """One ``(pk, score)`` row from the persisted-embedding topk
    endpoints (``/graph/:schema/node2vec/:rel/topk`` and the GraphSAGE
    sibling). Score semantics follow the metric: cosine/dot are
    higher-is-closer; l2/manhattan are lower-is-closer."""

    pk: str
    score: float

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "GraphEmbeddingHit":
        return cls(pk=str(payload["pk"]), score=float(payload["score"]))


@dataclass(frozen=True)
class GraphSageResult:
    """``POST /graph/:schema/graphsage`` response. ``embeddings`` is the
    ``{pk_label: [..f32..]}`` map the server returned (decoded via the
    same ``decode_pk_label`` path every graph endpoint uses).
    ``persisted`` mirrors the request's ``persist`` flag — when ``True``
    the sibling ``/graphsage/:rel/topk`` endpoint becomes callable."""

    embeddings: dict[str, list[float]]
    vocab_size: int
    training_pairs: int
    final_loss: float
    embedding_dim: int
    feature_dim: int
    persisted: bool

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "GraphSageResult":
        raw_emb = payload.get("embeddings", {})
        emb: dict[str, list[float]] = {
            str(k): [float(x) for x in v] for k, v in raw_emb.items()
        }
        return cls(
            embeddings=emb,
            vocab_size=int(payload.get("vocab_size", 0)),
            training_pairs=int(payload.get("training_pairs", 0)),
            final_loss=float(payload.get("final_loss", 0.0)),
            embedding_dim=int(payload.get("embedding_dim", 0)),
            feature_dim=int(payload.get("feature_dim", 0)),
            persisted=bool(payload.get("persisted", False)),
        )


@dataclass(frozen=True)
class MaterializedViewInstallResult:
    """``POST /sql/materialized-views`` response. ``rows_materialized``
    and ``bytes_written`` describe the snapshot the executor just
    stamped; ``refresh_ts`` is the server-side unix-epoch second at
    which the snapshot was taken."""

    name: str
    rows_materialized: int
    bytes_written: int
    refresh_ts: int

    @classmethod
    def _from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "MaterializedViewInstallResult":
        return cls(
            name=str(payload.get("name", "")),
            rows_materialized=int(payload.get("rows_materialized", 0)),
            bytes_written=int(payload.get("bytes_written", 0)),
            refresh_ts=int(payload.get("refresh_ts", 0)),
        )


@dataclass(frozen=True)
class MaterializedViewRefreshResult:
    """``POST /sql/materialized-views/:name/refresh`` response. Same
    shape as :class:`MaterializedViewInstallResult` — every refresh
    atomically overwrites the prior snapshot."""

    name: str
    rows_materialized: int
    bytes_written: int
    refresh_ts: int

    @classmethod
    def _from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "MaterializedViewRefreshResult":
        return cls(
            name=str(payload.get("name", "")),
            rows_materialized=int(payload.get("rows_materialized", 0)),
            bytes_written=int(payload.get("bytes_written", 0)),
            refresh_ts=int(payload.get("refresh_ts", 0)),
        )


@dataclass(frozen=True)
class MaterializedViewRows:
    """``GET /sql/materialized-views/:name`` response. ``rows`` carries
    the rmp-decoded row bundle the executor stamped at the last
    refresh; the shape is whatever the original SELECT projected."""

    name: str
    rows: list[Any]

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "MaterializedViewRows":
        rows = payload.get("rows", [])
        return cls(
            name=str(payload.get("name", "")),
            rows=list(rows),
        )


@dataclass(frozen=True)
class TenantConfigSnapshot:
    """``GET / POST /v1/admin/tenants/:tenant/config`` response.
    ``replication_mode`` is one of ``"active_passive"`` (today's default,
    direct-WAL writes) or ``"raft_quorum"`` (Phase D consensus path).
    ``installed`` is ``True`` on the install response and ``None`` on
    the read response — the same shape serves both endpoints."""

    replication_mode: str
    installed: Optional[bool] = None

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "TenantConfigSnapshot":
        inst = payload.get("installed")
        return cls(
            replication_mode=str(payload.get("replication_mode", "active_passive")),
            installed=None if inst is None else bool(inst),
        )


@dataclass(frozen=True)
class VectorHitV2:
    """Typed-namespace VectorHit. Distinct from the legacy
    :class:`VectorHit` because it also surfaces server-returned
    metadata (cosine-similar callers often want the original tag /
    source URL alongside the score). ``vec_id`` mirrors the spec field
    name; ``id`` is exposed as a property alias for legacy code."""

    vec_id: str
    score: float
    metadata: Optional[dict[str, Any]] = None

    @property
    def id(self) -> str:
        return self.vec_id

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "VectorHitV2":
        md_raw = payload.get("metadata")
        md: Optional[dict[str, Any]] = dict(md_raw) if isinstance(md_raw, dict) else None
        return cls(
            vec_id=str(payload.get("id") or payload.get("vec_id")),
            score=float(payload["score"]),
            metadata=md,
        )


# ─────────────────────────── Usage / configuration ───────────────────────────


@dataclass(frozen=True)
class TenantConfiguration:
    """Neutral, spec-based compute configuration from ``GET
    /v1/tenants/:t/usage``.

    This REPLACES the internal weather codename (thunder/storm/cyclone/…)
    the engine used to surface - the SDK never exposes that codename.
    ``slug`` is the stable machine id (``entry`` / ``standard`` /
    ``advanced`` / ``custom``); ``label`` is display text such as
    ``"4 vCPU / 16 GB, HA"``. Quantitative fields and ``monthly_price``
    are ``None`` for the sales-sized ``custom`` configuration."""

    slug: str
    label: str
    ha: bool
    vcpu: Optional[int] = None
    ram_gb: Optional[int] = None
    storage_gb: Optional[int] = None
    monthly_price: Optional[int] = None

    @classmethod
    def _from_payload(cls, p: Mapping[str, Any]) -> "TenantConfiguration":
        return cls(
            slug=str(p.get("slug", "")),
            label=str(p.get("label", "")),
            ha=bool(p.get("ha", False)),
            vcpu=int(p["vcpu"]) if p.get("vcpu") is not None else None,
            ram_gb=int(p["ram_gb"]) if p.get("ram_gb") is not None else None,
            storage_gb=int(p["storage_gb"]) if p.get("storage_gb") is not None else None,
            monthly_price=(
                int(p["monthly_price"]) if p.get("monthly_price") is not None else None
            ),
        )


@dataclass(frozen=True)
class TenantUsage:
    """Response of the engine's ``GET /v1/tenants/:t/usage``.

    ``tier`` is the neutral configuration slug (``entry`` / ``standard``
    / ``advanced`` / ``custom``) - never the internal weather codename,
    which the SDK does not expose. Prefer ``configuration`` for the full
    spec + list price. ``tier``, ``configuration``, and ``limits`` are
    all ``None`` in legacy per-addon mode."""

    tenant: str
    used: Mapping[str, Any]
    tier: Optional[str] = None
    configuration: Optional[TenantConfiguration] = None
    limits: Optional[Mapping[str, Any]] = None
    schemas: Tuple[Any, ...] = ()

    @classmethod
    def _from_payload(cls, p: Mapping[str, Any]) -> "TenantUsage":
        cfg = p.get("configuration")
        limits = p.get("limits")
        return cls(
            tenant=str(p.get("tenant", "")),
            used=dict(p.get("used", {})),
            tier=str(p["tier"]) if p.get("tier") is not None else None,
            configuration=(
                TenantConfiguration._from_payload(cfg)
                if isinstance(cfg, Mapping)
                else None
            ),
            limits=dict(limits) if isinstance(limits, Mapping) else None,
            schemas=tuple(p.get("schemas", ()) or ()),
        )


__all__ = [
    "SqlSelect",
    "SqlInsert",
    "SqlDelete",
    "SqlResponse",
    "SqlResult",
    "SqlExecResult",
    "VectorHit",
    "VectorHitV2",
    "FtsHit",
    "FtsHitWithHighlights",
    "FtsResult",
    "FacetBucket",
    "Neighbor",
    "GraphBfsHit",
    "GraphPath",
    "DijkstraResult",
    "Path",
    "InstallCentroidsResult",
    # 0.5 additions
    "VectorDeleteResult",
    "VectorDeleteBulkResult",
    "IvfRebalanceStatus",
    "TrainAndInstallCentroidsResult",
    "CentroidsPreview",
    "GraphEmbeddingHit",
    "GraphSageResult",
    "MaterializedViewInstallResult",
    "MaterializedViewRefreshResult",
    "MaterializedViewRows",
    "TenantConfigSnapshot",
    # weather-name removal
    "TenantConfiguration",
    "TenantUsage",
]
