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
    """``/vector/:table/install_centroids`` response. Echoes the number
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
]
