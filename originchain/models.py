"""Typed dataclass models for the SQL / vector / FTS / graph surfaces.

These mirror the JSON wire shapes emitted by ``oc-http``'s preview
endpoints (see ``backend/crates/oc-http/src/preview_endpoints.rs``).
Field names match the wire snake_case verbatim. Every dataclass is
``frozen=True`` so they're hashable and immutable — callers can stash
results in ``set`` / ``dict`` keys without worrying about mutation.

Decoders (``_from_payload`` classmethods) are kept tolerant of extra
fields so a server-side addition doesn't immediately break old clients
— unknown keys get dropped, known keys are coerced to the declared type
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
    is the raw JSON value list — each row is whatever shape the
    schema's projection emits."""

    rows: Tuple[Any, ...]
    kind: str = "select"


@dataclass(frozen=True)
class SqlInsert:
    """``{"kind": "insert", "schema": "...", "rows": [...]}``. Translated
    INSERT payload — the caller is expected to re-issue against
    ``/v1/tenants/:t/rows/:schema`` with idempotency. We don't auto-
    execute writes from ``/sql`` in v0; see preview_endpoints.rs."""

    schema: str
    rows: Tuple[Any, ...]
    kind: str = "insert"


@dataclass(frozen=True)
class SqlDelete:
    """``{"kind": "delete", "schema": "...", "pk": "..."}``. Translated
    DELETE — caller re-issues against ``/v1/tenants/:t/rows/:schema/:pk``."""

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
    depend on the metric the index was built with — cosine and dot
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
    the substrate returns in v0 — the actual path itself is not
    materialised. v1 will surface the edge list."""

    reachable: bool


@dataclass(frozen=True)
class DijkstraResult:
    """``/graph/:schema/dijkstra`` response. ``cost`` is ``None`` when
    the destination is unreachable from the source under the supplied
    weight function, otherwise the total weight along the cheapest
    path."""

    cost: Optional[float]


__all__ = [
    "SqlSelect",
    "SqlInsert",
    "SqlDelete",
    "SqlResponse",
    "VectorHit",
    "FtsHit",
    "Neighbor",
    "GraphBfsHit",
    "GraphPath",
    "DijkstraResult",
]
