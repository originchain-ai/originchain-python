"""OriginChain Python client.

Two clients ship in this package:

- :class:`OriginChain` - synchronous, suitable for scripts and Jupyter.
- :class:`AsyncOriginChain` - asyncio, suitable for ASGI / async data
  pipelines.

Both wrap the same ``/v1`` HTTP surface that the engine exposes; they
encode bearer auth, idempotency keys, and retries so callers don't have
to repeat themselves.

Quick start::

    from originchain import OriginChain

    db = OriginChain.from_env()                  # OC_BASE_URL + OC_BEARER
    db.schemas.register(open("orders.toml").read())
    db.rows.put("trading.orders", {"order_id": "o1", "symbol": "AAPL", "qty": 100})
    rows = db.ask("orders for AAPL above 50 shares last week")

The four substrate-extension surfaces (SQL, vector, full-text, graph)
are also typed first-class. See :class:`OriginChain.sql`,
``vector_topk`` / ``vector_put``, ``fts_search``, and the ``graph``
namespace.
"""

from .client import OriginChain
from .async_client import AsyncOriginChain
from .errors import (
    OCAuthError,
    OCError,
    OCNotFoundError,
    OCPaymentRequiredError,
    OCRateLimitedError,
    OCReplicationDegraded,
    OCServerError,
    OCValidationError,
    OriginChainBadRequest,
    OriginChainServerError,
)
from .models import (
    DijkstraResult,
    FacetBucket,
    FtsHit,
    FtsHitWithHighlights,
    FtsResult,
    GraphBfsHit,
    GraphPath,
    InstallCentroidsResult,
    Neighbor,
    Path,
    SqlDelete,
    SqlExecResult,
    SqlInsert,
    SqlResponse,
    SqlResult,
    SqlSelect,
    VectorHit,
    VectorHitV2,
)

__all__ = [
    "OriginChain",
    "AsyncOriginChain",
    # Errors
    "OCError",
    "OCAuthError",
    "OCNotFoundError",
    "OCPaymentRequiredError",
    "OCRateLimitedError",
    "OCServerError",
    "OCValidationError",
    "OCReplicationDegraded",
    "OriginChainBadRequest",
    "OriginChainServerError",
    # Models
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

__version__ = "0.4.0"
