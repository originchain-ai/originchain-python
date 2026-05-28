"""Exception hierarchy for the Python client.

All OC errors derive from :class:`OCError` so callers can catch the base
class. Specific subclasses let callers branch on retryability:

- :class:`OCAuthError` (401 / 403) - DON'T retry, the bearer is wrong.
- :class:`OCPaymentRequiredError` (402) - DON'T retry, an add-on is
  required to call this endpoint.
- :class:`OCNotFoundError` (404) - DON'T retry, the resource doesn't exist.
- :class:`OCValidationError` (400) - DON'T retry, the request is malformed.
- :class:`OCRateLimitedError` (429) - DO retry after ``Retry-After``.
- :class:`OCServerError` (5xx) - MAY retry with backoff if idempotent.
- :class:`OCReplicationDegraded` - write succeeded, but sync-replication
  timed out. Set on the response, not raised. Callers can opt to log /
  page on this signal.
"""

from __future__ import annotations

from typing import Any, Optional


class OCError(Exception):
    """Base for every error raised by this client."""

    def __init__(self, message: str, *, status: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class OCAuthError(OCError):
    """401 / 403. Bearer is missing, wrong, or not scoped to this tenant."""


class OCPaymentRequiredError(OCError):
    """402. The endpoint is gated behind an add-on the tenant hasn't
    purchased. The 402 body shape is:

    .. code-block:: json

        {
          "error": "addon_required",
          "addon": "vector-search",
          "name":  "Vector Search",
          "monthly_usd": 49,
          "preview": false,
          "enterprise_only": false,
          "purchase_url": "https://app.originchain.ai/billing/addons?enable=vector-search",
          "msg": "This endpoint requires the Vector Search add-on. ..."
        }

    Surfaced to callers as attributes so they don't have to dig into
    ``.body`` themselves. ``addon`` / ``name`` / ``monthly_usd`` /
    ``preview`` / ``enterprise_only`` / ``purchase_url`` / ``msg`` are
    populated when the server returns the canonical body and ``None``
    otherwise."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, **kwargs)
        body = self.body if isinstance(self.body, dict) else {}
        self.addon: Optional[str] = body.get("addon")
        self.name: Optional[str] = body.get("name")
        self.monthly_usd: Optional[float] = body.get("monthly_usd")
        self.preview: Optional[bool] = body.get("preview")
        self.enterprise_only: Optional[bool] = body.get("enterprise_only")
        self.purchase_url: Optional[str] = body.get("purchase_url")
        self.msg: Optional[str] = body.get("msg")


class OCNotFoundError(OCError):
    """404. The schema / row / instance doesn't exist."""


class OCValidationError(OCError):
    """400. The request body or query parameters are malformed."""


class OCRateLimitedError(OCError):
    """429. The token's bucket is exhausted. ``retry_after`` is in seconds."""

    def __init__(self, message: str, *, retry_after: float = 1.0, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class OCServerError(OCError):
    """5xx. Server-side failure; retryable for idempotent ops."""


class OCReplicationDegraded(Warning):
    """The leader returned 200 but the follower didn't ack within the
    ``--sync-timeout-ms`` window. Surfaced as a warning rather than an
    error because the write IS durable on the leader - a follower lag
    or fence is the real cause and should be paged separately."""
