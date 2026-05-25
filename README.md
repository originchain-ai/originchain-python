# originchain (Python)

Official Python client for [OriginChain](https://originchain.ai).

> Other languages: TypeScript / JavaScript → [`@originchain/sdk`](https://www.npmjs.com/package/@originchain/sdk) · Go → [`github.com/originchain-ai/originchain-go`](https://github.com/originchain-ai/originchain-go) · raw HTTP → [originchain.ai/docs](https://originchain.ai/docs).

## Install

```bash
pip install originchain
```

HTTP/2 is enabled by default — multiplexed cross-region calls save ~300 ms each vs HTTP/1.1.

## Quick start

```python
from originchain import OriginChain

# Reads OC_BASE_URL, OC_BEARER, OC_TENANT from env.
db = OriginChain.from_env()

# Register a schema.
db.schemas.register(open("orders.toml").read())

# Insert a row.
db.rows.put(
    "trading.orders",
    {"order_id": "o1", "symbol": "AAPL", "qty": 100},
    idempotency_key="user-action-42",
)

# Ask in natural language.
result = db.ask("orders for AAPL above 50 shares last week")
for row in result["rows"]:
    print(row)
```

## Async

```python
import asyncio
from originchain import AsyncOriginChain

async def main() -> None:
    async with AsyncOriginChain.from_env() as db:
        await db.rows.put("trading.orders", {"order_id": "o2", "symbol": "MSFT", "qty": 50})
        print(await db.ask("show me the largest MSFT order this hour"))

asyncio.run(main())
```

## SQL

```python
resp = db.sql("SELECT order_id, qty FROM trading.orders WHERE symbol = 'AAPL'")
for row in resp.rows:
    print(row["order_id"], row["qty"])
```

`db.sql_one("SELECT ... LIMIT 1")` returns the first row dict (or `None`).

## Vector

```python
db.vector_put("embeddings", id="doc-1", embedding=[0.1, 0.2, 0.3], dim=3)
hits = db.vector_topk("embeddings", query=[0.1, 0.2, 0.3], k=5, dim=3)
for h in hits:
    print(h.id, h.score)
```

## Full-text

```python
db.fts_index("articles", "body", doc_id="d1", text="the quick brown fox")
hits = db.fts_search("articles", "body", q="quick fox", mode="bm25", k=5)
print([(h.doc_id, h.score) for h in hits])
```

## Graph

```python
nbrs = db.graph.neighbors("users", rel="follows", pk="u1")
for n in nbrs:
    print(n.pk)
result = db.graph.dijkstra("users", rel="follows", src="u1", dst="u9",
                           weights={"u1|u2": 1.0, "u2|u9": 0.5})
print(result.cost)
```

## Auth

Bearer tokens are minted in the OriginChain console and scoped to a
single instance. Treat them like database passwords — env vars or
secrets manager, never committed.

## Errors

The client maps HTTP errors to a typed hierarchy:

```python
from originchain import (
    OCAuthError,
    OCPaymentRequiredError,
    OCRateLimitedError,
    OCServerError,
)

try:
    db.rows.put("trading.orders", row)
except OCRateLimitedError as e:
    time.sleep(e.retry_after)
    db.rows.put("trading.orders", row)
except OCPaymentRequiredError as e:
    print(f"enable {e.name} at {e.purchase_url}")
except OCAuthError:
    print("rotate your bearer in the console")
```

When the leader returns 200 but synchronous replication didn't ack
within the configured timeout, the client emits an
`OCReplicationDegraded` warning. The write IS durable on the leader;
the warning surfaces a follower-lag signal callers can monitor or page
on.

## Versioning

This client follows semver. The `0.x` line is for design partners;
the API surface may change before `1.0`.
