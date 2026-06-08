# originchain (Python)

Official Python client for [OriginChain](https://originchain.ai).

> Other languages: TypeScript / JavaScript → [`@originchain/sdk`](https://www.npmjs.com/package/@originchain/sdk) · Go → [`github.com/originchain-ai/originchain-go`](https://github.com/originchain-ai/originchain-go) · raw HTTP → [originchain.ai/docs](https://originchain.ai/docs).

## Install

```bash
pip install originchain
```

HTTP/2 is enabled by default - multiplexed cross-region calls save ~300 ms each vs HTTP/1.1.

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
# Typed surface (recommended):
result = db.sql.query(
    "SELECT order_id, qty FROM trading.orders WHERE symbol = :s",
    params={"s": "AAPL"},
)
for row in result.rows:
    print(row["order_id"], row["qty"])
print(result.columns)  # ["order_id", "qty"]

# Non-SELECT statements come back as a translation envelope:
exec_result = db.sql.execute("INSERT INTO trading.orders ...")
print(exec_result.kind, exec_result.rows_affected)
```

The legacy callable `db.sql("SELECT ...")` still works and returns the
tagged-union dataclass. `db.sql_one("SELECT ... LIMIT 1")` returns the
first row dict (or `None`).

## Vector

```python
# Typed surface (recommended): `dim` is derived from the vector length.
db.vector.put("embeddings", "doc-1", [0.1, 0.2, 0.3], metadata={"src": "wiki"})
hits = db.vector.topk(
    "embeddings",
    [0.1, 0.2, 0.3],
    k=5,
    metric="cosine",
    filter={"src": "wiki"},   # optional metadata filter
    nprobe=8,                  # optional IVF tuning knob
)
for h in hits:
    print(h.vec_id, h.score, h.metadata)

db.vector.delete("embeddings", "doc-1")

# Pre-install IVF centroids for cold-start tables:
res = db.vector.install_centroids("embeddings", centroids=[[...], [...]])
print(res.installed, res.partitions, res.dim)
```

The legacy `db.vector_put(...)` / `db.vector_topk(...)` methods stay
available for code written before the typed namespace landed.

## Full-text

```python
db.fts.index("articles", "body", "d1", "the quick brown fox")

# BM25 with highlight snippets + facet aggregation.
result = db.fts.search(
    "articles",
    "body",
    "quick fox",
    mode="bm25",
    fuzzy=1,
    highlight=True,
    facets=["color", "brand"],
)
for h in result.hits:
    print(h.doc_id, h.score, h.highlights)
print(result.facets)  # {"color": [FacetBucket(value="red", count=3), ...]}

# Per-(table, field) admin:
db.fts.install_synonyms("articles", "body", {"car": ["auto", "vehicle"]})
db.fts.install_stopwords("articles", "body", ["the", "a", "an"])
```

`db.fts_index(...)` / `db.fts_search(...)` remain for back-compat.

## Graph

```python
# Single-hop neighbours + BFS:
nbrs   = db.graph.neighbors_of("users", "u1", "follows")
bfs    = db.graph.bfs_of("users", "u1", "follows", max_depth=3)

# Shortest path + Yen's K-shortest:
path   = db.graph.shortest_path("users", "u1", "u9", "follows")
ranked = db.graph.k_shortest("users", "u1", "u9", "follows", k=5,
                             weight_col="edge_weight")

# Random walk (uniform; p/q != 1 routes through Node2Vec-biased):
walk = db.graph.random_walk("users", "u1", "follows", steps=10, seed=42)

# Community detection + centrality:
communities = db.graph.louvain("users", "follows")
labels      = db.graph.label_propagation("users", "follows", seed=1)
ranks       = db.graph.pagerank("users", "follows", nodes=["u1", "u2", "u3"])
between     = db.graph.betweenness("users", "follows")
```

The legacy kwarg-style helpers (`db.graph.neighbors(...)`,
`db.graph.bfs(...)`, `db.graph.path(...)`, `db.graph.dijkstra(...)`,
`db.graph.reverse_neighbors(...)`) stay available.

## Auth

Bearer tokens are minted in the OriginChain console and scoped to a
single instance. Treat them like database passwords - env vars or
secrets manager, never committed.

## Errors

The client maps HTTP errors to a typed hierarchy:

```python
from originchain import (
    OCAuthError,
    OCPaymentRequiredError,
    OCRateLimitedError,
    OCServerError,
    OriginChainBadRequest,   # alias of OCValidationError (4xx)
    OriginChainServerError,  # alias of OCServerError    (5xx)
)

try:
    db.rows.put("trading.orders", row)
except OCRateLimitedError as e:
    time.sleep(e.retry_after)
    db.rows.put("trading.orders", row)
except OCPaymentRequiredError as e:
    print(f"enable {e.name} at {e.purchase_url}")
except OriginChainBadRequest:
    print("bad request — fix it before retrying")
except OCAuthError:
    print("rotate your bearer in the console")
```

When the leader returns 200 but synchronous replication didn't ack
within the configured timeout, the client emits an
`OCReplicationDegraded` warning. The write IS durable on the leader;
the warning surfaces a follower-lag signal callers can monitor or page
on.

## What's new in 0.5

0.5.0 wires up the engine endpoints that shipped after the
typed-namespace batch (`0.4.0`). All new methods are additive — no
breaking changes — and require an engine deployed at commit
`2c1fe55a` or later.

### Vector: bulk delete + IVF lifecycle

```python
# Single delete: now actually works end-to-end. 0.4 was wire-ready but
# the Rust handler hadn't shipped; missing-row returns deleted=False
# (200), not 404, so cleanup loops never need try/except.
result = db.vector.delete("embeddings", "doc-1")
print(result.deleted)

# Bulk delete (up to 10 000 ids per call, one WAL frame):
result = db.vector.delete_bulk("embeddings", ["doc-1", "doc-2", "doc-3"])
print(result.deleted_count, result.missing_count)

# IVF centroid lifecycle — bootstrap an IVF table after a row of writes:
db.vector.train_and_install_centroids("embeddings", partitions=64, seed=42)
preview  = db.vector.centroids("embeddings")           # installed? sample?
status   = db.vector.rebalance_status("embeddings")    # skew + action hint
```

### Graph: Node2Vec topk + GraphSAGE

```python
# Top-k similar nodes against persisted Node2Vec embeddings:
hits = db.graph.node2vec_topk("users", "follows", "u1", k=10)

# GraphSAGE attribute-aware embeddings (train + optionally persist):
res = db.graph.graphsage(
    "users",
    feature_col="profile_vec",
    rel="follows",
    config={"embedding_dim": 128, "epochs": 5, "seed": 42},
    persist=True,
)
print(res.vocab_size, res.final_loss, res.persisted)

# Persisted-embedding similarity (same shape as node2vec_topk):
hits = db.graph.graphsage_topk("users", "follows", "u1", k=10)
```

### SQL: materialized views

```python
res = db.sql.install_materialized_view(
    "daily_orders",
    "SELECT order_id, qty FROM trading.orders",
    refresh_mode="manual",
)
print(res.rows_materialized, res.bytes_written, res.refresh_ts)

# On-demand refresh — atomically overwrites the snapshot:
db.sql.refresh_materialized_view("daily_orders")

# Snapshot read:
view = db.sql.read_materialized_view("daily_orders")
for row in view.rows:
    print(row)
```

### Admin: per-tenant replication config

```python
# Install / change the replication topology for one tenant:
db.admin.install_tenant_config(tenant_id, replication_mode="raft_quorum")

# Read the installed mode (returns active_passive when none installed):
cfg = db.admin.get_tenant_config(tenant_id)
print(cfg.replication_mode, cfg.installed)
```

### FTS lemmatization

No SDK API change — `db.fts.install_synonyms(...)` /
`install_stopwords(...)` are unchanged. Server-side, the analyzer now
applies dictionary-based lemmatization across 9 languages when the
table's analyzer config selects `lemmatizer="dictionary"`.

## Versioning

This client follows semver. The `0.x` line is for design partners;
the API surface may change before `1.0`.
