# Changelog

All notable changes to the OriginChain Python SDK. See the repo-root
`CHANGELOG.md` for engine releases.

## [0.5.0] — 2026-07-15

Everything below is new relative to 0.4.0 **as published on PyPI**
(see the 0.4.0 note): the typed-namespace batch plus the follow-on
engine surfaces ship together in this release.

### Typed namespaces (sync client; async parity planned)

- **`client.sql` / `client.vector` / `client.fts` / `client.graph`
  namespaces** on the sync `OriginChain` client. Customers no longer
  hand-roll dicts and parse JSON manually for the four
  substrate-extension surfaces:
  - `client.sql.query(...)` / `client.sql.execute(...)`; the callable
    `client.sql(query)` and `client.sql_one(query)` return a
    `SqlSelect` / `SqlInsert` / `SqlDelete` discriminated union.
  - `client.vector.put(...)` / `client.vector.topk(...)` (plus the
    legacy `vector_put` / `vector_topk` methods) — return
    `list[VectorHit]`.
  - `client.fts.index(...)` / `client.fts.search(...)` with BM25,
    highlights, facets, `install_synonyms`, `install_stopwords`
    (plus legacy `fts_index` / `fts_search`).
  - `client.graph.{neighbors_of, bfs_of, shortest_path, k_shortest,
    random_walk, louvain, label_propagation, pagerank, betweenness}`
    plus the legacy kwarg-style `neighbors` / `reverse_neighbors` /
    `bfs` / `path` / `dijkstra`.
- **Frozen-dataclass response models** — `SqlSelect`, `SqlInsert`,
  `SqlDelete`, `VectorHit`, `FtsHit`, `Neighbor`, `GraphBfsHit`,
  `GraphPath`, `DijkstraResult` and friends — hashable + immutable,
  snake_case field names matching the wire.
- **`OCPaymentRequiredError`** — 402 add-on-required mapping. Surfaces
  `addon` / `name` / `monthly_usd` / `preview` / `enterprise_only` /
  `purchase_url` / `msg` as attributes.

### Vector

- `client.vector.delete(table, vec_id)` — single delete, end-to-end
- `client.vector.delete_bulk(table, ids)` — bulk-delete route (up to
  10 000 ids per call)
- `client.vector.install_centroids(table, centroids)` +
  `train_and_install_centroids(table, partitions, ...)` +
  `centroids(table)` — IVF centroid lifecycle
- `client.vector.rebalance_status(table)` — IVF rebalance status

### Graph

- `client.graph.node2vec_topk(schema, rel, query_pk, k, metric)` —
  over persisted Node2Vec embeddings
- `client.graph.graphsage(schema, feature_col, rel, config)` — train +
  optional persist
- `client.graph.graphsage_topk(schema, rel, query_pk, k, metric)` —
  over persisted GraphSAGE embeddings

### SQL

- `client.sql.install_materialized_view(name, query, refresh_mode)`
- `client.sql.refresh_materialized_view(name)` →
  `{ rows_materialized, bytes_written, refresh_ts }`
- `client.sql.read_materialized_view(name)`

### Admin

- `client.admin.install_tenant_config(tenant_id, replication_mode)`
- `client.admin.get_tenant_config(tenant_id)`

### Usage

- `client.usage()` (sync + async) — live usage counters, per-schema
  breakdown, and the tenant's compute configuration. New dataclasses
  `TenantUsage` and `TenantConfiguration`; the `tier` field carries the
  neutral configuration slug (`entry` / `standard` / `advanced` /
  `custom`).

### FTS (no API change — behavior upgraded server-side)

- Lemmatization is automatic when the table's analyzer config has
  `lemmatizer="dictionary"`; 9 languages now supported

### Packaging

- `project.urls` (Source / Issues) fixed to point at
  `github.com/originchain-ai/originchain-python` (previously a dead
  repository path)
- Added a `LICENSE` file matching the `Proprietary` license metadata
- Removed committed `__pycache__` artifacts; added `.gitignore`
- HTTP/2 (`h2`) is a hard dependency; the `[http2]` extra is kept as a
  no-op alias for 0.3.x requirements files

### Engine compatibility

- Requires an engine build that contains the IVF-PQ + GraphSAGE +
  materialized views + Raft Phase D commits (any deploy after
  2026-06-08 commit `2c1fe55a`)

### Migration from 0.4.0

- No breaking changes; all new methods are additive
- `client.vector.install_centroids` URL fixed from `install_centroids`
  → `install-centroids` to match the engine's admin-route convention
- Default `User-Agent` is `originchain-python/0.5.0`

## [0.4.0]

- **Note:** 0.4.0 as published contained no functional changes over
  0.3.0 — the published artifacts were byte-identical to 0.3.0 apart
  from the version string in the `User-Agent`. The typed-namespace
  work intended for 0.4.0 first ships in 0.5.0.

## 0.3.0

### Changed
- **`vector_topk` `mode` parameter is now `"fast" | "high_recall"`** -
  replaces the previous `"hnsw" | "bruteforce"` value space. The
  parameter is now optional (`mode=None` by default) and is omitted
  from the request body when unset; the server defaults to
  `"high_recall"` when the field is absent. `"fast"` favours latency,
  `"high_recall"` favours recall.
- Default `User-Agent` bumped to `originchain-python/0.3.0`.
