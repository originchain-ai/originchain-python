# Changelog

All notable changes to the OriginChain Python SDK. See the repo-root
`CHANGELOG.md` for engine releases.

## 0.5.0 — 2026-06-08

### New typed methods

#### Vector
- `client.vector.delete(table, vec_id)` — Rust handler now exists
- `client.vector.delete_bulk(table, ids)` — new bulk-delete route
- `client.vector.install_centroids(table, centroids)` + `train_and_install_centroids(table, partitions, ...)` + `centroids(table)` — IVF admin surface
- `client.vector.rebalance_status(table)` — IVF rebalance status

#### Graph
- `client.graph.node2vec_topk(schema, rel, query_pk, k, metric)` — over persisted Node2Vec embeddings
- `client.graph.graphsage(schema, feature_col, rel, config)` — train + optional persist
- `client.graph.graphsage_topk(schema, rel, query_pk, k, metric)` — over persisted GraphSAGE embeddings

#### SQL
- `client.sql.install_materialized_view(name, query, refresh_mode)`
- `client.sql.refresh_materialized_view(name)` → `{ rows_materialized, bytes_written, refresh_ts }`
- `client.sql.read_materialized_view(name)`

#### Admin
- `client.admin.install_tenant_config(tenant_id, replication_mode)`
- `client.admin.get_tenant_config(tenant_id)`

#### FTS (already had install_synonyms / install_stopwords; no API change — behavior upgraded server-side)
- Lemmatization is automatic when the table's analyzer config has `lemmatizer="dictionary"`; 9 languages now supported (was 0 in v0.4)

### Engine compatibility
- Requires engine 1.x build that contains the IVF-PQ + GraphSAGE + materialized views + Raft Phase D commits (any deploy after 2026-06-08 commit `2c1fe55a`)

### Migration from 0.4.0
- No breaking changes; all new methods are additive
- Existing `client.vector.delete` was wire-ready in 0.4 but errored at runtime because the handler didn't exist; now works end-to-end against engine 1.x
- `client.vector.install_centroids` URL fixed from `install_centroids` → `install-centroids` to match the engine's admin-route convention. 0.4 callers were hitting 404 against the deployed engine; 0.5 is the first version that actually reaches the handler.

### Changed
- Default `User-Agent` bumped to `originchain-python/0.5.0` (was `0.4.0` on sync, `0.3.0` on async — both now aligned).

## 0.3.0

### Changed
- **`vector_topk` `mode` parameter is now `"fast" | "high_recall"`** -
  replaces the previous `"hnsw" | "bruteforce"` value space. The
  parameter is now optional (`mode=None` by default) and is omitted
  from the request body when unset; the server defaults to
  `"high_recall"` when the field is absent. `"fast"` favours latency,
  `"high_recall"` favours recall.
- Default `User-Agent` bumped to `originchain-python/0.3.0`.

## Unreleased

### Added
- **Typed methods for `/sql`, `/vector/*`, `/fts/*`, `/graph/*`.** Customers
  no longer hand-roll dicts and parse JSON manually for the four
  substrate-extension surfaces. New methods on both `OriginChain` and
  `AsyncOriginChain`:
  - `client.sql(query)` and `client.sql_one(query)` - return a
    `SqlSelect` / `SqlInsert` / `SqlDelete` discriminated union.
  - `client.vector_put(table, *, id, embedding, dim, metric, metadata)`
    and `client.vector_topk(table, *, query, k, dim, metric, filter,
    mode)` - return `list[VectorHit]`.
  - `client.fts_index(table, field, *, doc_id, text)` and
    `client.fts_search(table, field, *, q, mode, k)` - return
    `list[FtsHit]`. Boolean / phrase / BM25 modes share one shape.
  - `client.graph.{neighbors, reverse_neighbors, bfs, path, dijkstra}` -
    return `list[Neighbor]`, `list[GraphBfsHit]`, `GraphPath`, and
    `DijkstraResult` respectively.
- **Frozen-dataclass response models** - `SqlSelect`, `SqlInsert`,
  `SqlDelete`, `VectorHit`, `FtsHit`, `Neighbor`, `GraphBfsHit`,
  `GraphPath`, `DijkstraResult` - all hashable + immutable, snake_case
  field names matching the wire.
- **`OCPaymentRequiredError`** - 402 add-on-required mapping. Surfaces
  `addon` / `name` / `monthly_usd` / `preview` / `enterprise_only` /
  `purchase_url` / `msg` as attributes.
- **Tests** under `sdk/python/tests/` using `httpx.MockTransport`. One
  test per method on the sync client; one e2e test gated behind
  `OC_E2E_TEST=1` against `localhost:8080`.

### Changed
- Default `User-Agent` bumped to `originchain-python/0.2.0`.
