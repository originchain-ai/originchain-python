# Changelog

All notable changes to the OriginChain Python SDK. See the repo-root
`CHANGELOG.md` for engine releases.

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
