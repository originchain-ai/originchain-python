# Changelog

All notable changes to the OriginChain Python SDK. See the repo-root
`CHANGELOG.md` for engine releases.

## [0.7.0] — 2026-09-25

Faster repeat calls from applications that call the API every few seconds.

### Changed

- **Idle connections are kept for 300 s** (was httpx's default of 5 s).
  An application calling every few seconds used to re-open TCP + TLS on
  most calls — two extra network round trips each time, ~45-50 ms per call
  from a client one region away. The server never closes an idle keep-alive
  connection; the shortest idle limit on the path is the network load
  balancer's 350 s, so the client stays below it.
- Pool sizes stay at httpx's defaults (100 connections, 20 kept alive),
  now passed explicitly.

### Added

- `keepalive_expiry=` on `OriginChain(...)` and `AsyncOriginChain(...)`
  (seconds; clamped to 0-340).
- Default `User-Agent` is `originchain-python/0.7.0`.

## [0.6.0] — 2026-07-29

Three wire-format bug fixes. All three surfaces were **broken in
0.5.0** — none ever reached the engine behaviour it advertised — so
nothing that works today changes behaviour.

### Fixed

- **`vector.topk(nprobe=...)` now sends the field the engine reads.**
  0.5.0 sent `ivf_nprobe`; `VecTopkReq` declares `nprobe`. The struct
  has no `deny_unknown_fields`, so serde dropped the key silently and
  the query ran at the server default of `min(8, partitions)`. Unlike
  the two bugs below this one returned a confident **200** with
  quietly degraded recall rather than an error, which is why it went
  unnoticed. `vector.topk` also gains the `index` parameter
  (`"hnsw"` | `"ivf"` | `"ivf_pq"`): `nprobe` is only consulted on the
  IVF arms, so without it the knob was inert even under the right
  name.
- **`sql.query(params=...)` now sends positional bind parameters.**
  0.5.0 sent `params` as a named JSON object (`{"a": 1}`). The `/sql`
  handler binds positionally: `params` is a JSON **array** and `$1` /
  `$2` index into it. Every parameterised query failed. The parameter
  is now `Sequence[Any]` and reaches the wire verbatim as an array —
  the same shape the TypeScript client (`params?: unknown[]`) and the
  Go client (variadic `params ...any`) already send.
- **`sql.install_materialized_view(refresh_mode=...)` now sends live
  wire values.** 0.5.0 sent `"manual"` / `"on_write"`. The engine's
  `RefreshMode` enum is `#[serde(rename_all = "snake_case")]` over
  `OnDemand | Incremental`, and unknown values are a hard 400 — so
  every install failed. Accepted values are now `"on_demand"`
  (default) and `"incremental"`.

### Changed

- `refresh_mode` default is `"on_demand"` (was `"manual"`); same
  intended behaviour, correct wire value. `"manual"` / `"on_write"`
  still work as deprecated aliases for `"on_demand"` /
  `"incremental"` and emit a `DeprecationWarning`; they are removed in
  1.0. Any other value raises `OCValidationError` **before** the
  request is sent rather than round-tripping to a 400.
- `sql.query(params=...)` no longer accepts a `Mapping`. A `dict`
  raises `OCValidationError` naming the positional rewrite. Flattening
  a dict into a guessed order would bind values to placeholders the
  caller never wrote down, and the accompanying SQL would still carry
  named placeholders the engine can't parse — a clear error beats a
  silently-wrong-row bug. A bare `str` / `bytes` is rejected for the
  same reason (it would otherwise bind one parameter per character).

### Materialized-view refresh-mode semantics (documented, unchanged)

- `"on_demand"` — full recompute on an explicit
  `refresh_materialized_view()` call. The only mode to build against
  today.
- `"incremental"` — apply-time maintenance as source rows are written.
  **Aggregate** materialized views are behind a preview flag that
  ships OFF, so an incremental *aggregate* view currently returns
  `422`. Don't depend on it without confirming the flag for your
  instance.

### Migration from 0.5.0

- `db.vector.topk(t, q, nprobe=32)` →
  `db.vector.topk(t, q, nprobe=32, index="ivf")` (or `"ivf_pq"`).
  Without `index` the engine queries HNSW and ignores `nprobe`. Recall
  and latency will change on any call that was passing `nprobe`,
  because it now actually takes effect.
- `db.sql.query("... WHERE s = :s", params={"s": "AAPL"})` →
  `db.sql.query("... WHERE s = $1", params=["AAPL"])`
- `refresh_mode="manual"` → `refresh_mode="on_demand"`;
  `refresh_mode="on_write"` → `refresh_mode="incremental"`
- Default `User-Agent` is `originchain-python/0.6.0`

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
