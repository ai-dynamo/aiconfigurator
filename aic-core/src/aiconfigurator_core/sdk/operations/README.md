# Operations package

This package holds every op class the SDK queries when it estimates the
performance of a model on a given system / backend / version.

Each op class owns three things in the same module:

1. **A loader** (`load_<name>_data`) that parses one or more CSV files
   into a nested-dict perf table.
2. **A class** (subclass of `Operation`) that holds the per-op
   class-level cache, the `load_data` classmethod, and the
   `_query_*_table` classmethod implementing the lookup / interpolation
   logic.
3. **A query entry point** (`PerfDatabase.query_<name>`) — a one-line
   forward in `perf_database.py`. Lives there for backward compat with
   external callers; the op class can be queried directly via
   `Op._query_*_table(database, ...)` too.

**Lazy per-op data ownership** is the design rule the rest of this
document refers to. "Data + load + query live together" in the same op
module; the load is deferred to the first query. Two corollaries:

- `PerfDatabase()` opens zero CSV files. Data loads on first query
  (or on first read of `database.supported_quant_mode[<key>]`).
- Adding a new op never touches `PerfDatabase.__init__`.

## Adding a new op class

The pattern below covers a CSV-backed op. If your op computes
analytically from `database.system_spec` (no CSV — e.g. `ElementWise`,
`Embedding`), skip steps 2-4 and step 7; you don't need a loader, a
class-level cache, or a `load_data` classmethod, just a `query()` that
reads `database.system_spec` and a delegating `PerfDatabase.query_*`.

### 1. Pick a module

If your op belongs to an existing family (GEMM, attention, MLA, MoE,
DSA, DSV4, mamba, communication), add it to that file. Otherwise create
`operations/<your_op>.py`.

### 2. Write the CSV loader

```python
def load_my_op_data(my_op_file):
    """Parse my_op CSV into a nested dict keyed by [quant_mode][shape...].

    Returns None when the file is missing — the rest of the loader stack
    treats None as "no silicon data for this op". For an empty file
    (header but no rows), return an empty dict, not None.
    """
    rows = _read_filtered_rows(my_op_file)
    if rows is None:
        logger.debug(f"My-op data file {my_op_file} not found.")
        return None
    data = defaultdict(lambda: defaultdict(...))
    for row in rows:
        # parse the row; store per-shape leaves as {"latency": float, "power": float, "energy": float}
        ...
    return data
```

Imports needed at module top: `defaultdict` from `collections`,
`_read_filtered_rows` and `resolve_op_data_path` from
`aiconfigurator_core.sdk.operations.base`, `logger` from
`logging.getLogger(__name__)`, and `common` for quant-mode enums.

`_read_filtered_rows` accepts either a path string OR an iterable of
`(path, kernel_source_filter)` tuples (used by the shared-layer load
path inside `_build_op_sources`). You don't need to do anything
special — pass whatever your `load_data` classmethod receives from
`database._build_op_sources`.

### 3. Subclass `Operation`

```python
from typing import ClassVar
from aiconfigurator_core.sdk.operations.base import Operation

class MyOp(Operation):
    # Override the base Operation._data_cache with your own dict — the
    # cache is class-level, keyed by (systems_root, system, backend,
    # version, enable_shared_layer), and shared across all instances
    # of the same op class.
    _data_cache: ClassVar[dict] = {}

    def __init__(self, name: str, scale_factor: float, ...your-op-specific-args...):
        super().__init__(name, scale_factor)
        # store the per-instance state your ``query`` reads
```

### 4. Add the cache key + `load_data`

Most op modules define a module-level `_cache_key(database)`. Use it:

```python
def _cache_key(database):
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )

class MyOp(Operation):
    ...

    @classmethod
    def _cache_key(cls, database):
        return _cache_key(database)

    @classmethod
    def load_data(cls, database):
        """Idempotent. On cache miss: load CSV, populate class cache.
        Always: bind the instance attribute the legacy query body reads."""
        import os
        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.my_op.value
            )
            sources = database._build_op_sources(PerfDataFilename.my_op, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_my_op_data(sources), PerfDataFilename.my_op, primary_path
            )
            cls._record_load()

        # Respect intentional test overrides — only bind if the test
        # didn't already set the attribute.
        if "_my_op_data" not in database.__dict__:
            database._my_op_data = cls._data_cache[key]
```

Four things to notice:

- The CSV loader is referenced directly (module-local name). Don't
  import it from `perf_database.py`; that path would capture
  `perf_database`'s re-export and defeat conftest patches.
- `LoadedOpData` and `PerfDataFilename` come from `perf_database.py` —
  import them function-locally to avoid a module-level circular
  import.
- The instance-attribute bind is gated on `database.__dict__`, not
  `hasattr(...)`. Tests that pre-set `db._my_op_data = ...` rely on
  this gate.
- Per-op paths must always be resolved through `resolve_op_data_path`,
  never a bare `os.path.join(system_data_root, database.backend,
  database.version, ...)` — the family-first layout may place this
  op's table under a sibling family dir instead of a flat
  `<backend>/<version>` dir.

### 5. Add the lookup logic

```python
class MyOp(Operation):
    ...

    @classmethod
    def _query_my_op_table(cls, database, *args, database_mode=None):
        cls.load_data(database)

        def get_sol():
            # analytical formula — derive from ``database.system_spec``
            ...

        def get_empirical():
            return get_sol() / database.system_spec["gpu"]["mem_bw_empirical_scaling_factor"]

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return database._interp_pr(get_sol()[0])
        return database._query_silicon_or_hybrid(
            get_silicon=lambda: ...your lookup against database._my_op_data...,
            get_empirical=get_empirical,
            database_mode=database_mode,
            error_msg="Failed to query my_op data ...",
        )
```

Notes:

- `cls.load_data(database)` at the top of every `_query_*_table` is
  what makes the lazy contract work. Forget it and your op silently
  loads via a side path (the `_LazySupportMatrix` resolver) or fails
  with `AttributeError: 'PerfDatabase' object has no attribute
  '_my_op_data'`.
- Interpolation / extrapolation helpers live in
  `aiconfigurator_core.sdk.interpolation`. Call them directly.
  `interp_2d_linear` and `interp_3d` require `database._extracted_metrics_cache`
  as the last positional argument.

### 6. Wire the query entry point

In `perf_database.py`:

```python
@functools.lru_cache(maxsize=32768)
def query_my_op(self, *args, database_mode=None):
    """Delegates to MyOp; see operations.<module>.MyOp._query_my_op_table."""
    from aiconfigurator_core.sdk.operations.<module> import MyOp
    return MyOp._query_my_op_table(self, *args, database_mode=database_mode)
```

### 7. Wire the support matrix (optional)

If your op contributes a quant-mode dimension to support-matrix output,
add a resolver in
`perf_database.py:_LazySupportMatrix._resolve` and add your key name
to the relevant entries in `_BACKEND_KEYS`. Skip this if your op's
quant modes are already covered by another key.

### 8. Export the class

Add to `operations/__init__.py`:

```python
from aiconfigurator_core.sdk.operations.<module> import MyOp
__all__ = [..., "MyOp", ...]
```

### 9. Write tests

Three layers, in roughly this order:

- **Unit tests** in `tests/unit/sdk/database/test_<your_op>.py`. Cover
  the loader (parses CSV correctly), the SOL formula (matches hand
  math), the silicon lookup path (returns expected interpolated value
  on a small synthetic table). Use the existing `stub_perf_db` or
  `comprehensive_perf_db` fixtures.
- **Conftest hookup**: add your loader name to `_LOADER_STUBS` in
  `tests/unit/sdk/database/conftest.py` so the standard fixtures
  patch your loader at the right module path. Add an `override` for
  `comprehensive_perf_db` if your tests need realistic data shape.
- **Regression snapshot** in `~/forks/sdk-refactor-regression/` if
  your op is used by an existing model — the snapshot will catch
  perf-number drift the unit tests miss.

## Conventions to follow

### Idempotency

`load_data` must be safe to call repeatedly. The cache-key check at the
top is the contract: same `(systems_root, system, backend, version,
enable_shared_layer)` → same result, regardless of how many times it
fires.

### Instance attribute bind gating

Use `if "_my_op_data" not in database.__dict__:` rather than
`hasattr(database, "_my_op_data")` for the post-load bind. Tests
sometimes pre-set the attribute to inject custom data; `hasattr`
would silently overwrite the override.

### Cache atomicity for ops with multiple slots

If your op owns N caches (GEMM has 3 — gemm / compute_scale /
scale_matrix; MoE has 4; ContextDeepSeekV4AttentionModule has 3), load
all of them into local variables FIRST, then commit them all to the
class cache as the last step. A loader exception mid-sequence would
otherwise leave the class cache with the first slot populated but the
others empty, and the next `load_data` cache-hit would skip the load
and crash downstream.

```python
# Good
gemm_loaded = _load(...)
compute_scale_loaded = _load(...)
scale_matrix_loaded = _load(...)
# all three succeeded — commit atomically
cls._data_cache[key] = gemm_loaded
cls._compute_scale_cache[key] = compute_scale_loaded
cls._scale_matrix_cache[key] = scale_matrix_loaded
```

### Source attribution

Every `PerformanceResult` carries a `source` tag —
`"silicon"` / `"empirical"` / `"mixed"`. Use
`database._interp_pr(latency, energy=...)` (sets `"silicon"`),
`PerformanceResult(latency, energy=..., source="empirical")` for SOL
fallbacks, and the arithmetic-preserving constructor when combining.
Don't blindly overwrite an existing `PerformanceResult.source`.

### Don't fight the lazy contract

A few things that look reasonable but break the lazy invariant:

- Eager `load_data` calls in `PerfDatabase.__init__`. The whole point
  of lazy per-op data ownership is that init opens no CSVs.
- Reading `database._my_op_data` without first calling
  `cls.load_data(database)`. The instance attr only exists after the
  first load.
- Calling `clear_all_op_caches()` mid-test if your test reuses the
  `comprehensive_perf_db` singleton. Wiping the class cache forces a
  fresh-disk reload with no loader patches active. Use
  `_warm_lazy_op_caches` in conftest for per-test isolation instead;
  see `test_load_data_counts.py` for the save/restore pattern when a
  test specifically needs to exercise `clear_all_op_caches`.

## Pointers

- `operations/base.py` — `Operation` base class, the `_data_cache`
  contract, the `_load_data_call_count` instrumentation,
  `clear_all_op_caches()`, and `_read_filtered_rows`.
- `operations/gemm.py` — the cleanest fully-migrated example. Read
  `GEMM.load_data` and `GEMM._query_gemm_table` for the canonical
  pattern.
- `aiconfigurator_core/sdk/interpolation.py` — the numeric helpers
  (`interp_1d`, `interp_3d`, `nearest_1d_point_helper`,
  `validate_interpolation_result`, …). 3-axis+ op queries route through
  `aiconfigurator_core/sdk/perf_interp/` (declarative per-op configs + shared
  resolver); interpolation.py remains for 1-D/2-D tables (comm, bmm, scales).
- `aiconfigurator_core/sdk/perf_database.py` — `PerfDatabase`,
  `_LazySupportMatrix` (the dict-like view powering
  `database.supported_quant_mode`), `LoadedOpData`, the query entry
  points.

## Why this design

The design problem statement was:

1. **Extensibility** — adding a new op or model shouldn't require
   touching a 600-line `__init__` wiring block.
2. **Isolation** — changing one op shouldn't risk breaking unrelated
   ops.
3. **Startup performance** — only load the CSV data the model under
   evaluation actually needs.

Lazy per-op data ownership solves all three in one move: each op owns
its data + load + query, the class-level cache makes loading
idempotent and shared across instances, and the `_LazySupportMatrix`
defers loading until something actually reads the support matrix.
