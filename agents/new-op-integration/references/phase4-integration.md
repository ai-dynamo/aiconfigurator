# Phase 4: SDK Integration

## Goal

Integrate a new operation into the aiconfigurator modeling stack so it can participate in end-to-end latency estimation.

## Files to Modify

| # | File | Change |
|---|------|--------|
| 1 | `src/aiconfigurator/sdk/common.py` | Add `PerfDataFilename` enum entries |
| 2 | `src/aiconfigurator/sdk/perf_database.py` | Add data loading and query logic |
| 3 | `src/aiconfigurator/sdk/operations.py` | Add operation subclasses |
| 4 | `src/aiconfigurator/sdk/models.py` | Use new operations in model op lists |

## Step 1: Add `PerfDataFilename`

**File**: `src/aiconfigurator/sdk/common.py`

```python
class PerfDataFilename(Enum):
    gemm = "gemm_perf.txt"
    # ... existing files ...
    mamba2 = "mamba2_perf.txt"
    new_op_context = "new_op_context_perf.txt"        # add
    new_op_generation = "new_op_generation_perf.txt"  # add if split by phase
```

## Step 2: Add Data Loading and Query

**File**: `src/aiconfigurator/sdk/perf_database.py`

### 2.0) Apply axis decision from Phase 2

At this stage, do not redefine modeling axes. Reuse the axis decision made in Phase 2 mock-layer design.

Use that Phase 2 decision to finalize:
1. Query signature (`x` only vs separate `b`,`s`)
2. Data schema in loader/interpolation
3. Operation kwargs contract (`operations.py`)

If Phase 2 marked the op as token-equivalent, query can be `x`-centric.
If Phase 2 marked the op as token-non-equivalent, keep explicit `b` and `s`.

### 2a) Data loader

Use `load_context_mla_data()` / `load_mamba2_data()` as templates.

```python
def load_new_op_data(new_op_file):
    """
    Load new-op performance data from new_op_perf.txt.

    CSV columns:
      framework, version, device, op_name, kernel_source,
      param1, param2, batch_size, seq_len, latency [, power]
    """
    if not os.path.exists(new_op_file):
        logger.warning("New-op data file %s not found.", new_op_file)
        return None

    # data[param1][param2][batch_size][seq_len] = {"latency": ..., "power": ..., "energy": ...}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    with open(new_op_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy format in %s, power defaults to 0.0", new_op_file)

    for row in rows:
        param1 = int(row["param1"])
        param2 = int(row["param2"])
        b = int(row["batch_size"])
        s = int(row["seq_len"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0)) if has_power else 0.0
        energy = power * latency  # W*ms

        data[param1][param2][b][s] = {
            "latency": latency,
            "power": power,
            "energy": energy,
        }

    return data
```

### 2b) Load in `PerfDatabase.__init__`

```python
self._new_op_data = load_new_op_data(
    os.path.join(data_dir, common.PerfDataFilename.new_op_context.value)
)
```

### 2c) Query method with full database-mode support

```python
def query_new_op(
    self,
    b: int,
    s: int,
    param1: int,
    param2: int,
    database_mode: common.DatabaseMode | None = None,
) -> PerformanceResult | tuple[float, float, float]:
    """Query new-op latency/energy."""

    def get_sol(b: int, s: int, param1: int, param2: int) -> tuple[float, float, float]:
        # Example SOL model
        flops = b * s * param1 * param2 * 2
        sol_math = flops / self.system_spec["gpu"]["float16_tc_flops"] * 1000

        mem_bytes = b * s * (param1 + param2) * 2
        sol_mem = mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000
        return max(sol_math, sol_mem), sol_math, sol_mem

    def get_empirical(b: int, s: int, param1: int, param2: int) -> float:
        return get_sol(b, s, param1, param2)[0] / 0.7

    if database_mode is None:
        database_mode = self._default_database_mode

    if database_mode == common.DatabaseMode.SOL:
        return PerformanceResult(get_sol(b, s, param1, param2)[0], energy=0.0)
    elif database_mode == common.DatabaseMode.SOL_FULL:
        return get_sol(b, s, param1, param2)
    elif database_mode == common.DatabaseMode.EMPIRICAL:
        return PerformanceResult(get_empirical(b, s, param1, param2), energy=0.0)
    else:  # SILICON or HYBRID
        try:
            if self._new_op_data is None:
                raise PerfDataNotAvailableError("New-op data not loaded.")

            # After fixing param1, interpolate on param2 -> b -> s
            result = self._interp_3d(param2, b, s, self._new_op_data[param1], "cubic")
            return PerformanceResult(result["latency"], energy=result.get("energy", 0.0))
        except Exception:
            if database_mode == common.DatabaseMode.HYBRID:
                return PerformanceResult(get_empirical(b, s, param1, param2), energy=0.0)
            raise
```

If your op is token-equivalent, use `x` directly in query signature instead:

```python
def query_new_op(self, x: int, param1: int, ..., database_mode=None):
    # SOL and empirical are functions of x
    ...
```

### 2d) Interpolation support

At the end of `PerfDatabase.__init__()`, add pre-interpolation for unseen points:

```python
if self._new_op_data is not None:
    for param1 in self._new_op_data:
        data_dict = self._new_op_data[param1]
        target_x_list = [...]  # param2 targets
        target_y_list = [...]  # batch_size targets
        target_z_list = [...]  # seq_len targets

        # Fixed param1, then x=param2, y=batch_size, z=seq_len
        self._new_op_data[param1] = interp_3d(
            data_dict, target_x_list, target_y_list, target_z_list
        )
```

## Step 3: Add Operation Subclasses

**File**: `src/aiconfigurator/sdk/operations.py`

Do not use `@dataclass`; follow existing class-style operation definitions.

Keep `Operation.query()` kwargs consistent with axis choice:
- token-equivalent op: pass `x`
- token-non-equivalent op: pass both `b` and `s` (or enough inputs to reconstruct both safely)

```python
class ContextNewOp(Operation):
    def __init__(self, name: str, scale_factor: float, param1: int, param2: int) -> None:
        super().__init__(name, scale_factor)
        self._param1 = param1
        self._param2 = param2
        self._weights = param1 * param2 * 2  # FP16 bytes

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        x = kwargs.get("x")
        result = database.query_new_op(
            b=kwargs.get("batch_size", 1),
            s=x,
            param1=self._param1,
            param2=self._param2,
        )
        return PerformanceResult(
            latency=float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GenerationNewOp(Operation):
    def __init__(self, name: str, scale_factor: float, param1: int, param2: int) -> None:
        super().__init__(name, scale_factor)
        self._param1 = param1
        self._param2 = param2
        self._weights = param1 * param2 * 2

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        result = database.query_new_op(
            b=kwargs.get("x"),
            s=kwargs.get("s"),
            param1=self._param1,
            param2=self._param2,
        )
        return PerformanceResult(
            latency=float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
```

## Step 4: Use New Ops in `models.py`

**File**: `src/aiconfigurator/sdk/models.py`

Add operations to `self.context_ops` / `self.generation_ops` following `LLAMAModel` or `DeepSeekModel` patterns:

```python
self.context_ops.extend([
    # ... standard ops ...
    ops.ContextNewOp(
        "context_new_op",
        self._num_layers,  # scale_factor
        param1=self._param1,
        param2=self._param2,
    ),
    # ... standard ops ...
])

self.generation_ops.extend([
    # ... standard ops ...
    ops.GenerationNewOp(
        "generation_new_op",
        self._num_layers,
        param1=self._param1,
        param2=self._param2,
    ),
    # ... standard ops ...
])
```

## Step 5: Place Performance Data Files

Put collected files under:

```
src/aiconfigurator/systems/data/{system_name}/{backend}/{version}/
  |- new_op_context_perf.txt
  |- new_op_generation_perf.txt
```

## CLI Validation

```bash
aiconfigurator cli support --model-path <model> --system h200_sxm --backend trtllm

aiconfigurator cli default --model-path <model> --total-gpus 8 --system h200_sxm \
  --isl 4000 --osl 1000 --ttft 2000 --tpot 30
```

## Common Issues

| Issue | Cause | Fix |
|------|------|------|
| `query` returns 0 | Data not loaded or wrong file path | Check `PerfDataFilename` and init loading path |
| `KeyError` in query | Parameter combo missing | Add interpolation or collect more data |
| Large model-vs-real gap | Wrong op decomposition | Re-check Phase 3 kernel/latency alignment |
| `_interp_3d` returns dict | Interpolation result is not a float | Wrap with `PerformanceResult(result["latency"], energy=result.get("energy", 0.0))` |
| Empty dict leaves after loading | `defaultdict(dict)` at leaf level | Use `defaultdict()` leaf or explicit key checks |
| Wrong interpolation axes | Dict nesting does not match `_interp_3d(x, y, z)` | Keep axis order aligned with key order |

## `perf_database.py` Rules

Use the closest existing op as the exact template:

1. **Loader shape**: dict nesting must match existing patterns exactly
2. **Interpolation registration**: add to `__init__` interpolation block
3. **Query behavior**: return `PerformanceResult`, avoid cross-granularity fallback
4. **Quantization keys**: only register supported quant modes; override unsupported ones in model logic

## Query Must Support Full Database Modes

Each query method must handle all modes: `SOL`, `SOL_FULL`, `EMPIRICAL`, `SILICON`, `HYBRID`.

```python
if database_mode == DatabaseMode.SOL:
    ...
elif database_mode == DatabaseMode.SOL_FULL:
    ...
elif database_mode == DatabaseMode.EMPIRICAL:
    ...
else:  # SILICON or HYBRID
    try:
        ...
    except Exception:
        if database_mode == DatabaseMode.HYBRID:
            ...
        raise
```

## SOL Derivation Notes

1. Break op into sub-computations and compute FLOPs per part
2. Compute bound: `total_flops / gpu_peak_flops * 1000`
3. Memory bound: `total_bytes / gpu_mem_bw * 1000`
4. `SOL = max(compute, memory)`
5. Empirical mode typically uses `SOL / scale_factor`

## Context vs Generation SOL Differences

| Item | Context | Generation |
|------|---------|------------|
| Tokens | `b * s` | `b` |
| Typical bottleneck | Usually compute-bound | Usually memory-bound |
| Attention term | `O(tokens * s)` or `O(tokens * topk)` | `O(b * s)` or `O(b * topk)` |
| GEMM behavior | Larger M, better compute efficiency | Smaller M (`=b`), often memory-dominated |
