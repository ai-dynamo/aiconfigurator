# Rule Plugin Authoring Reference

The rule plugin system (`.rule` files) computes derived parameters using a custom DSL.
This reference covers safe practices for writing and modifying rules.

## When to Use This Reference

- Adding a new computed parameter to the generator
- Modifying parameter computation logic (batch size, parallelism, CUDA graphs)
- Adding MoE, speculative decoding, or prefix caching support
- Creating a new rule mode (like `benchmark/`)

## Key Files

| File | Purpose |
|---|---|
| `src/aiconfigurator/generator/rule_plugin/*.rule` | Production rules per backend |
| `src/aiconfigurator/generator/rule_plugin/benchmark/*.rule` | Benchmark rules per backend |
| `src/aiconfigurator/generator/rendering/rule_engine.py` | Rule DSL evaluation engine |
| `src/aiconfigurator/generator/config/backend_config_mapping.yaml` | Parameter backend support |

## DSL Reference

```text
# === SCOPE PREFIXES ===
# Apply to specific worker roles:
prefill <key> = <expr>              # Only prefill workers
decode <key> = <expr>               # Only decode workers
agg <key> = <expr>                  # Only aggregated (single) workers
agg_decode <key> = <expr>           # Both agg and decode workers
prefill_decode <key> = <expr>       # Both prefill and decode workers
agg_prefill_decode <key> = <expr>   # ALL worker roles

# === ASSIGNMENT ===
<scope> <key> = <expression>
# Expression uses Jinja2 syntax, evaluated via compile_expression()

# === CONDITIONALS ===
when <condition>:
    <scope> <key> = <expr>
    <scope> <key> = <expr>

# === CONFIG GROUP TARGETING (global, not per-worker) ===
DynConfig.enable_router = true
BenchConfig.estimated_concurrency = <expr>

# === AVAILABLE CONTEXT VARIABLES ===
# From deployment_config.yaml:  ServiceConfig.*, K8sConfig.*, ModelConfig.*, SlaConfig.*
# From params (per-worker):     tensor_parallel_size, max_batch_size, max_num_tokens, etc.
# Special shorthand:            isl, osl, bs, is_moe
```

## Workflow

### 1. Understand the Parameter

- What is the parameter's purpose?
- Which backends support it? (check `backend_config_mapping.yaml`)
- Which worker roles need it? (prefill, decode, agg, or all?)
- Does it depend on other computed parameters?

### 2. Check Existing Rules

- Read ALL `.rule` files: `trtllm.rule`, `vllm.rule`, `sglang.rule`
- Check `benchmark/` variants too
- Identify if the parameter is already computed elsewhere
- Identify dependencies (e.g., `max_num_tokens` depends on `max_batch_size`)

### 3. Write the Rule

- Use the correct scope prefix (most common mistake!)
- Guard nullable values: `(value or 0)`, `(value if value else default)`
- Place new rules AFTER their dependencies in the file
- Use Jinja2 expression syntax (not Python -- no list comprehensions, no f-strings)

### 4. Cross-Backend Consistency

For each backend's `.rule` file:

1. Does this backend support the parameter? (check mapping yaml)
2. If yes, does the rule exist? If not, add it.
3. If the computation differs by backend, document WHY.
4. If the parameter is null for this backend in mapping, skip it.

### 5. Test

- Write a test case that exercises the rule with known inputs
- Test edge cases: MoE models, disagg mode, nullable params
- Verify rule ordering: if rule B depends on rule A's output, A must come first

## Common Patterns

```python
# Pattern: Scale batch size for production (agg + decode get bigger batches)
agg_decode max_batch_size = (512 if (max_batch_size or 0) < 512 else (max_batch_size * 2))
prefill max_batch_size = (max_batch_size if max_batch_size else 1)

# Pattern: MoE parallelism remapping (TP = moe_tp * moe_ep)
when ModelConfig.is_moe and (moe_tensor_parallel_size and moe_expert_parallel_size):
    agg_prefill_decode tensor_parallel_size = moe_tensor_parallel_size * moe_expert_parallel_size

# Pattern: Feature toggle based on model config
when (ModelConfig.prefix or 0) > 0:
    agg_prefill_decode disable_prefix_cache = false
    DynConfig.enable_router = true

# Pattern: Token budget alignment to block size
agg max_num_tokens = ((max_batch_size + SlaConfig.isl + 1500 + tokens_per_block - 1) // tokens_per_block) * tokens_per_block
```

## Gotchas

1. **`agg_prefill_decode` means ALL roles**, not a special combined role. Use it when
   a rule applies universally.

2. **Rule ordering matters** -- Rules are evaluated top-to-bottom. If rule B uses
   `max_batch_size` that rule A modifies, A must appear before B.

3. **Nullable guards are mandatory** -- `SlaConfig.isl` can be None. Always use
   `(SlaConfig.isl or 0)` in arithmetic.

4. **`when` blocks don't nest** -- There's no `when` inside `when`. Combine conditions:
   `when A and B:`.

5. **Backend-specific parameters** -- Some params only exist for one backend (e.g.,
   `moe_dense_tp_size` was SGLang-only but removed as invalid per PR #613). Check
   mapping yaml before adding.

6. **Config group targeting is global** -- `DynConfig.enable_router = true` applies to
   the entire deployment, not per-worker.

## Checklist

```text
[ ] Identify parameter purpose, backend support, and worker scope
[ ] Read all existing .rule files (production + benchmark)
[ ] Write rule with correct scope prefix and null guards
[ ] Verify rule ordering (dependencies first)
[ ] Add equivalent rule to all applicable backend .rule files
[ ] Document intentional cross-backend differences
[ ] Add test case covering the new rule
[ ] Test with MoE and disagg scenarios if applicable
```
