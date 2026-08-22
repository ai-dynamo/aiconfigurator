# PROPOSAL: facts-gated upgrades, and facts-lookup instead of rules

Status: draft for owner approval. Rule files are human-owned policy
(`.claude/rules/` charter); this document PROPOSES the changes below — it does
not enact them.

## 1. The gate: no facts, no upgrade

Add to `docs/perf_database/collector-upgrade-playbook.md` as step 0 of every
framework version bump:

> **Step 0 — generate facts on both versions.** Run the `collector/facts`
> sweep against the old pinned image and the new one (same targets, same
> platform). The facts diff — resolved config fields, per-module quant
> methods, API call chains, kernel sets — IS the upgrade work list. No
> collector file may be edited for a version bump before this diff exists.

And amend `.claude/rules/collector/layer_permissions.md`:

> *(current)* "manual pinning requires source proof … citing the framework
> source (file:line at the pinned version)"
>
> *(proposed)* "manual pinning requires a **facts record id** — a probe run
> that executed the pinned framework and observed the claimed behavior.
> Reading framework source is for understanding mechanisms, never for
> establishing facts; every load-bearing claim (default backend, kernel
> identity, quant binding, flag requirement) must cite
> `archive/records.jsonl` provenance (run id + image digest)."

Acceptance for an upgraded collector op becomes three machine-checked
equalities (see `check_facts.py`):

1. the op's quant identity matches the facts record (A),
2. the API it drives is the one serving executes (B),
3. its emitted `kernel_source` translates (via `kernel_taxonomy.yaml` +
   `kernel_source_backends.yaml`) to the kernels the probe observed (C).

"Runs green" is necessary, never sufficient.

## 2. The stance: AIC does not carry rules, it queries facts

Special-case knowledge keeps accumulating (topology-dependent flag
requirements, per-family backend forks, block-size constraints, KV-dtype
support holes). Encoding these as code rules reproduces the retired
`sm_exceptions` failure (PR #1302 deleted a 1400-line rule engine;
"observe-don't-predict"). The maintainable shape:

- **Facts layer** — probe-generated records keyed by
  `(model, quant, topology, sm, framework, version)`. New knowledge lands as
  ROWS with provenance, not as `if` branches. Rows cannot conflict with each
  other; a wrong row is deleted and re-probed.
- **Consumption layer** — every place AIC would write a rule becomes a
  lookup. The generator, after rendering, asks: has this combination been
  probed? did it boot? does it need `required_flags`
  (e.g. NVFP4+SM90+sglang -> `moe_runner_backend=marlin`)? The support
  matrix is the projection of `status=ok` rows, not a hand-maintained table.
- **Fallback layer** — an unprobed combination is answered
  "unknown — probe first" (a dummy-model boot probe is minutes), never
  guessed. AIC degrades from an all-knowing rule engine to an experiment
  system with memory — which is the maintainable thing to be.

Complexity does not disappear; it moves from code (superlinear maintenance,
rules interact) to data (linear maintenance, refreshed wholesale by rerunning
the sweep on every version bump).

## 3. Generator boot gate (CI proposal)

The same probe doubles as the validation loop the generator lacks. Observed
generator gaps, each reproduced by the probe:

- `tokens_per_block` is an unvalidated passthrough; the FPM example's 64
  breaks MiniMax-M3 AND DeepSeek-V4 on vLLM.
- fp8-profile GLM + vLLM + SM90 renders a deployment with no usable
  attention backend.
- FPM rendering silently falls back to the dynamo target when its
  preconditions fail; template version resolution silently uses older
  templates for newer engines.

Proposed gate: for each (model, profile, backend) the generator claims to
support, render + dummy-boot-probe in CI (minutes per combination, shared
JIT cache). Render-time silent fallbacks become loud CI diffs.

## 4. Record schema field additions

- `required_flags`: flags without which the combination refuses to boot,
  with the rejection record as evidence.
- `known_bad`: combination confirmed non-bootable (with error class).
- `evidence`: real | mocked | unverified — replay on each platform promotes.
