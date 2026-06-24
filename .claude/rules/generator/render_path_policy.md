# Render-Path Policy Reference

Which artifacts render via Jinja templates vs. typed Python builders, where
defaults live, and the two-path divergence to watch for. Read before adding a
backend version, a new artifact, or a new default.

## Artifact → render mechanism

| Artifact | Mechanism | Source of truth |
|---|---|---|
| `k8s_deploy.yaml` (DynamoGraphDeployment) | **typed builder** | `builders/k8s_builder.py` + `builders/dgd_model.py`, fed by `resolved_facts` |
| trtllm `extra_engine_args*.yaml` | Jinja | `config/backend_templates/trtllm/extra_engine_args.<ver>.yaml.j2` |
| `cli_args` (all backends) | Jinja | `config/backend_templates/<backend>/cli_args[.<ver>].j2` |
| `run.sh`, `llm-d-values.yaml`, sflow, benchmark | Jinja | `config/backend_templates/...` |

## Policy

1. **Typed builders are reserved for structured / multi-document artifacts**
   (the k8s DGD: nested YAML, conditional EFA resources, env-merge precedence).
   **Do NOT introduce a typed builder for flat-text artifacts** (cli_args,
   run.sh, engine args) — Jinja is simpler there and a builder only adds a
   transliteration plus a second source of truth.

2. **The engine-config typed-builder experiment was reverted.** There is no
   `builders/trtllm_engine_config.py`, no `manifests/`, and no `contract/`
   loader. trtllm engine configs render from versioned Jinja templates (floor
   match). Do not resurrect the builder/manifest mechanism; add a new version
   template instead (see `new_backend_version.md`).

3. **Adding a new trtllm engine version = add a Jinja template**, not a builder
   or manifest. Copy the closest prior `extra_engine_args.<ver>.yaml.j2`, edit,
   never touch prior versions.

## Where defaults live (do not split across layers)

- **Per-worker engine defaults** (max_batch_size, tokens_per_block,
  skip_tokenizer_init value, etc.) live in `rule_plugin/<backend>.rule`. The
  rule layer is the single home for per-worker defaults.
- **`backend_config_mapping.yaml` is pure translation**: `param_key` → backend
  flag name, plus value-transforms that *shape a CLI flag value* (e.g. the
  vLLM/SGLang `kv-cache-dtype` `bfloat16 -> auto` transform, which must be a
  mapping transform because it targets the emitted CLI flag).
- **Do not add new `default:` retention to the mapping.** A mapping `default:`
  is consumed only by `render-config` (see divergence below), so it advertises
  flags that are never deployed. Put the default in the rule layer.

## Guard: kv-dtype translation lives in the rule, per backend

The `float16/bfloat16 -> auto` kv-cache-dtype guard (CRASH-severity) is split by
where each backend emits the value, and that split is intentional:

- **trtllm**: in `rule_plugin/trtllm.rule` (trtllm emits dtype in the engine
  YAML, which the mapping value-transform cannot reach). Keep it in the rule.
- **vLLM / SGLang**: in `backend_config_mapping.yaml` value transforms (they emit
  it as a CLI flag).

Do not move the trtllm translation out of the rule.

## The two render paths can diverge — know which one you're touching

There are two parameter-resolution paths and they do not resolve defaults
identically:

- **`render-config`** (`aic-generator render-config` → `render_parameters`):
  applies `backend_config_mapping.yaml` value/`default:` retention. Inspection
  only.
- **Artifact generation** (`generate_backend_artifacts` → `make_worker_context`
  / `prepare_template_context`): only emits flags for params actually present in
  `params[<role>]` (set by rules / facts / explicit input). It does **not**
  apply mapping `default:` retention.

Consequence: a mapping `default:` shows up in `render-config` output but **not**
in the deployed artifact. When reasoning about "what gets deployed," trace the
artifact path, not `render-config`. When adding a default that must be deployed,
put it in the rule layer so it lands in `params[<role>]`.

## Checklist

```text
[ ] New flat-text artifact/flag -> Jinja template + (if computed) rule, never a builder
[ ] New per-worker default -> rule_plugin/<backend>.rule, not a mapping default:
[ ] New trtllm engine version -> copy closest extra_engine_args.<ver>.yaml.j2
[ ] kv-dtype-style guards: trtllm in .rule, vllm/sglang in mapping transform
[ ] Verifying "what deploys" -> check the artifact path, not render-config
```
