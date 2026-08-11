# WideEP Collectors

WideEP collectors live under this namespace so tooling can choose the right
runtime image separately from the normal framework collectors.

Each supported framework owns a WideEP-only `registry.py`. Normal framework
registries stay free of WideEP ops; `collect.py` appends a WideEP registry only
when the collector-v2 plan or explicit `--ops` requests those ops.

The authoritative framework versions and collector images are in
`collector/framework_manifest.yaml`. WideEP entries describe their special
runtime independently from the non-WideEP framework entry.

Layout:

- `sglang/collect_deepep_moe.py`: SGLang DeepEP MoE entrypoint (op `moe_ep`).
- `sglang/collect_moe_a2a.py`: standalone multi-node DeepEP all-to-all
  collector (unified `moe_a2a_perf` table); launched by
  `collector/network/slurm/submit_moe_a2a.sh`.
- `sglang/deepep/`: DEPRECATED multi-node DeepEP log collection and
  extraction scripts (see below).
- `trtllm/collect_moe_compute.py`: TensorRT-LLM WideEP MoE compute entrypoint
  (op `moe_ep`; pins the same image as stock trtllm, so model plans activate
  it by default for wideep-declared models).
- `vllm/collect_moe_ep.py`: vLLM fused-experts moe_ep bench path, DORMANT —
  no registry, no manifest entry, no hash-closures entry until a vLLM-DeepEP
  image is pinned (plan decision D3; activation procedure below).

## The `sglang/deepep/` pipeline is deprecated

The manual pipeline — run the modified DeepEP test scripts on each node,
tee the stdout to log files, then scrape them with `extract_data.py` —
derived every identity column (framework, version, device) from module-level
constants and parsed `node_num` out of a log *filename*. Its replacement,
`sglang/collect_moe_a2a.py`, runs the same DeepEP HT and low-latency
benchmarks as a Collector-V3 citizen: identity columns are live (the
distributed world supplies `ep_size`/`node_num`, torch supplies the device,
the installed sglang version is cross-checked against the `wideep_sglang`
manifest pin), rows are emitted into the unified `moe_a2a_perf` table at
measurement time, and each run finalizes parquet plus a real
`collection_meta.yaml` sidecar. The old scripts remain in-tree only until
the new collector is hardware-validated; do not add new data to the shipped
tables via `extract_data.py`. The legacy `wideep_deepep_{normal,ll}_perf`
tables already shipped keep loading through the SDK's legacy adapters
indefinitely — deprecation retires the *pipeline*, not the data.

## Activating the vLLM `moe_ep` collector (plan decision D3)

`vllm/collect_moe_ep.py` is complete but dormant because no vLLM-DeepEP
runtime is pinned. Its dormancy is itself under test
(`tests/unit/collector/test_vllm_collect_moe_ep.py`): no registry module, no
manifest family, no hash-closures entry may exist before enrollment.
Enrollment is one coordinated change:

1. Pin a vLLM-DeepEP image: add a `wideep_vllm` entry to
   `collector/framework_manifest.yaml` (`base_framework: vllm`,
   `collector_dir: "collector/wideep/vllm"`, `data_backend: vllm`, a
   `default.version` and digest-pinned `images`).
2. Create `collector/wideep/vllm/registry.py` with
   `OpEntry(op="moe_ep", module="collector.wideep.vllm.collect_moe_ep",
   get_func="get_moe_ep_test_cases", run_func="run_moe_ep",
   perf_filename=PerfFile.MOE_EP)`, and enroll the framework key in
   `collector/framework_manifest.py` `_REGISTRY_MODULES`
   (`"wideep_vllm": "collector.wideep.vllm.registry"`) — registry/runtime
   resolution raises for a manifest entry with no registered module.
3. In the SAME commit (Task-1 sequencing rule: closures may not precede
   registration), add the `collector.wideep.vllm.collect_moe_ep` entry to
   `collector/hash_closures.yaml` (base-op yaml extras + `__model_cases__`,
   like its sglang/trtllm siblings).
4. Set `__compat__` in `collector/wideep/vllm/collect_moe_ep.py` to the
   pinned vllm version, and add the kernel-source fact to
   `collector/kernel_source_backends.yaml`: the module writes
   `kernel_source: deepep_moe`, which that table currently maps only for
   `framework: sglang` — the new `{framework: vllm, kernel_source:
   deepep_moe, backend: ...}` entry needs a citation into the verified vLLM
   dispatch (the same verification as item 6).
5. Flip the dormancy pins in
   `tests/unit/collector/test_vllm_collect_moe_ep.py`
   (`test_no_vllm_wideep_registry_exists`,
   `test_manifest_has_no_wideep_vllm_pin`,
   `test_registry_modules_have_no_wideep_vllm_entry`,
   `test_kernel_source_backends_have_no_vllm_deepep_moe_mapping`,
   `test_hash_closures_has_no_entry_for_the_unregistered_module`) into their
   positive counterparts.
6. On the pinned image, resolve the module's marked VERIFICATION ITEMs
   before trusting collected rows: prove that the benchmarked
   `fused_experts` call is the kernel vLLM's own serving dispatch selects on
   the large-EP DeepEP path (layer_permissions.md: `kernel_source` records
   ground truth, manual pins need source proof), and confirm the
   global-token accounting against a live run.
