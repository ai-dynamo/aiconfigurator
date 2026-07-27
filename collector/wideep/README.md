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

- `sglang/collect_mla_module.py`: legacy WideEP MLA wrappers (not registered
  while stock SGLang and the WideEP image use different releases).
- `sglang/collect_deepep_moe.py`: SGLang DeepEP MoE entrypoint.
- `sglang/deepep/`: multi-node DeepEP log collection and extraction scripts.
- `trtllm/collect_moe_compute.py`: TensorRT-LLM WideEP MoE compute entrypoint.
