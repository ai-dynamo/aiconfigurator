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
- `sglang/deepep/`: multi-node DeepEP log collection and extraction scripts.
- `trtllm/collect_moe_compute.py`: TensorRT-LLM WideEP MoE compute entrypoint
  (op `moe_ep`; pins the same image as stock trtllm, so model plans activate
  it by default for wideep-declared models).
- `vllm/collect_moe_ep.py`: vLLM fused-experts moe_ep bench path, DORMANT —
  no registry, no manifest entry, no hash-closures entry until a vLLM-DeepEP
  image is pinned (plan decision D3).
