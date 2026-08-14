# WideEP collectors

Large-EP expert compute has no dedicated collector or table. The SDK maps a
uniformly balanced rank-local assignment stream onto reusable stock
`moe_perf`; communication remains measured independently in `moe_a2a_perf`.

Layout:

- `sglang/collect_moe_a2a.py`: standalone multi-node DeepEP all-to-all
  collector, launched by `collector/network/slurm/submit_moe_a2a.sh`.
- `vllm/collect_moe_a2a.py`: vLLM serving-path adapter over the shared A2A
  collection and row lifecycle.
- `trtllm/collect_moe_a2a.py`: TensorRT-LLM serving-path A2A collector.
- `sglang/deepep/`: deprecated manual multi-node log collection/extraction.

The legacy `wideep_deepep_{normal,ll}_perf` communication tables remain
readable through SDK adapters. Deprecation retires their producer pipeline,
not already shipped communication data.
