# Collector coverage manifests

`collector_v1/` is the canonical checked-in source for planner/runtime
physical-key coverage guards. Tests load these files in place; there is no
duplicate copy under `tests/fixtures`.

The directory name reflects the original context/generation baseline. Encoder
attention did not exist at that collector-v1 ref, so its files are a separate
`encoder_initial_baseline`:

| Operations | Source ref | Scope | Exact keys |
| --- | --- | --- | ---: |
| Context/generation, CUDA | `a4827ce203e9fbc24fe6c6779a7eaa2a7dc79f1a` | `b200_sxm`, SM 100 | Per manifest |
| Context/generation, XPU | `a4827ce203e9fbc24fe6c6779a7eaa2a7dc79f1a` | `xpu`, SM 0 | Per manifest |
| Encoder, each CUDA backend | `36808ecced9af9d0d71d944c716ae96d1d4a2a47` | `b200_sxm`, SM 100 | 7,008 |

The encoder ref is the original hardcoded collector commit inside PR #1092.
The PR's final squash, `1ce6ff602f18e5d6fa46955fe8ca71e540bbc60e`,
had already moved those cases to YAML and changed its sequence grid. Encoder
headers use the original modules' minimum compatible framework versions:
SGLang 0.5.11, TensorRT-LLM 1.3.0rc5, and vLLM 0.21.0.

Regenerate or byte-check every manifest deterministically with:

```shell
.venv/bin/python tools/collector/generate_v1_attention_manifests.py
.venv/bin/python tools/collector/generate_v1_attention_manifests.py --check
```

Manifest entries are consumer-visible physical row keys, not raw scheduler
cases. Coverage enforcement must first match the full header scope; a
different backend, framework version outside the declared exact/range scope,
GPU system id, SM, or perf table is out-of-scope rather than a comparable
failure.
