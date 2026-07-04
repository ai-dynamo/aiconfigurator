# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def test_mhc_sweep_fails_closed_after_an_inner_error(monkeypatch):
    torch = ModuleType("torch")
    torch.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})
    torch.cuda = SimpleNamespace(
        OutOfMemoryError=type("CudaOutOfMemoryError", (Exception,), {}),
        empty_cache=lambda: None,
        get_device_name=lambda _device: "test-gpu",
    )
    torch.distributed = SimpleNamespace(is_initialized=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    parallel_state = ModuleType("sglang.srt.distributed.parallel_state")
    parallel_state.destroy_model_parallel = lambda: None
    for name in ("sglang", "sglang.srt", "sglang.srt.distributed"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, parallel_state.__name__, parallel_state)

    module_path = Path("collector/sglang/collect_mhc_module.py")
    spec = importlib.util.spec_from_file_location("mhc_fail_closed_target", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    layer = SimpleNamespace(hc_mult=4, config=SimpleNamespace(hc_sinkhorn_iters=20))
    runner = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=[layer])))
    monkeypatch.setattr(module, "_load_one_layer_runner", lambda *_args, **_kwargs: runner)
    monkeypatch.setattr(module, "_hidden_size", lambda _layer: 4096)
    monkeypatch.setattr(module, "get_version", lambda _package: "0.5.14")
    monkeypatch.setattr(module, "_make_residual", lambda _layer, num_tokens, _device: num_tokens)
    monkeypatch.setattr(module, "_make_kernel", lambda _layer, _op, residual: residual)
    monkeypatch.setattr(module, "_mhc_call_args", lambda _layer: (None, None))
    monkeypatch.setattr(module, "_log_result", lambda **_kwargs: None)

    def benchmark(*, kernel_func, **_kwargs):
        if kernel_func == 2:
            raise RuntimeError("injected inner failure")
        return {"latency_ms": 1.0, "num_runs_executed": 3}

    monkeypatch.setattr(module, "_benchmark_mhc_kernel", benchmark)

    with pytest.raises(RuntimeError, match=r"mHC sweep failed: ok=1 error=1 skip=0 total=2"):
        module.run_mhc_module(ops=["pre"], num_tokens_cases=[1, 2], num_iterations=3)
