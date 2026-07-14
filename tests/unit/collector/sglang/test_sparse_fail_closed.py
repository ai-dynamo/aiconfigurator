# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import os
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
DSV4_SOURCE = REPO_ROOT / "collector" / "sglang" / "deepseekv4_sparse_modules.py"
GLM5_SOURCE = REPO_ROOT / "collector" / "sglang" / "glm5_dsa_sparse_modules.py"


def _load_functions(source_path: Path, *names: str, namespace: dict | None = None) -> dict:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    loaded = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), loaded)
    return loaded


def _fake_torch():
    class OutOfMemoryError(RuntimeError):
        pass

    return SimpleNamespace(
        cuda=SimpleNamespace(
            OutOfMemoryError=OutOfMemoryError,
            empty_cache=lambda: None,
            get_device_name=lambda _device: "test-gpu",
        ),
        device=lambda device: device,
        no_grad=nullcontext,
    )


@pytest.mark.parametrize("source_path", [DSV4_SOURCE, GLM5_SOURCE])
def test_sparse_benchmarks_never_allow_cuda_graph_fallback(source_path):
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Name) or call.func.id != "_bench_cuda_graph":
            continue
        allow_graph_fail = next((keyword.value for keyword in call.keywords if keyword.arg == "allow_graph_fail"), None)
        assert not (isinstance(allow_graph_fail, ast.Constant) and allow_graph_fail.value is True)


def test_sparse_benchmark_rejects_helper_result_without_cuda_graph():
    benchmark_kwargs = {}

    class BenchmarkContext:
        def __enter__(self):
            return {"latency_ms": 1.0, "used_cuda_graph": False}

        def __exit__(self, *_args):
            return False

    def benchmark_with_power(**kwargs):
        benchmark_kwargs.update(kwargs)
        return BenchmarkContext()

    bench = _load_functions(
        DSV4_SOURCE,
        "_bench_cuda_graph",
        namespace={
            "Callable": Callable,
            "benchmark_with_power": benchmark_with_power,
            "torch": _fake_torch(),
        },
    )["_bench_cuda_graph"]

    with pytest.raises(RuntimeError, match="did not use CUDA Graph"):
        bench(lambda: None, allow_graph_fail=False)

    assert benchmark_kwargs["allow_graph_fail"] is False


def test_dsv4_grouped_worker_reports_every_inner_exception():
    fake_torch = _fake_torch()

    def fail_shape(_kernel, prefix, *_args, **_kwargs):
        if prefix == 11:
            raise ValueError("first failure")
        raise RuntimeError("second failure")

    worker = _load_functions(
        DSV4_SOURCE,
        "_guarded_bench",
        "run_dsv4_sparse_kernel_worker",
        namespace={
            "Callable": Callable,
            "KERNEL_TO_OP_NAME": {"paged_mqa_logits": "test_op"},
            "_bench_sparse_kernel_shape": fail_shape,
            "_dsv4_context_derived_shapes": lambda _model: [(11, 2, 3), (22, 4, 3)],
            "_dsv4_generation_derived_shapes": lambda _model: [],
            "_dsv4_sparse_config": lambda _model: object(),
            "_make_perf_filename": lambda _kernel, output_dir: os.path.join(output_dir, "test.txt"),
            "_write_row": lambda *_args, **_kwargs: pytest.fail("failed inner shapes must not be persisted"),
            "os": os,
            "torch": fake_torch,
            "traceback": SimpleNamespace(print_exc=lambda: None),
        },
    )["run_dsv4_sparse_kernel_worker"]

    with pytest.raises(RuntimeError) as exc_info:
        worker("test-model", "paged_mqa_logits", 3, perf_filename="out.txt")

    message = str(exc_info.value)
    assert "bs=3 isl=2 past_kv=11: ValueError: first failure" in message
    assert "bs=3 isl=4 past_kv=22: RuntimeError: second failure" in message


def test_glm5_grouped_worker_reports_every_inner_exception():
    fake_torch = _fake_torch()

    def fail_shape(_kernel, prefix, *_args, **_kwargs):
        if prefix == 13:
            raise KeyError("missing first")
        raise TypeError("bad second")

    worker = _load_functions(
        GLM5_SOURCE,
        "run_glm5_dsa_sparse_kernel_worker",
        namespace={
            "GLM5_ARCHITECTURE": "GlmMoeDsaForCausalLM",
            "KERNEL_TO_OP_NAME": {"mqa": "test_op"},
            "_bench_glm5_sparse_kernel_shape": fail_shape,
            "_dsa_context_derived_shapes": lambda _model: [(13, 2, 3), (29, 4, 3)],
            "_dsa_generation_derived_shapes": lambda _model: [],
            "_glm5_sparse_config": lambda _model: object(),
            "_guarded_bench": _load_functions(
                DSV4_SOURCE,
                "_guarded_bench",
                namespace={
                    "Callable": Callable,
                    "torch": fake_torch,
                    "traceback": SimpleNamespace(print_exc=lambda: None),
                },
            )["_guarded_bench"],
            "_make_perf_filename": lambda _kernel, output_dir, _op_name_map: os.path.join(output_dir, "test.txt"),
            "_write_row": lambda *_args, **_kwargs: pytest.fail("failed inner shapes must not be persisted"),
            "os": os,
            "sys": SimpleNamespace(argv=[]),
            "torch": fake_torch,
        },
    )["run_glm5_dsa_sparse_kernel_worker"]

    with pytest.raises(RuntimeError) as exc_info:
        worker("test-model", "mqa", 3, perf_filename="out.txt")

    message = str(exc_info.value)
    assert "bs=3 isl=2 past_kv=13: KeyError: 'missing first'" in message
    assert "bs=3 isl=4 past_kv=29: TypeError: bad second" in message
