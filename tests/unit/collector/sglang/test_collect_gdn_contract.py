# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import gc
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_gdn.py"
BACKEND_MAP_PATH = Path(__file__).resolve().parents[4] / "collector" / "kernel_source_backends.yaml"


def test_gdn_causal_conv_mapping_preserves_version_ambiguity_and_exact_siblings():
    import yaml

    mappings = yaml.safe_load(BACKEND_MAP_PATH.read_text(encoding="utf-8"))["mappings"]
    by_source_and_op = {
        (entry["kernel_source"], (entry.get("match") or {}).get("op_file")): entry["backend"]
        for entry in mappings
        if entry["framework"] == "sglang"
    }

    assert by_source_and_op[("causal_conv1d_update", "gdn_perf")] == "unverified"
    assert by_source_and_op[("causal_conv1d_update", "kda_perf")] == "triton"
    assert by_source_and_op[("flashinfer_gated_delta_rule_decode", None)] == "flashinfer"


def test_gdn_context_does_not_silently_drop_fixed_capacity_shapes():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_gdn_context_benchmark"
    )
    referenced_names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}

    assert "MAX_GDN_CONTEXT_TOKENS" not in referenced_names
    assert "MAX_GDN_CONTEXT_VALUE_ELEMENTS" not in referenced_names
    assert "skipped_points" not in referenced_names


def test_gdn_context_raises_on_conv_int32_offset_overflow():
    # Verified framework kernel limit, not a silent skip: stock 0.5.14
    # _causal_conv1d_fwd_kernel int32 token-offset overflow at 2**31 packed
    # elements (causal_conv1d_triton.py:373-379; RTX 6000 Pro memcheck
    # 2026-07-06). The guard must RAISE inside the sweep loop so the cell
    # contributes to the failing group summary instead of corrupting the CUDA
    # context and aborting the remaining cells.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "total_tokens * conv_channels >= 2**31" in source
    assert "int32 token-offset overflow" in source
    assert "causal_conv1d_triton.py:373-379" in source


def _load_function(source_path: Path, name: str, namespace: dict | None = None):
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    loaded = dict(namespace or {})
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), loaded)
    return loaded[name]


class TestResolveFlashinferGdnDecode:
    """_resolve_flashinfer_gdn_decode (CodeRabbit collect_gdn.py:369, Major;
    lane-predicate fix, jasonqinzhou PR #1533 review): serving auto-selects
    the FlashInfer bf16-state GDN decode kernel only on capability major 10
    (is_sm100_supported() — sm 100/103 yes, sm 120 NO) AND when the model's
    mamba_ssm_dtype is bfloat16 (server_args.py:4884-4915 @0.5.14). An
    import failure must surface as a classified failure the caller raises
    ONLY for bf16-state cases; fp32-state (every bundled Qwen3.5/3.6 config)
    and non-major-10 SMs stay legitimate no-ops even when FlashInfer is
    importable."""

    def _resolve(self, sm_version: int):
        return _load_function(
            SOURCE_PATH,
            "_resolve_flashinfer_gdn_decode",
            {"get_sm_version": lambda: sm_version},
        )

    def test_not_applicable_below_sm100(self):
        resolve = self._resolve(90)

        assert resolve("bfloat16") == (None, None)

    def test_not_applicable_on_sm120_regardless_of_dtype(self):
        # is_sm100_supported() is capability major EXACTLY 10: sm 120
        # (rtx_pro_6000_server) never takes the FlashInfer decode lane, so
        # the resolver must not require (or even attempt) it there.
        resolve = self._resolve(120)

        assert resolve("float32") == (None, None)
        assert resolve("bfloat16") == (None, None)

    def test_boundary_sm100_takes_mandatory_lane_branch_for_bf16(self, monkeypatch):
        # SM100 itself (the boundary) must already take the mandatory-lane
        # branch for a bf16-state case: the guard is `100 <= sm < 110`.
        monkeypatch.delitem(sys.modules, "flashinfer.gdn_decode", raising=False)
        monkeypatch.delitem(sys.modules, "flashinfer", raising=False)
        resolve = self._resolve(100)

        kernel_fn, error_message = resolve("bfloat16")

        assert kernel_fn is None
        assert error_message is not None
        assert "SM100" in error_message

    def test_classified_error_when_unavailable_on_sm100_bf16(self, monkeypatch):
        # flashinfer is genuinely not installed in this dev/CI venv, so this
        # reproduces the real gap without needing a sys.modules trick: the
        # previous code returned a bare None here (CodeRabbit finding), so
        # the caller happily skipped the row and the case reported success.
        monkeypatch.delitem(sys.modules, "flashinfer.gdn_decode", raising=False)
        monkeypatch.delitem(sys.modules, "flashinfer", raising=False)
        resolve = self._resolve(103)

        kernel_fn, error_message = resolve("bfloat16")

        assert kernel_fn is None
        assert error_message is not None
        assert "SM103" in error_message
        assert "collection environment gap" in error_message

    def test_fp32_state_never_resolves_flashinfer_on_sm100(self, monkeypatch):
        # Serving never selects the FlashInfer backend for an fp32-state
        # model (every bundled Qwen3.5/3.6 config), even when the package is
        # importable in the collection image.
        sentinel = object()
        fake_gdn_decode = types.ModuleType("flashinfer.gdn_decode")
        fake_gdn_decode.gated_delta_rule_decode_pretranspose = sentinel
        fake_flashinfer = types.ModuleType("flashinfer")
        fake_flashinfer.gdn_decode = fake_gdn_decode
        monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
        monkeypatch.setitem(sys.modules, "flashinfer.gdn_decode", fake_gdn_decode)
        resolve = self._resolve(103)

        assert resolve("float32") == (None, None)

    def test_explicit_bf16_fixture_returns_kernel_when_available_on_sm100(self, monkeypatch):
        sentinel = object()
        fake_gdn_decode = types.ModuleType("flashinfer.gdn_decode")
        fake_gdn_decode.gated_delta_rule_decode_pretranspose = sentinel
        fake_flashinfer = types.ModuleType("flashinfer")
        fake_flashinfer.gdn_decode = fake_gdn_decode
        monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
        monkeypatch.setitem(sys.modules, "flashinfer.gdn_decode", fake_gdn_decode)
        resolve = self._resolve(103)

        kernel_fn, error_message = resolve("bfloat16")

        assert kernel_fn is sentinel
        assert error_message is None

    def test_current_repo_plan_has_zero_non_serving_flashinfer_invocations(self, monkeypatch):
        from collector.case_generator import get_common_gdn_test_cases

        sentinel = object()
        fake_gdn_decode = types.ModuleType("flashinfer.gdn_decode")
        fake_gdn_decode.gated_delta_rule_decode_pretranspose = sentinel
        fake_flashinfer = types.ModuleType("flashinfer")
        fake_flashinfer.gdn_decode = fake_gdn_decode
        monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
        monkeypatch.setitem(sys.modules, "flashinfer.gdn_decode", fake_gdn_decode)
        resolve = self._resolve(103)

        generation_cases = [case for case in get_common_gdn_test_cases() if case.phase == "generation"]

        assert len(generation_cases) == 37
        assert {case.mamba_ssm_dtype for case in generation_cases} == {"float32"}
        assert [case for case in generation_cases if resolve(case.mamba_ssm_dtype)[0] is not None] == []


class _FakeTensor:
    def view(self, *_args):
        return self

    def detach(self):
        return self

    def float(self):
        return self


class _FakeCuda:
    @staticmethod
    def set_device(_device):
        pass

    @staticmethod
    def get_device_name(_device):
        return "fake-gpu"

    @staticmethod
    def empty_cache():
        pass


class _FakeTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"
    int32 = "int32"
    cuda = _FakeCuda()

    @staticmethod
    def device(value):
        return value

    @staticmethod
    def set_default_device(_device):
        pass

    @staticmethod
    def randn(*_args, **_kwargs):
        return _FakeTensor()

    @staticmethod
    def zeros(*_args, **_kwargs):
        return _FakeTensor()

    @staticmethod
    def ones(*_args, **_kwargs):
        return _FakeTensor()

    @staticmethod
    def arange(*_args, **_kwargs):
        return _FakeTensor()

    @staticmethod
    def empty(*_args, **_kwargs):
        return _FakeTensor()

    @staticmethod
    def split(_tensor, _sizes, dim=-1):
        assert dim == -1
        return _FakeTensor(), _FakeTensor(), _FakeTensor()


class _FakeBenchmark:
    def __init__(self, kernel_func):
        self._kernel_func = kernel_func

    def __enter__(self):
        self._kernel_func()
        return {"latency_ms": 1.0, "power_stats": None}

    def __exit__(self, *_args):
        return False


def _run_generation_case(
    *, dtype, flashinfer_kernel, flashinfer_error=None, invoked=None, logged=None, allocations=None, resolve=None
):
    invoked = [] if invoked is None else invoked
    logged = [] if logged is None else logged
    allocations = [] if allocations is None else allocations
    fake_torch = _FakeTorch()
    original_zeros = fake_torch.zeros

    def recording_zeros(*args, **kwargs):
        allocations.append((args, kwargs.get("dtype")))
        return original_zeros(*args, **kwargs)

    fake_torch.zeros = recording_zeros
    run_generation = _load_function(
        SOURCE_PATH,
        "run_gdn_generation_benchmark",
        {
            "gc": gc,
            "torch": fake_torch,
            "aic_debug": 0,
            "causal_conv1d_update": lambda *_args, **_kwargs: invoked.append("causal_conv1d_update"),
            "fused_recurrent_gated_delta_rule_packed_decode": lambda **_kwargs: invoked.append(
                "fused_recurrent_gated_delta_rule_packed_decode"
            ),
            "_resolve_flashinfer_gdn_decode": resolve or (lambda _dtype: (flashinfer_kernel, flashinfer_error)),
            "benchmark_with_power": lambda *, kernel_func, **_kwargs: _FakeBenchmark(kernel_func),
            "log_perf": lambda **kwargs: logged.append(kwargs["kernel_source"]) or True,
        },
    )

    run_generation(
        d_model=2048,
        d_conv=4,
        num_k_heads=16,
        head_k_dim=128,
        num_v_heads=32,
        head_v_dim=128,
        batch_size_list=[1],
        model_name="explicit-test-fixture",
        perf_filename="unused.txt",
        sglang_version="0.5.14",
        device="cuda:0",
        mamba_ssm_dtype=dtype,
    )
    return invoked, logged, allocations


@pytest.mark.parametrize(
    ("dtype", "flashinfer_selected", "expected_decode_source"),
    [
        pytest.param("float32", False, "fused_recurrent_gated_delta_rule_packed_decode", id="sm100-fp32-fla"),
        pytest.param("bfloat16", False, "fused_recurrent_gated_delta_rule_packed_decode", id="sm120-bf16-fla"),
        pytest.param("bfloat16", True, "flashinfer_gated_delta_rule_decode", id="sm100-bf16-flashinfer"),
    ],
)
def test_generation_invokes_and_logs_exactly_one_serving_decode_backend(
    dtype, flashinfer_selected, expected_decode_source
):
    def flashinfer_kernel(**_kwargs):
        invoked_by_flashinfer.append("flashinfer_gated_delta_rule_decode")

    invoked_by_flashinfer = []
    invoked, logged, _allocations = _run_generation_case(
        dtype=dtype,
        flashinfer_kernel=flashinfer_kernel if flashinfer_selected else None,
    )
    invoked += invoked_by_flashinfer

    assert [source for source in invoked if "gated_delta_rule" in source] == [expected_decode_source]
    assert [source for source in logged if "gated_delta_rule" in source] == [expected_decode_source]


def test_generation_missing_selected_flashinfer_raises_classified_failure_without_fla_row():
    error = "SM103 FlashInfer collection environment gap"
    invoked = []
    logged = []

    with pytest.raises(RuntimeError, match=error):
        _run_generation_case(
            dtype="bfloat16",
            flashinfer_kernel=None,
            flashinfer_error=error,
            invoked=invoked,
            logged=logged,
        )

    assert "fused_recurrent_gated_delta_rule_packed_decode" not in invoked
    assert "fused_recurrent_gated_delta_rule_packed_decode" not in logged
    assert logged == ["causal_conv1d_update"]


@pytest.mark.parametrize(
    ("sm_version", "dtype", "flashinfer_selected", "expected_decode_source", "expected_state_dtype"),
    [
        pytest.param(
            100, "float32", False, "fused_recurrent_gated_delta_rule_packed_decode", "float32", id="sm100-fp32-fla"
        ),
        pytest.param(
            103, "float32", False, "fused_recurrent_gated_delta_rule_packed_decode", "float32", id="sm103-fp32-fla"
        ),
        pytest.param(
            120, "bfloat16", False, "fused_recurrent_gated_delta_rule_packed_decode", "bfloat16", id="sm120-bf16-fla"
        ),
        pytest.param(
            90, "float16", False, "fused_recurrent_gated_delta_rule_packed_decode", "float16", id="sm90-fp16-fla"
        ),
        pytest.param(
            103, "bfloat16", True, "flashinfer_gated_delta_rule_decode", "bfloat16", id="sm103-bf16-flashinfer"
        ),
    ],
)
def test_generation_dynamic_dispatch_allocates_serving_state_dtype(
    monkeypatch,
    sm_version,
    dtype,
    flashinfer_selected,
    expected_decode_source,
    expected_state_dtype,
):
    invoked_by_flashinfer = []

    def flashinfer_kernel(**_kwargs):
        invoked_by_flashinfer.append("flashinfer_gated_delta_rule_decode")

    fake_gdn_decode = types.ModuleType("flashinfer.gdn_decode")
    fake_gdn_decode.gated_delta_rule_decode_pretranspose = flashinfer_kernel
    fake_flashinfer = types.ModuleType("flashinfer")
    fake_flashinfer.gdn_decode = fake_gdn_decode
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.gdn_decode", fake_gdn_decode)
    resolve = _load_function(
        SOURCE_PATH,
        "_resolve_flashinfer_gdn_decode",
        {"get_sm_version": lambda: sm_version},
    )

    invoked, logged, allocations = _run_generation_case(
        dtype=dtype,
        flashinfer_kernel=flashinfer_kernel if flashinfer_selected else None,
        resolve=resolve,
    )
    invoked += invoked_by_flashinfer

    assert [source for source in invoked if "gated_delta_rule" in source] == [expected_decode_source]
    assert [source for source in logged if "gated_delta_rule" in source] == [expected_decode_source]
    state_allocations = [allocation_dtype for (shape, allocation_dtype) in allocations if len(shape) == 4]
    assert state_allocations == [expected_state_dtype]


def test_generation_dynamic_missing_flashinfer_raises_without_fla_invocation_or_row(monkeypatch):
    monkeypatch.delitem(sys.modules, "flashinfer.gdn_decode", raising=False)
    monkeypatch.delitem(sys.modules, "flashinfer", raising=False)
    resolve = _load_function(
        SOURCE_PATH,
        "_resolve_flashinfer_gdn_decode",
        {"get_sm_version": lambda: 100},
    )
    invoked = []
    logged = []

    with pytest.raises(RuntimeError, match="FlashInfer bf16 GDN decode lane required but unavailable"):
        _run_generation_case(
            dtype="bfloat16",
            flashinfer_kernel=None,
            resolve=resolve,
            invoked=invoked,
            logged=logged,
        )

    assert "fused_recurrent_gated_delta_rule_packed_decode" not in invoked
    assert "fused_recurrent_gated_delta_rule_packed_decode" not in logged
    assert logged == ["causal_conv1d_update"]
