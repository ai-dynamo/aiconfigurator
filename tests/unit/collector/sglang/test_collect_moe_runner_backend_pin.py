# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proves ``_pin_moe_runner_backend`` (collect_moe.py) lands the fused-MoE
runner-backend pin through the correct mechanism on both sglang 0.5.14 and
0.5.17 (AIC-1762 Task 4d).

No GPU/sglang install is available in this environment (verified:
``import sglang`` raises ``ModuleNotFoundError`` here), so this cannot run
the real ``FusedMoE`` construction chain end-to-end. What CAN be proven off-
GPU, and is proven below: (1) the version-branch selects the right mechanism
for each sglang generation, verified by injecting a fake
``sglang.srt.runtime_context`` module via ``monkeypatch.setitem(sys.modules,
...)`` -- the same technique already used by
``test_collect_gdn_contract.py``'s ``TestResolveFlashinferGdnDecode`` for an
analogous runtime-capability-probe function; (2) each mechanism's pin lands
on the object sglang's own runner-construction chain actually reads
(``sglang.srt.layers.moe.utils.MOE_RUNNER_BACKEND`` at 0.5.14,
``get_flags().moe.runner_backend`` at 0.5.17 -- see the docstring on
``_pin_moe_runner_backend`` itself for the full file:line trace into
``FusedMoE.__init__``); (3) both mechanisms restore the previous value on a
clean exit AND on an exception raised inside the ``with`` block (the
collector's own construction/benchmark code can raise mid-pin); (4) the pin
reproduces the exact backend strings the case plan's ``sglang_moe_backends``
declares for this model -- ``triton`` (bf16/fp8_block, sglang's own "auto"
default per ``Qwen3_5MoeForCausalLM_cases.yaml:97-118``) and
``flashinfer_trtllm`` (nvfp4, ``Qwen3_5MoeForCausalLM_cases.yaml:143-144``)
-- not generic placeholder values, so a value-corrupting bug in the pin
helper (e.g. an off-by-one enum, a hardcoded default) would fail these
tests even if a generic-sentinel version only proved plumbing.
"""

import ast
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_moe.py"


def _load_pin_function(namespace: dict | None = None):
    """AST-extract _pin_moe_runner_backend and exec it standalone.

    ``MoeRunnerBackend`` is only ever used here as a type annotation (never
    called/compared), so a plain placeholder satisfies it without needing to
    reimplement the enum.
    """
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_pin_moe_runner_backend"
    )
    loaded = {"contextmanager": contextmanager, "MoeRunnerBackend": object, **(namespace or {})}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded["_pin_moe_runner_backend"]


class FakeMoeFlags:
    """Stand-in for sglang 0.5.17's MoeFlags (runtime_context.py:359-378):
    same override() shape as the real _FlagGroupBase.override()
    (runtime_context.py:323-341) -- transactional, restores on exit."""

    def __init__(self):
        self.runner_backend = None
        self.override_calls: list[dict] = []

    @contextmanager
    def override(self, **kwargs):
        self.override_calls.append(dict(kwargs))
        saved = {name: getattr(self, name) for name in kwargs}
        for name, value in kwargs.items():
            setattr(self, name, value)
        try:
            yield self
        finally:
            for name, value in saved.items():
                setattr(self, name, value)


def _inject_fake_runtime_context(monkeypatch) -> FakeMoeFlags:
    """Install a fake sglang.srt.runtime_context exporting get_flags(), the
    same sys.modules-injection technique test_collect_gdn_contract.py's
    TestResolveFlashinferGdnDecode uses for flashinfer.gdn_decode."""
    fake_moe_flags = FakeMoeFlags()
    fake_flags_singleton = types.SimpleNamespace(moe=fake_moe_flags)
    fake_runtime_context = types.ModuleType("sglang.srt.runtime_context")
    fake_runtime_context.get_flags = lambda: fake_flags_singleton
    fake_sglang_srt = types.ModuleType("sglang.srt")
    fake_sglang_srt.runtime_context = fake_runtime_context
    fake_sglang = types.ModuleType("sglang")
    fake_sglang.srt = fake_sglang_srt

    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", fake_sglang_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", fake_runtime_context)
    return fake_moe_flags


@pytest.mark.parametrize("backend_value", ["triton", "flashinfer_trtllm"])
class TestPin0514BareGlobal:
    """sglang genuinely is not installed in this venv (verified:
    ``import sglang`` raises ModuleNotFoundError), so `from sglang.srt.
    runtime_context import get_flags` fails here exactly as it would on a
    real 0.5.14 install (that module has no `get_flags` export before
    0.5.17 -- see _pin_moe_runner_backend's docstring). No injection needed:
    this is the collector's real behavior on 0.5.14 today, unchanged."""

    def test_pins_and_restores_the_module_global(self, backend_value):
        fake_moe_utils = types.SimpleNamespace(MOE_RUNNER_BACKEND=None)
        pin = _load_pin_function({"_moe_utils": fake_moe_utils})

        assert fake_moe_utils.MOE_RUNNER_BACKEND is None
        with pin(backend_value):
            assert backend_value == fake_moe_utils.MOE_RUNNER_BACKEND
        assert fake_moe_utils.MOE_RUNNER_BACKEND is None

    def test_restores_on_exception(self, backend_value):
        fake_moe_utils = types.SimpleNamespace(MOE_RUNNER_BACKEND="PRE_EXISTING")
        pin = _load_pin_function({"_moe_utils": fake_moe_utils})

        with pytest.raises(RuntimeError, match="boom"), pin(backend_value):
            assert backend_value == fake_moe_utils.MOE_RUNNER_BACKEND
            raise RuntimeError("boom")
        assert fake_moe_utils.MOE_RUNNER_BACKEND == "PRE_EXISTING"


@pytest.mark.parametrize("backend_value", ["triton", "flashinfer_trtllm"])
class TestPin0517RuntimeContextFlags:
    """sglang 0.5.17: get_flags().moe.runner_backend via the RuntimeContext/
    Flags singleton, pinned through its own override() primitive."""

    def test_pins_through_get_flags_moe_override_and_restores(self, monkeypatch, backend_value):
        fake_moe_flags = _inject_fake_runtime_context(monkeypatch)
        fake_moe_utils = types.SimpleNamespace(MOE_RUNNER_BACKEND="MUST_NOT_BE_TOUCHED")
        pin = _load_pin_function({"_moe_utils": fake_moe_utils})

        assert fake_moe_flags.runner_backend is None
        with pin(backend_value):
            assert fake_moe_flags.runner_backend == backend_value
            assert fake_moe_flags.override_calls == [{"runner_backend": backend_value}]
            # The 0.5.14 mechanism must be left completely alone once the
            # 0.5.17 mechanism is in play -- a bug that pinned both would
            # be invisible to a test that only checks one.
            assert fake_moe_utils.MOE_RUNNER_BACKEND == "MUST_NOT_BE_TOUCHED"
        assert fake_moe_flags.runner_backend is None

    def test_restores_on_exception(self, monkeypatch, backend_value):
        fake_moe_flags = _inject_fake_runtime_context(monkeypatch)
        fake_moe_utils = types.SimpleNamespace(MOE_RUNNER_BACKEND=None)
        pin = _load_pin_function({"_moe_utils": fake_moe_utils})

        with pytest.raises(RuntimeError, match="boom"), pin(backend_value):
            assert fake_moe_flags.runner_backend == backend_value
            raise RuntimeError("boom")
        assert fake_moe_flags.runner_backend is None
