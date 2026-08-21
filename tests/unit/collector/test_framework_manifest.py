# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for collector framework version/image manifest."""

from pathlib import Path

import pytest
import yaml

from collector.framework_manifest import (
    get_collector_runtime,
    require_collector_runtime,
    resolve_op_runtime,
    validate_resolution,
)
from collector.sglang.registry import REGISTRY as SGLANG_REGISTRY
from collector.trtllm.registry import REGISTRY as TRTLLM_REGISTRY
from collector.version_resolver import _check_compat
from collector.vllm.registry import REGISTRY as VLLM_REGISTRY
from collector.wideep.sglang import dataset_version_label
from collector.wideep.sglang.registry import REGISTRY as WIDEEP_SGLANG_REGISTRY
from collector.wideep.trtllm.registry import REGISTRY as WIDEEP_TRTLLM_REGISTRY

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTOR_ROOT = REPO_ROOT / "collector"


def test_manifest_exposes_current_framework_versions_and_images():
    sglang = get_collector_runtime("sglang")
    trtllm = get_collector_runtime("trtllm")
    vllm = get_collector_runtime("vllm")

    assert sglang.version == "0.5.14"
    assert sglang.image().startswith("lmsysorg/sglang:v0.5.14@sha256:")
    assert sglang.image("cu130").startswith("lmsysorg/sglang:v0.5.14-cu130@sha256:")
    assert trtllm.version == "1.3.0rc20"
    assert trtllm.image().startswith("nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20@sha256:")
    assert vllm.version == "0.24.0"
    assert vllm.image().startswith("vllm/vllm-openai:v0.24.0@sha256:")
    assert vllm.image("cu129").startswith("vllm/vllm-openai:v0.24.0-cu129@sha256:")
    # Unknown variants intentionally fall back to the pinned default image.
    assert vllm.image("cu130") == vllm.image()


def test_active_cuda_vllm_collectors_are_exactly_pinned_to_manifest_version():
    assert all(not entry.versions for entry in VLLM_REGISTRY)

    # Each module pins the runtime that actually collects it: the manifest
    # default, or its family override (e.g. kda runs only on the vllm kimi-k3
    # preview image, frameworks.vllm.families.kda). moe/gdn/gemm now carry a
    # 2-version-set __compat__ (AIC-1782 Task V1, the same grammar-limited
    # approximation as the sglang collectors -- see TestExactVersionSetVllm
    # in test_version_resolver.py for the probed leak semantics), so the
    # invariant checked here is semantic (the declared __compat__ accepts
    # the version that actually resolves for this module) rather than a
    # literal single-version string match -- the same check collect.py/
    # fullnode.py perform at runtime via _check_compat. This still fully
    # subsumes the old exact-equality behavior for every other vllm module
    # (an `==X.Y.Z` pin only ever accepts that one version).
    module_versions: dict[str, set[str]] = {}
    for entry in VLLM_REGISTRY:
        module_versions.setdefault(entry.module, set()).add(resolve_op_runtime("vllm", entry.op).version)

    for module, versions in sorted(module_versions.items()):
        assert len(versions) == 1, (module, versions)
        resolved_version = next(iter(versions))
        source = (REPO_ROOT / f"{module.replace('.', '/')}.py").read_text(encoding="utf-8")
        declarations = [line.strip() for line in source.splitlines() if line.startswith("__compat__")]
        assert len(declarations) == 1, module
        declared = declarations[0].split("=", 1)[1].strip().strip('"')
        assert _check_compat(declared, resolved_version), (module, declared, resolved_version)


@pytest.mark.parametrize(
    "module",
    ["collector.vllm.collect_moe", "collector.vllm.collect_gdn", "collector.vllm.collect_gemm"],
)
def test_vllm_target_lane_collectors_declare_the_exact_bumped_compat_range(module):
    # AIC-1782 Task V1: these three collectors were re-verified against vLLM
    # v0.27.1 and bumped from the exact pin `vllm==0.24.0` to the widest
    # range __compat__'s grammar allows for an accepted {0.24.0, 0.27.1}
    # pair while still excluding the never-verified 0.25.0/0.25.1/0.26.0/
    # 0.27.0 intermediate releases. This locks the literal string (the test
    # above only checks it semantically accepts the resolved default
    # version) so an accidental widening/narrowing of the range is caught
    # even though 0.24.0 alone would still satisfy a looser or tighter one.
    expected = '__compat__ = "vllm>=0.24.0,<=0.27.1,!=0.25.0,!=0.25.1,!=0.26.0,!=0.27.0"'
    source = (REPO_ROOT / f"{module.replace('.', '/')}.py").read_text(encoding="utf-8")
    declarations = [line.strip() for line in source.splitlines() if line.startswith("__compat__")]
    assert declarations == [expected], module


def test_wideep_runtime_stays_independent_from_default_framework_runtime():
    wideep_sglang = get_collector_runtime("sglang", workload="wideep")
    assert wideep_sglang.version == "0.5.10"
    assert wideep_sglang.version != get_collector_runtime("sglang").version
    assert wideep_sglang.collector_dir == "collector/wideep/sglang"
    assert "deepseek-v4" in wideep_sglang.image()


def test_deepep_ops_resolve_to_the_comm_family_runtime(monkeypatch):
    # The `comm` family override retargets exactly the two DeepEP ops; moe_ep
    # (the retired wideep_moe's successor) is family `moe` and stays on the
    # DeepSeek-V4 runtime its 0.5.10 dataset was collected with, where DSv4
    # module support is verified.
    moe = resolve_op_runtime("wideep_sglang", "moe_ep")
    assert (moe.family, moe.version) == ("moe", "0.5.10")
    assert "deepseek-v4" in moe.image()

    for op, env_var in (("deepep_ll", "DEEPEP_LL_VERSION"), ("deepep_normal", "DEEPEP_NORMAL_VERSION")):
        monkeypatch.delenv(env_var, raising=False)
        runtime = resolve_op_runtime("wideep_sglang", op)
        assert (runtime.family, runtime.version) == ("comm", "0.5.12")
        assert runtime.image().startswith("lmsysorg/sglang:v0.5.12-cu130@sha256:")
        # multi-arch index: one entry serves arm64 too, so no grace variant
        assert runtime.image("grace_blackwell") == runtime.image()
        # the version column on the rows must name the directory they land in
        assert dataset_version_label(env_var, op) == runtime.version
        monkeypatch.setenv(env_var, "9.9.9")
        assert dataset_version_label(env_var, op) == "9.9.9"


def test_deepep_and_wideep_moe_cannot_share_one_container():
    with pytest.raises(RuntimeError) as excinfo:
        require_collector_runtime("sglang", "0.5.12", requested_ops={"moe_ep", "deepep_ll"}, wideep_ops=WIDEEP_OPS)
    message = str(excinfo.value)
    assert "deepep_ll→0.5.12" in message
    assert "moe_ep→0.5.10" in message
    assert "run each version group in its own container" in message


def test_wideep_entries_are_flattened_peer_frameworks():
    # workload="wideep" is the compatibility spelling for manifest key wideep_<fw>
    via_workload = get_collector_runtime("sglang", workload="wideep")
    direct = get_collector_runtime("wideep_sglang")
    assert via_workload == direct
    assert direct.framework == "wideep_sglang"
    assert direct.data_backend == "sglang"
    assert direct.collector_dir == "collector/wideep/sglang"
    # wideep inherits the base framework's source_repo unless overridden
    assert direct.source_repo == get_collector_runtime("sglang").source_repo


def test_public_images_must_be_digest_pinned(tmp_path):
    manifest = tmp_path / "framework_manifest.yaml"
    manifest.write_text(
        """
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digest-pinned"):
        get_collector_runtime("sglang", path=manifest)


def test_wideep_entry_missing_base_framework_is_rejected(tmp_path):
    digest = "@sha256:" + "0" * 64
    manifest = tmp_path / "framework_manifest.yaml"
    manifest.write_text(
        f"""
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14{digest}"
  wideep_sglang:
    collector_dir: "collector/wideep/sglang"
    data_backend: "sglang"
    default:
      version: "0.5.10"
      images:
        default: "deepseek-v4-blackwell"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="base_framework"):
        get_collector_runtime("wideep_sglang", path=manifest)


def test_wideep_entry_missing_data_backend_is_rejected(tmp_path):
    digest = "@sha256:" + "0" * 64
    manifest = tmp_path / "framework_manifest.yaml"
    manifest.write_text(
        f"""
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14{digest}"
  wideep_sglang:
    base_framework: sglang
    collector_dir: "collector/wideep/sglang"
    default:
      version: "0.5.10"
      images:
        default: "deepseek-v4-blackwell"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data_backend"):
        get_collector_runtime("wideep_sglang", path=manifest)


WIDEEP_OPS = {entry.op for entry in WIDEEP_SGLANG_REGISTRY}


@pytest.mark.parametrize(
    ("installed_version", "requested_ops", "workload", "version"),
    [
        # "all ops" is no longer resolvable in one container for sglang — the
        # kda family pins the kimi-k3 branch runtime (0.5.16), so the default
        # expectation is asserted on an explicit default-family op instead.
        ("0.5.14+cu130", {"gemm"}, "default", "0.5.14"),
        ("0.5.16", {"kda"}, "default", "0.5.16"),
        ("0.5.10", {"moe_ep"}, "wideep", "0.5.10"),
    ],
)
def test_runtime_selection_accepts_only_the_matching_pin(installed_version, requested_ops, workload, version):
    runtime = require_collector_runtime("sglang", installed_version, requested_ops=requested_ops, wideep_ops=WIDEEP_OPS)
    assert (runtime.workload, runtime.version) == (workload, version)


@pytest.mark.parametrize(
    ("installed_version", "requested_ops", "match"),
    [
        ("0.5.13", {"gemm"}, r"stock collector requires exactly 0\.5\.14"),
        ("0.5.14rc1", {"gemm"}, r"stock collector requires exactly 0\.5\.14"),
        ("0.5.14.post1", {"gemm"}, r"stock collector requires exactly 0\.5\.14"),
        ("0.5.14", {"moe_ep"}, r"WideEP collector requires exactly 0\.5\.10"),
        ("0.5.14", {"gemm", "moe_ep"}, r"0\.5\.14 != 0\.5\.10.*separate containers"),
        # kda runs only on the kimi-k3 branch runtime (families.kda pin):
        # mixing it with a default-family op must fail closed.
        ("0.5.14", {"gemm", "kda"}, r"multiple runtime versions"),
    ],
)
def test_runtime_selection_rejects_mismatched_or_mixed_pins(installed_version, requested_ops, match):
    with pytest.raises(RuntimeError, match=match):
        require_collector_runtime("sglang", installed_version, requested_ops=requested_ops, wideep_ops=WIDEEP_OPS)


# --- Task 4b: model-scoped runtime pins (frameworks.<key>.models) ---------


@pytest.mark.parametrize(
    ("installed_version", "requested_ops", "workload", "version"),
    [
        ("0.5.14+cu130", {"gemm"}, "default", "0.5.14"),
        ("0.5.16", {"kda"}, "default", "0.5.16"),
        ("0.5.10", {"moe_ep"}, "wideep", "0.5.10"),
    ],
)
def test_no_model_identity_resolves_exactly_like_today(installed_version, requested_ops, workload, version):
    # model_path is purely additive: omitting it, or passing None explicitly,
    # must reproduce pre-4b resolution byte-for-byte across default, family,
    # and wideep pins (CollectorRuntime is a frozen dataclass, so == is a
    # full field comparison).
    baseline = require_collector_runtime(
        "sglang", installed_version, requested_ops=requested_ops, wideep_ops=WIDEEP_OPS
    )
    explicit_none = require_collector_runtime(
        "sglang", installed_version, requested_ops=requested_ops, wideep_ops=WIDEEP_OPS, model_path=None
    )
    assert explicit_none == baseline
    assert (explicit_none.workload, explicit_none.version) == (workload, version)


def test_unknown_model_id_falls_back_to_default_resolution():
    baseline = require_collector_runtime("sglang", "0.5.14", requested_ops={"gemm"}, wideep_ops=WIDEEP_OPS)
    unmatched = require_collector_runtime(
        "sglang",
        "0.5.14",
        requested_ops={"gemm"},
        wideep_ops=WIDEEP_OPS,
        model_path="some-org/not-a-pinned-model",
    )
    assert unmatched == baseline


@pytest.mark.parametrize(
    "model_path",
    [
        "Qwen/Qwen3.8-2.4T-A95B",
        "Qwen/Qwen3.8-2.4T-A95B-FP8",
        "RadixArk/Qwen3.8-2.4T-A95B-NVFP4",
    ],
)
def test_model_pin_match_resolves_qwen38_max_to_sglang_0_5_17(model_path):
    # Real manifest (no path= override): the checked-in frameworks.sglang.models
    # entries added for Qwen3.8-Max day-0 support.
    runtime = require_collector_runtime(
        "sglang", "0.5.17", requested_ops={"gemm"}, wideep_ops=WIDEEP_OPS, model_path=model_path
    )
    assert runtime.version == "0.5.17"
    assert runtime.image().startswith("lmsysorg/sglang:v0.5.17@sha256:")
    assert runtime.image("cu130").startswith("lmsysorg/sglang:v0.5.17-cu130@sha256:")
    # Not a family classification — see _model_pinned_runtime docstring.
    assert runtime.family is None


def test_model_pin_mismatch_error_names_the_model_scoped_image():
    with pytest.raises(RuntimeError) as excinfo:
        require_collector_runtime(
            "sglang",
            "0.5.14",
            requested_ops={"gemm"},
            wideep_ops=WIDEEP_OPS,
            model_path="Qwen/Qwen3.8-2.4T-A95B",
        )
    message = str(excinfo.value)
    # Same template as the pre-4b guard ("~:249"), but naming the
    # model-scoped runtime/image instead of the framework default.
    assert "sglang stock collector requires exactly 0.5.17, found 0.5.14" in message
    assert "use lmsysorg/sglang:v0.5.17@sha256:" in message


# --- Task V1 (AIC-1782): vLLM model-scoped runtime pins, same mechanism ---


@pytest.mark.parametrize(
    "model_path",
    [
        "Qwen/Qwen3.8-2.4T-A95B",
        "Qwen/Qwen3.8-2.4T-A95B-FP8",
        "RadixArk/Qwen3.8-2.4T-A95B-NVFP4",
    ],
)
def test_model_pin_match_resolves_qwen38_max_to_vllm_0_27_1(model_path):
    # Real manifest (no path= override): the checked-in frameworks.vllm.models
    # entries added for Qwen3.8-Max (AIC-1782 Task V1), mirroring the sglang
    # Task 4b mechanism above. vllm has no wideep_vllm framework key, so
    # wideep_ops is empty here (unlike the sglang WIDEEP_OPS parametrization).
    runtime = require_collector_runtime(
        "vllm", "0.27.1", requested_ops={"gemm"}, wideep_ops=set(), model_path=model_path
    )
    assert runtime.version == "0.27.1"
    assert runtime.image().startswith("vllm/vllm-openai:v0.27.1@sha256:")
    assert runtime.image("cu129").startswith("vllm/vllm-openai:v0.27.1-cu129@sha256:")
    # Not a family classification — see _model_pinned_runtime docstring.
    assert runtime.family is None


def test_vllm_model_pin_mismatch_error_names_the_model_scoped_image():
    with pytest.raises(RuntimeError) as excinfo:
        require_collector_runtime(
            "vllm",
            "0.24.0",
            requested_ops={"gemm"},
            wideep_ops=set(),
            model_path="Qwen/Qwen3.8-2.4T-A95B",
        )
    message = str(excinfo.value)
    # Same template as the sglang guard above, naming the vllm model-scoped
    # runtime/image instead of the framework default.
    assert "vllm stock collector requires exactly 0.27.1, found 0.24.0" in message
    assert "use vllm/vllm-openai:v0.27.1@sha256:" in message


def test_vllm_unknown_model_id_falls_back_to_default_resolution():
    baseline = require_collector_runtime("vllm", "0.24.0", requested_ops={"gemm"}, wideep_ops=set())
    unmatched = require_collector_runtime(
        "vllm",
        "0.24.0",
        requested_ops={"gemm"},
        wideep_ops=set(),
        model_path="some-org/not-a-pinned-model",
    )
    assert unmatched == baseline


def test_real_manifest_models_section_does_not_break_validate_resolution():
    # Task 4b spec: model pins are additive to validate_resolution()'s
    # contract ("every registry op resolves to a pinned runtime"); adding
    # frameworks.sglang.models must not introduce a resolution error anywhere
    # in the real manifest.
    assert validate_resolution() == []


def test_unknown_requested_op_fails_with_key_error():
    with pytest.raises(KeyError, match=r"has no op\(s\): \['not_a_real_op'\]"):
        require_collector_runtime("sglang", "0.5.14", requested_ops={"not_a_real_op"}, wideep_ops=set())


def test_typo_mixed_with_real_op_fails_closed():
    # A typo must not be silently dropped just because another requested op is valid.
    with pytest.raises(KeyError, match=r"has no op\(s\): \['not_a_real_op'\]"):
        require_collector_runtime("sglang", "0.5.14", requested_ops={"gemm", "not_a_real_op"}, wideep_ops=set())


def test_wideep_registry_entries_are_separate_from_stock_backend_registries():
    sglang_modules = {entry.op: entry.module for entry in SGLANG_REGISTRY}
    trtllm_modules = {entry.op: entry.module for entry in TRTLLM_REGISTRY}
    wideep_sglang_modules = {entry.op: entry.module for entry in WIDEEP_SGLANG_REGISTRY}
    wideep_trtllm_modules = {entry.op: entry.module for entry in WIDEEP_TRTLLM_REGISTRY}

    assert "wideep_mla_context" not in sglang_modules
    assert "wideep_mla_generation" not in sglang_modules
    assert "moe_ep" not in sglang_modules
    assert "moe_ep" not in trtllm_modules
    assert "wideep_mla_context" not in wideep_sglang_modules
    assert "wideep_mla_generation" not in wideep_sglang_modules
    assert wideep_sglang_modules["moe_ep"].startswith("collector.wideep.sglang.")
    assert wideep_trtllm_modules["moe_ep"].startswith("collector.wideep.trtllm.")


def test_deepep_collectors_live_under_wideep_namespace():
    assert (COLLECTOR_ROOT / "wideep" / "sglang" / "collect_deepep_moe.py").exists()
    assert (COLLECTOR_ROOT / "wideep" / "sglang" / "deepep" / "extract_data.py").exists()
    assert (COLLECTOR_ROOT / "wideep" / "trtllm" / "collect_moe_compute.py").exists()

    assert not (COLLECTOR_ROOT / "deep_collector").exists()
    assert not (COLLECTOR_ROOT / "sglang" / "collect_wideep_deepep_moe.py").exists()
    assert not (COLLECTOR_ROOT / "trtllm" / "collect_wideep_moe_compute.py").exists()


def test_retired_wideep_mla_shim_stays_gone():
    # collector/wideep/sglang/collect_mla_module.py was a pure re-export shim
    # over collector.sglang.collect_mla_module with zero importers; retired in
    # the moe_a2a/moe_ep registration change. It must not come back, and no
    # registry or hash-closure entry may reference it.
    retired_module = "collector.wideep.sglang.collect_mla_module"
    assert not (COLLECTOR_ROOT / "wideep" / "sglang" / "collect_mla_module.py").exists()

    for registry in (
        SGLANG_REGISTRY,
        TRTLLM_REGISTRY,
        VLLM_REGISTRY,
        WIDEEP_SGLANG_REGISTRY,
        WIDEEP_TRTLLM_REGISTRY,
    ):
        for entry in registry:
            assert entry.module != retired_module
            assert all(route.module != retired_module for route in entry.versions)

    closures = yaml.safe_load((COLLECTOR_ROOT / "hash_closures.yaml").read_text(encoding="utf-8"))
    assert retired_module not in closures


def test_family_overrides_split_ops_across_runtimes(tmp_path):
    digest = "@sha256:" + "0" * 64
    (tmp_path / "framework_manifest.yaml").write_text(
        f"""
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14{digest}"
    families:
      gemm:
        version: "0.5.15"
        images:
          default: "lmsysorg/sglang:v0.5.15{digest}"
""",
        encoding="utf-8",
    )
    (tmp_path / "op_backend_catalog.yaml").write_text(
        """
schema_version: 1
families:
  - family: gemm
    op_files: [gemm_perf]
  - family: attention
    op_files: [context_attention_perf, generation_attention_perf]
""",
        encoding="utf-8",
    )
    # One container cannot serve two pins: fail closed with the op->version split.
    with pytest.raises(RuntimeError, match="multiple runtime versions"):
        require_collector_runtime(
            "sglang",
            "0.5.14",
            requested_ops={"gemm", "attention_context"},
            wideep_ops=set(),
            path=tmp_path / "framework_manifest.yaml",
            catalog_path=tmp_path / "op_backend_catalog.yaml",
        )
    # A single-family request against the matching container succeeds.
    runtime = require_collector_runtime(
        "sglang",
        "0.5.15",
        requested_ops={"gemm"},
        wideep_ops=set(),
        path=tmp_path / "framework_manifest.yaml",
        catalog_path=tmp_path / "op_backend_catalog.yaml",
    )
    assert (runtime.family, runtime.version) == ("gemm", "0.5.15")


def test_family_override_same_version_different_image_is_rejected(tmp_path):
    digest_a = "@sha256:" + "a" * 64
    digest_b = "@sha256:" + "b" * 64
    (tmp_path / "framework_manifest.yaml").write_text(
        f"""
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14{digest_a}"
    families:
      gemm:
        version: "0.5.14"
        images:
          default: "lmsysorg/sglang:v0.5.14-gemm{digest_b}"
""",
        encoding="utf-8",
    )
    (tmp_path / "op_backend_catalog.yaml").write_text(
        """
schema_version: 1
families:
  - family: gemm
    op_files: [gemm_perf]
  - family: attention
    op_files: [context_attention_perf, generation_attention_perf]
""",
        encoding="utf-8",
    )
    # Runtime identity is (version, images), not version alone: the same package
    # version pinned to two different images is still two containers, so a mixed
    # request must fail closed with the op->runtime split instead of letting
    # registry order pick one image silently.
    with pytest.raises(RuntimeError) as excinfo:
        require_collector_runtime(
            "sglang",
            "0.5.14",
            requested_ops={"gemm", "attention_context"},
            wideep_ops=set(),
            path=tmp_path / "framework_manifest.yaml",
            catalog_path=tmp_path / "op_backend_catalog.yaml",
        )
    message = str(excinfo.value)
    assert "same runtime version but different images" in message
    assert f"gemm→0.5.14 [default=lmsysorg/sglang:v0.5.14-gemm{digest_b}]" in message
    assert f"attention_context→0.5.14 [default=lmsysorg/sglang:v0.5.14{digest_a}]" in message
    # Each image group alone is still a valid single-container request.
    runtime = require_collector_runtime(
        "sglang",
        "0.5.14",
        requested_ops={"gemm"},
        wideep_ops=set(),
        path=tmp_path / "framework_manifest.yaml",
        catalog_path=tmp_path / "op_backend_catalog.yaml",
    )
    assert (runtime.family, runtime.version) == ("gemm", "0.5.14")
    assert runtime.image() == f"lmsysorg/sglang:v0.5.14-gemm{digest_b}"


def test_stock_and_wideep_same_version_different_image_is_rejected(tmp_path):
    digest_a = "@sha256:" + "a" * 64
    digest_b = "@sha256:" + "b" * 64
    (tmp_path / "framework_manifest.yaml").write_text(
        f"""
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14{digest_a}"
  wideep_sglang:
    base_framework: sglang
    collector_dir: "collector/wideep/sglang"
    data_backend: "sglang"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14-wideep{digest_b}"
""",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError) as excinfo:
        require_collector_runtime(
            "sglang",
            "0.5.14",
            requested_ops={"gemm", "moe_ep"},
            wideep_ops={"moe_ep"},
            path=tmp_path / "framework_manifest.yaml",
        )
    message = str(excinfo.value)
    assert "different images for the same runtime version" in message
    assert f"lmsysorg/sglang:v0.5.14{digest_a}" in message
    assert f"lmsysorg/sglang:v0.5.14-wideep{digest_b}" in message
    assert "separate containers" in message


def test_model_pin_overrides_a_synthetic_family_pin_for_the_same_op(tmp_path):
    # Task 4b precedence: a model pin wins over a family pin for every op in
    # the run, even one the family pin would otherwise claim — mirrors "a
    # hypothetical kda under a model pin" from the framework_manifest.yaml
    # comment. Synthetic fixture; the real kda entry is untouched.
    digest = "@sha256:" + "0" * 64
    model_digest = "@sha256:" + "1" * 64
    (tmp_path / "framework_manifest.yaml").write_text(
        f"""
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14{digest}"
    families:
      gemm:
        version: "0.5.15"
        images:
          default: "lmsysorg/sglang:v0.5.15{digest}"
    models:
      "some-org/pinned-model":
        version: "0.5.17"
        images:
          default: "lmsysorg/sglang:v0.5.17{model_digest}"
""",
        encoding="utf-8",
    )
    (tmp_path / "op_backend_catalog.yaml").write_text(
        """
schema_version: 1
families:
  - family: gemm
    op_files: [gemm_perf]
  - family: attention
    op_files: [context_attention_perf, generation_attention_perf]
""",
        encoding="utf-8",
    )
    # Without a model pin: gemm (family override, 0.5.15) and
    # attention_context (default, 0.5.14) split across two runtimes and fail
    # closed, exactly as today.
    with pytest.raises(RuntimeError, match="multiple runtime versions"):
        require_collector_runtime(
            "sglang",
            "0.5.14",
            requested_ops={"gemm", "attention_context"},
            wideep_ops=set(),
            path=tmp_path / "framework_manifest.yaml",
            catalog_path=tmp_path / "op_backend_catalog.yaml",
        )
    # With the model pin active for this run: the gemm family override no
    # longer applies — both ops resolve uniformly to the model-pinned runtime.
    runtime = require_collector_runtime(
        "sglang",
        "0.5.17",
        requested_ops={"gemm", "attention_context"},
        wideep_ops=set(),
        model_path="some-org/pinned-model",
        path=tmp_path / "framework_manifest.yaml",
        catalog_path=tmp_path / "op_backend_catalog.yaml",
    )
    assert runtime.version == "0.5.17"
    assert runtime.image() == f"lmsysorg/sglang:v0.5.17{model_digest}"
    assert runtime.family is None


def test_model_pin_images_must_be_digest_pinned(tmp_path):
    digest = "@sha256:" + "0" * 64
    manifest = tmp_path / "framework_manifest.yaml"
    manifest.write_text(
        f"""
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14{digest}"
    models:
      "some-org/pinned-model":
        version: "0.5.17"
        images:
          default: "lmsysorg/sglang:v0.5.17"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digest-pinned"):
        get_collector_runtime("sglang", path=manifest)
