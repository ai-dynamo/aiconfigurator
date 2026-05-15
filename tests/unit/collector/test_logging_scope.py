import importlib.util
from pathlib import Path


def _import_helper_module():
    module_name = "collector.helper_logging_scope_test"
    helper_path = Path(__file__).resolve().parents[3] / "collector" / "helper.py"
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collector_log_dir_name_preserves_short_scope():
    helper = _import_helper_module()

    name = helper._collector_log_dir_name(["gemm", "moe"], "20260515_120000")

    assert name == "gemm+moe_20260515_120000"


def test_collector_log_dir_name_shortens_long_scope_deterministically():
    helper = _import_helper_module()
    scope = [
        "dsv4_flash_hca_context_module",
        "dsa_generation_module",
        "dsv4_flash_csa_generation_module",
        "wideep_mla_generation",
        "dsv4_flash_hca_generation_module",
        "wideep_mla_context",
        "wideep_moe",
        "dsv4_flash_csa_context_module",
        "mhc_module",
        "dsv4_flash_paged_mqa_logits_module",
        "dsv4_flash_hca_attn_module",
        "dsa_context_module",
    ]

    name = helper._collector_log_dir_name(scope, "20260515_120000")

    assert len(name) < 255
    assert name.endswith("_20260515_120000")
    assert name.startswith("dsv4_flash_hca_context_module+")
    assert name == helper._collector_log_dir_name(scope, "20260515_120000")
    assert name != f"{'+'.join(scope)}_20260515_120000"
