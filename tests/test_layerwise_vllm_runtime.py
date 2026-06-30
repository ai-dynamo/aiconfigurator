from pathlib import Path
from types import SimpleNamespace

from collector.layerwise.common import config_patch
from collector.layerwise.vllm import runtime


def test_system_and_vllm_probe_results_are_cached(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(tuple(cmd))
        if cmd[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="NVIDIA B300 SXM6 AC\n")
        return SimpleNamespace(returncode=0, stdout="0.20.1\n")

    monkeypatch.setattr(runtime, "_SYSTEM_NAME_CACHE", None)
    monkeypatch.setattr(runtime, "_VLLM_VERSION_CACHE", None)
    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    assert runtime._get_system_name() == "NVIDIA B300 SXM6 AC"
    assert runtime._get_system_name() == "NVIDIA B300 SXM6 AC"
    assert runtime._get_vllm_version() == "0.20.1"
    assert runtime._get_vllm_version() == "0.20.1"

    assert calls == [
        ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"),
        (runtime.sys.executable, "-c", "import vllm; print(getattr(vllm, '__version__', 'unknown'))"),
    ]


def test_default_max_num_seqs_probe_result_is_cached(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(tuple(cmd))
        return SimpleNamespace(returncode=0, stdout="1024\n")

    monkeypatch.setattr(runtime, "_VLLM_DEFAULT_MAX_NUM_SEQS_CACHE", {})
    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    assert runtime._get_vllm_default_max_num_seqs(world_size=8) == 1024
    assert runtime._get_vllm_default_max_num_seqs(world_size=8) == 1024

    assert len(calls) == 1


def test_deployment_effective_config_uses_helper_and_cache(monkeypatch) -> None:
    runtime._DEPLOYMENT_EFFECTIVE_CONFIG_CACHE.clear()
    calls = []

    def fake_helper(common_dir: Path, payload: dict) -> dict:
        calls.append((common_dir, payload))
        return {"scheduler_config.max_num_batched_tokens": 2048}

    def fail_once(common_dir: Path, payload: dict) -> dict:
        raise AssertionError("fallback one-shot resolver should not be used")

    monkeypatch.setattr(runtime, "_query_deployment_config_helper", fake_helper)
    monkeypatch.setattr(runtime, "_resolve_deployment_effective_config_once", fail_once)

    first = runtime._get_vllm_deployment_effective_config(
        model="model-a",
        tensor_parallel_size=4,
        max_num_seqs=128,
        extra_args=("--kv-cache-dtype", "fp8"),
    )
    second = runtime._get_vllm_deployment_effective_config(
        model="model-a",
        tensor_parallel_size=4,
        max_num_seqs=128,
        extra_args=("--kv-cache-dtype", "fp8"),
    )

    assert first == {"scheduler_config.max_num_batched_tokens": 2048}
    assert second == first
    assert len(calls) == 1
    assert calls[0][1]["extra_args"] == ("--kv-cache-dtype", "fp8", "--nnodes", "4")


def test_deployment_effective_config_falls_back_to_one_shot(monkeypatch) -> None:
    runtime._DEPLOYMENT_EFFECTIVE_CONFIG_CACHE.clear()
    fallback_calls = []

    monkeypatch.setattr(runtime, "_query_deployment_config_helper", lambda common_dir, payload: None)

    def fake_once(common_dir: Path, payload: dict) -> dict:
        fallback_calls.append((common_dir, payload))
        return {
            "scheduler_config.max_num_batched_tokens": 1024,
            "cache_config.block_size": 2048,
            "cache_config.mamba_cache_mode": "align",
        }

    monkeypatch.setattr(runtime, "_resolve_deployment_effective_config_once", fake_once)

    value = runtime._get_vllm_deployment_max_num_batched_tokens(
        model="model-b",
        tensor_parallel_size=1,
        extra_args=("--foo", "bar"),
    )

    assert value == 2048
    assert len(fallback_calls) == 1
    assert fallback_calls[0][1]["extra_args"] == ("--foo", "bar")


def test_patch_model_path_symlinks_aux_files_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(config_patch._COPY_AUX_FILES_ENV, raising=False)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"test","num_hidden_layers":2}')
    (source / "tokenizer.json").write_text('{"tokens":[]}')
    (source / "model.safetensors").write_text("weights")

    patched = Path(
        config_patch.patch_model_path(
            str(source),
            overrides={"num_hidden_layers": 1},
            cache_dir=str(tmp_path / "cache"),
        )
    )

    aux = patched / "tokenizer.json"
    assert aux.is_symlink()
    assert aux.resolve() == source / "tokenizer.json"
    assert not (patched / "model.safetensors").exists()
    assert (patched / ".complete").exists()


def test_patch_model_path_can_copy_aux_files_when_requested(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(config_patch._COPY_AUX_FILES_ENV, "1")
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"test","num_hidden_layers":2}')
    (source / "tokenizer.json").write_text('{"tokens":[]}')

    patched = Path(
        config_patch.patch_model_path(
            str(source),
            overrides={"num_hidden_layers": 1},
            cache_dir=str(tmp_path / "cache"),
        )
    )

    aux = patched / "tokenizer.json"
    assert aux.exists()
    assert not aux.is_symlink()
