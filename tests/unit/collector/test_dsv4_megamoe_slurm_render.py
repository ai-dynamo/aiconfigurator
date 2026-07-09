# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

import pytest

from collector.sglang.dsv4_megamoe.render_slurm_job import render

pytestmark = pytest.mark.unit


def _args(**overrides):
    args = Namespace(
        job_name="aic-test-p-e8",
        account="coreai_tritoninference_triton3",
        partition="gb300",
        time_limit="02:00:00",
        system_name="gb300",
        ep_size=8,
        gpus_per_node=4,
        phase="context",
        remote_workdir="/lustre/aic/run/repo",
        output_path="/lustre/aic/run/results/p-e8",
        log_dir="/lustre/aic/run/logs/p-e8",
        container_image="lmsysorg/sglang-staging:deepseek-v4-grace-blackwell-dev",
        container_mounts="/lustre/aic/run:/lustre/aic/run,/lustre:/lustre",
        container_writable=True,
        exclusive=True,
        model_config="dsv4_pro",
        perf_file="dsv4_megamoe_module_perf.txt",
        prefill_tokens="8192",
        decode_tokens="1",
        distributions="balanced,power_law_sampled_1.9",
        source_policy="random",
        routing_seed=0,
        routing_seeds="",
        pre_dispatch="sglang_jit",
        include_routed_scale=1,
        renormalize_topk_weights=1,
        num_warmup=5,
        num_iterations=20,
        num_max_tokens_per_rank=32768,
        cap_policy="fixed",
        env=[],
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_slurm_renderer_uses_one_task_per_gpu_and_megamoe_flags():
    script = render(_args())

    assert "#SBATCH --nodes=2\n" in script
    assert "#SBATCH --ntasks=8\n" in script
    assert "#SBATCH --ntasks-per-node=4\n" in script
    assert '--container-image="${CONTAINER_IMAGE}"' in script
    assert "--container-writable" in script
    assert "packages/aiconfigurator/collector/sglang/collect_dsv4_megamoe.py" in script
    assert '--routing-seeds "${ROUTING_SEEDS}"' in script
    assert '--sglang-version "${SGLANG_VERSION}"' in script
    assert "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE" in script
    assert 'export RANK="${SLURM_PROCID:?}"' in script
    assert 'export WORLD_SIZE="${SLURM_NTASKS:?}"' in script
    assert 'export LOCAL_RANK="${SLURM_LOCALID:-0}"' in script
    assert "_write_rank0_env" in script
    assert 'env | sort >"${OUTPUT_PATH}/rank0_env.txt"' not in script


def test_slurm_renderer_rejects_ep_not_divisible_by_node_size():
    with pytest.raises(SystemExit, match="--ep-size must be divisible"):
        render(_args(ep_size=6))
