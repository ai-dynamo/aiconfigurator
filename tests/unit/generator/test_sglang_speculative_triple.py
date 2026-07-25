# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang speculative decoding must emit the FULL (num-steps, eagle-topk,
num-draft-tokens) triple.

SGLang auto-derives the triple only when ``--speculative-num-steps`` is None
(and asserts the other two are None in that case); emitting num-steps alone
leaves ``speculative_eagle_topk`` / ``speculative_num_draft_tokens`` unset and
crashes server startup. Chain MTP semantics: topk=1, draft = nextn+1, matching
the SDK's ``(nextn+1)`` decode-batch model.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from aiconfigurator.generator.api import generate_backend_artifacts

pytestmark = pytest.mark.unit

_BACKEND_VERSION = "0.5.11"

_PARAMS = {
    "ServiceConfig": {
        "model_path": "deepseek-ai/DeepSeek-V3",
        "served_model_path": "deepseek-ai/DeepSeek-V3",
        "served_model_name": "DeepSeek-V3",
        "include_frontend": True,
    },
    "K8sConfig": {"name_prefix": "test", "k8s_namespace": "default"},
    "DynConfig": {"mode": "agg"},
    "WorkerConfig": {"agg_workers": 1, "agg_gpus_per_worker": 8, "prefill_workers": 0, "decode_workers": 0},
    "NodeConfig": {"num_gpus_per_node": 8},
    "SlaConfig": {"isl": 1024, "osl": 256},
    "ModelConfig": {"is_moe": True, "prefix": 0, "nextn": 1},
    "BenchConfig": {},
    "params": {
        "agg": {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "moe_tensor_parallel_size": 1,
            "moe_expert_parallel_size": 8,
            "max_batch_size": 64,
            "max_num_tokens": 4096,
            "max_seq_len": 4096,
        }
    },
}


def _worker_args(artifacts):
    k8s = yaml.safe_load(artifacts["k8s_deploy.yaml"])
    services = k8s["spec"]["services"]
    worker = next(svc for name, svc in services.items() if name != "Frontend")
    return worker["extraPodSpec"]["mainContainer"]["args"]


def _render(nextn: int):
    params = copy.deepcopy(_PARAMS)
    params["ModelConfig"]["nextn"] = nextn
    return generate_backend_artifacts(
        params,
        "sglang",
        backend_version=_BACKEND_VERSION,
        deployment_target="dynamo-j2",
    )


def _flag_value(args: list[str], flag: str) -> str:
    joined = " ".join(args)
    assert flag in joined, f"{flag} missing from: {joined}"
    tokens = joined.split()
    return tokens[tokens.index(flag) + 1].strip('"')


def test_nextn_emits_full_triple():
    args = _worker_args(_render(nextn=1))
    assert _flag_value(args, "--speculative-algorithm") == "NEXTN"
    assert _flag_value(args, "--speculative-num-steps") == "1"
    assert _flag_value(args, "--speculative-eagle-topk") == "1"
    assert _flag_value(args, "--speculative-num-draft-tokens") == "2"


def test_nextn3_draft_tokens_track_nextn():
    args = _worker_args(_render(nextn=3))
    assert _flag_value(args, "--speculative-num-steps") == "3"
    assert _flag_value(args, "--speculative-eagle-topk") == "1"
    assert _flag_value(args, "--speculative-num-draft-tokens") == "4"


def test_nextn_zero_emits_no_speculative_flags():
    args = _worker_args(_render(nextn=0))
    joined = " ".join(args)
    assert "--speculative" not in joined
