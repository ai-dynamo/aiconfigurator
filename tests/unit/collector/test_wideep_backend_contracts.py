# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from collector.wideep.backend_contracts import DEEPEP_COLLECTOR_CONTRACTS, contract_for

pytestmark = pytest.mark.unit


def test_backend_labels_are_framework_isolated_and_complete():
    assert set(DEEPEP_COLLECTOR_CONTRACTS) == {
        "deepep_ht",
        "deepep_ll",
        "deepep_v2",
        "trtllm_deepep_ht",
        "trtllm_deepep_ll",
    }
    assert contract_for("vllm", "deepep_ht").selector.endswith("deepep_high_throughput")
    assert contract_for("vllm", "deepep_ll").inference_phases == ("generation",)
    assert contract_for("vllm", "deepep_v2").inference_phases == ("context", "generation")
    assert contract_for("trtllm", "trtllm_deepep_ht").sms_policy == "fixed:0"
    assert contract_for("trtllm", "trtllm_deepep_ll").capacity_policy == "TRTLLM_DEEP_EP_TOKEN_LIMIT"


def test_framework_mismatch_cannot_relabel_rows():
    with pytest.raises(ValueError, match="cross-framework latency reuse is forbidden"):
        contract_for("trtllm", "deepep_ht")
    with pytest.raises(ValueError, match="cross-framework latency reuse is forbidden"):
        contract_for("vllm", "trtllm_deepep_ht")


def test_unknown_backend_fails_closed():
    with pytest.raises(KeyError, match="unknown DeepEP collector backend"):
        contract_for("vllm", "deepep")


def test_dtype_and_sms_contracts_match_serving_paths():
    assert contract_for("vllm", "deepep_ht").sms_policy == "fixed:20"
    assert contract_for("vllm", "deepep_ll").sms_policy == "fixed:0"
    assert contract_for("vllm", "deepep_v2").sms_policy == "ElasticBuffer.get_theoretical_num_sms"
    assert contract_for("trtllm", "trtllm_deepep_ht").comm_dtypes == ("bfloat16", "nvfp4")
    assert contract_for("trtllm", "trtllm_deepep_ll").comm_dtypes == (
        "bfloat16",
        "fp8",
        "nvfp4",
        "w4afp8",
        "fp4",
    )
