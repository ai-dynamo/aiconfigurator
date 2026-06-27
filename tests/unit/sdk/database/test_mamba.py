# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk.operations.mamba import load_mamba2_data

pytestmark = pytest.mark.unit


def test_mamba2_generation_loader_inserts_batch_only_first_wins_key(tmp_path):
    perf_file = tmp_path / "mamba2_perf.txt"
    perf_file.write_text(
        "framework,version,device,op_name,kernel_source,phase,batch_size,seq_len,num_tokens,"
        "d_model,d_state,d_conv,nheads,head_dim,n_groups,chunk_size,model_name,latency\n"
        "TRTLLM,test,GPU,mamba2,causal_conv1d_update,generation,4,1,4,"
        "4096,128,4,128,64,8,128,NEMOTRON_H_3_Super,1.25\n"
        "TRTLLM,test,GPU,mamba2,causal_conv1d_update,generation,4,99,4,"
        "4096,128,4,128,64,8,128,NEMOTRON_H_3_Super,9.75\n",
        encoding="utf-8",
    )

    data = load_mamba2_data(str(perf_file))
    model_key = (4096, 128, 4, 128, 64, 8, 128)
    generation = data["causal_conv1d_update"]["generation"][model_key]

    assert set(generation) == {4}
    assert generation[4]["latency"] == 1.25
