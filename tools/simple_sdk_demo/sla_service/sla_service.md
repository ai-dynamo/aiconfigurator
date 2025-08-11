<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# clone the repo
git clone the repo

# install aiconfigurator follow readme including the required dependency

# launch the container
python3 tools/sla_service/sla_service.py --server_name 0.0.0.0 --server_port 7860

# access the service
```
curl -X 'POST' \
  'http://127.0.0.1:7860/sla' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "0.20.0",
    "model": "LLAMA2_7B",
    "ttft": 1000,
    "isl": 1024,
    "osl": 128,
    "tpot": 20,
    "hardware": "h200_sxm",
    "quant": "fp8",
    "kvcache_quant": "fp8"
  }'
```
get supported models
```
curl -X 'GET' \
  'http://127.0.0.1:7860/sla/supported_models' \
  -H 'accept: application/json'
```
