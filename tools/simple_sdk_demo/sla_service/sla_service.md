<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Web Service

This example demonstrates how to use the aiconfigurator SDK to build a simple SLA service.

NOTE: this script is not actively maintained and may contain bugs.

# Install aiconfigurator with extra dependencies

Install aiconfigurator with some extra packages that are needed for the service:

```bash
pip install aiconfigurator[service]
```

# Launch the Web Service

```bash
python3 tools/simple_sdk_demo/sla_service/sla_service.py --server_name 0.0.0.0 --server_port 7860
```

# Access the Service

Estimate performance:
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

Get a list of the supported models:
```
curl -X 'GET' \
  'http://127.0.0.1:7860/sla/supported_models' \
  -H 'accept: application/json'
```
