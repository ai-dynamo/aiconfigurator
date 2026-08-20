<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Migrate from AIConfigurator to AISimulate

AIConfigurator 0.12.0 is the compatibility release for moving CLI and Sweeper
workflows to AISimulate. Legacy AIC CLI and Sweeper entry points continue to
work during this window, emit targeted deprecation warnings, and are planned
for removal in AIConfigurator 0.13.0.

## Install the replacement

```bash
python -m pip uninstall -y aiconfigurator aiconfigurator-core
python -m pip install "aisimulate==0.12.0"
```

The `aisimulate` wheel contains the complete application and installs the
`aisimulate` command. During the 0.12 compatibility window it also retains the
`aiconfigurator` command and import namespace; these compatibility names do not
represent separately published AIC artifacts.

## Migrate CLI commands

Keep the arguments for your current mode and change the executable:

```bash
# Before
aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm
aiconfigurator cli recommend --model-path Qwen/Qwen3-32B-FP8 --system h200_sxm --target-request-rate 50

# AISimulate 0.12 replacement
aisimulate cli default --model-path Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm
aisimulate cli recommend --model-path Qwen/Qwen3-32B-FP8 --system h200_sxm --target-request-rate 50
```

The same executable substitution applies to the `estimate`, `exp`, `generate`,
and `support` modes. In 0.12, both executable names deliberately route to the
same proven implementation. The future `predict` and redesigned `recommend`
commands will not replace that implementation until their parity and product
gates pass.

## Migrate Sweeper code

Replace direct calls to `aiconfigurator.sdk.sweep.sweep_agg`,
`sweep_disagg`, or `sweep_afd` with the AISimulate Sweeper and an explicit
runner:

```python
from aisimulate import EngineReplayRunnerFactory
from aisimulate.sweeper import SmartSearchConfig, Sweeper

config = SmartSearchConfig.from_yaml("smart_sweep.yaml")
candidates = Sweeper(
    runner_factory=EngineReplayRunnerFactory(),
).run(config)
```

Use a Dynamo runner factory instead when the configuration selects Dynamo
Router or Planner adapters. See the
[AISimulate Sweeper documentation](https://github.com/ai-dynamo/aisimulate/tree/main/docs/sweeper)
for configuration, runner, result, and adapter guidance.

## Temporary compatibility

Code installed from `aisimulate==0.12.0` may temporarily continue importing
the migrated AIC namespace while it moves to the supported AISimulate APIs.
Do not use that compatibility namespace for new integrations: it is retained
only to make the one-release migration window non-breaking.
