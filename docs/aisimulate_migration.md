<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Migrate from AIConfigurator to AISimulate

AIConfigurator 0.12.0 is the compatibility release for moving the application
distribution and Sweeper workflows to AISimulate. The established
`aiconfigurator` CLI command continues to work, while the legacy AIC
distribution and direct Sweeper entry points emit targeted migration warnings.

## Install the replacement

```bash
python -m pip uninstall -y aiconfigurator aiconfigurator-core
python -m pip install "aisimulate==0.12.0"
```

The `aisimulate` wheel contains the complete application and installs the
established `aiconfigurator` command. It does not install a second top-level
application command named `aisimulate`. The command name is independent of the
distribution name and does not represent a separately published AIC artifact.

## Keep CLI commands unchanged

Change the installed distribution, then keep the command and arguments for
your current mode:

```bash
# Before and after installing AISimulate 0.12
aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm
aiconfigurator cli recommend --model-path Qwen/Qwen3-32B-FP8 --system h200_sxm --target-request-rate 50
```

The `default`, `estimate`, `recommend`, `exp`, `generate`, and `support` modes
retain their current command paths. AISimulate becomes the package and source
owner without renaming the user-facing CLI.

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
