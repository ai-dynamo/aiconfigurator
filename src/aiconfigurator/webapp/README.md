<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run
After install aiconfigurator
```bash
 aiconfigurator webapp --server-name 0.0.0.0 --server-port 7860
```

Optional: override system YAML/data search paths:

```bash
aiconfigurator webapp --systems-paths "default,/opt/aic/systems,/data/aic/systems"
```