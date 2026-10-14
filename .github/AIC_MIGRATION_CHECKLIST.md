<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIC repository cutover checklist

This checklist covers transition actions that cannot be completed by changing
tracked repository files. Complete each action only at its approved release
milestone; do not imply that a replacement artifact is available before its
published release has been verified.

## Stable AISimulate launch

- [ ] Verify the stable `aisimulate` Python wheel and `aisimulate-core` Rust
      crate are published for every supported migration platform.
- [ ] Run clean-environment install and representative smoke tests against the
      exact published artifacts.
- [ ] Lead the AIC 0.12 release notes with the maintenance-only policy, the
      AISimulate successor link, and the migration guide; do not leave the
      transition buried in generated changelog entries.
- [ ] Update the AIC GitHub description to identify AISimulate as the active
      successor and AIC as maintenance-only.
- [ ] Pin transition DEP issue
      [#1517](https://github.com/ai-dynamo/aiconfigurator/issues/1517).
- [ ] Confirm the README installation gate matches actual PyPI and crates.io
      availability before recommending migration.

## AIC 0.13 compatibility closeout

- [ ] Publish the approved removal version and date alongside the supported
      compatibility matrix and rollback guidance.
- [ ] Triage every open AIC issue and pull request as migrated, compatibility
      work, or historical context.
- [ ] Archive the AIC repository at the approved 0.13 cutover milestone; do not
      delete repository history or yank historical artifacts solely because of
      the migration.
- [ ] Verify the archived README, security policy, documentation redirects,
      issue history, releases, and package/crate provenance remain accessible.
