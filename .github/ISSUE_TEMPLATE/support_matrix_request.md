---
name: AIC Compatibility Coverage Gap
about: Report a regression or migration blocker for previously supported AIC coverage
title: "[AIC Compatibility] Coverage gap for <model/hardware/framework>"
labels: support-matrix
---

> [!IMPORTANT]
> New model, hardware, and framework coverage belongs in
> [AISimulate](https://github.com/ai-dynamo/aisimulate/issues). AIConfigurator
> accepts only bug, security, and migration-blocking fixes during its
> maintenance-only compatibility window. Continue here only if this combination
> was already supported by AIC or the gap blocks migration to AISimulate.

## What existing coverage regressed or blocks migration?

**Model (HuggingFace ID):**
<!-- e.g. meta-llama/Llama-4-Scout-17B-16E-Instruct -->

**Hardware / System:**
<!-- e.g. B200_SXM, GB200, H100_SXM, A100_SXM -->

**Backend / Framework:**
<!-- e.g. vllm, sglang, trtllm -->

**Backend Version (if specific):**
<!-- e.g. vllm 0.14.0, trtllm 1.2.0rc6 -->

## Mode

- [ ] Aggregated (agg)
- [ ] Disaggregated (disagg)
- [ ] Both

## Additional context

<!-- Include the last working AIC version and evidence that the combination was supported, or explain exactly how the gap blocks AISimulate migration. -->
