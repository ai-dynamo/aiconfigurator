# AIConfigurator Core

`aiconfigurator-core` contains the standalone forward-pass latency estimator:
the canonical Python SDK under `aiconfigurator_core`, the Rust extension,
and the system/model data required by the estimator.

The package must build and run without the upper `aiconfigurator` package.
Operational tooling such as collectors and future auto-collectors belongs in
the upper `packages/aiconfigurator/` project.
