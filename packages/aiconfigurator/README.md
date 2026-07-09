# AIConfigurator

`aiconfigurator` contains service-level simulation, configuration search,
deployment generation, CLI, web application, Spica integration, and collector
tooling. It depends on `aiconfigurator-core` for forward-pass latency and memory
estimation.

The collector lives under `collector/`. Future automated collection tooling
also belongs in this upper project rather than in `aiconfigurator-core`.

Historical `aiconfigurator.sdk` imports remain available as compatibility
facades for public APIs implemented by `aiconfigurator_core.sdk`.
