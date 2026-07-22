# Web App User Guide

AIConfigurator ships with an interactive Gradio web interface that provides the
same configuration exploration as the CLI, but with visual controls and plots.
The webapp is not a CLI subcommand -- launch it directly as a Python module:

```bash
# Install the webapp extra (Gradio and dependencies)
pip install 'aiconfigurator[webapp]'        # from PyPI
pip install -e '.[webapp]'               # editable / dev install

# Launch
python -m aiconfigurator.webapp.main
```

The app binds to `0.0.0.0:7860` by default (all network interfaces).
Access it locally at `http://localhost:7860`. For local-only access, use
`--server-name 127.0.0.1`.

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--server-name` | `0.0.0.0` | Bind address |
| `--server-port` | `7860` | Bind port |
| `--enable-agg` | off | Show the Agg tab (aggregated serving explorer) |
| `--enable-disagg-pd-ratio` | off | Show the Disagg PD Ratio tab |
| `--enable-profiling` | off | Show the Profiling tab |
| `--debug` | off | Enable debug-level logging |
| `--experimental` | off | Enable experimental features |
| `--systems-paths` | built-in | Override system data search paths (comma-separated; use `default` for the built-in path) |

Example with all optional tabs enabled:

```bash
python -m aiconfigurator.webapp.main \
  --enable-agg --enable-profiling --enable-disagg-pd-ratio \
  --server-port 8080
```

## Tabs

The webapp is organized into tabs. Some are always visible; others require a
flag to activate.

**Always visible:**

- **Readme** -- Embedded documentation and version info.
- **Static Estimation** -- Single-point static latency exploration (select model, system, parallelism, and batch size to see estimated TTFT/TPOT).
- **Agg(IFB) Pareto Estimation** -- Aggregated serving Pareto frontier search (analogous to `cli default` in agg mode).
- **Disaggregation Pareto Estimation** -- Disaggregated serving Pareto frontier search.
- **Pareto Comparison** -- Save and compare results from the Agg/Disagg Pareto tabs side by side.
- **Support Matrix** -- Interactive support matrix for model/system/backend combinations.

**Optional (flag-gated):**

- **Agg Estimation** (`--enable-agg`) -- Aggregated serving configuration explorer.
- **Disaggregation PD Ratio Analysis** (`--enable-disagg-pd-ratio`) -- Disaggregated prefill/decode ratio analysis.
- **Profiling** (`--enable-profiling`) -- Per-operation latency breakdown visualization (GEMM, attention, communication, MoE, etc.) for understanding where inference time is spent.
