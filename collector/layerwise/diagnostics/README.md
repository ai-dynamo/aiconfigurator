# Layerwise Diagnostics

This directory contains standalone helper tools for inspecting traces and
debugging collector behavior. Files here are not part of the normal layerwise
collection path.

- `analyze_nsys_comm_overlap.py`: inspect compute/communication overlap from a
  layerwise Nsight sqlite export.
- `compare_layerwise_fpm.py`: compare layerwise CSV rows against FPM phase CSV
  rows while explicitly reporting exact, nearest, or explicit pooled decode
  KV-window matches.
- `plot_aic_fpm_comparison.py`: plot existing AIC-vs-FPM comparison CSVs,
  including mixed-step context-token and decode-token histograms.
