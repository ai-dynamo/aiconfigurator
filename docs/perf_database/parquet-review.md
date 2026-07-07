# Reviewing Perf Parquet Changes

AIC stores perf tables as compressed parquet files. GitHub cannot render useful
binary diffs for parquet by itself, so use the review helpers in
`tools/perf_database`.

For local diffs, install the textconv driver once per clone:

```bash
git config diff.parquet.textconv 'uv run python tools/perf_database/parquet_textconv.py'
```

After that, regular git commands show parquet files as CSV-like text:

```bash
git diff origin/main...HEAD -- packages/aiconfigurator-core/src/aiconfigurator_core/systems/data
```

For PR review summaries, run:

```bash
uv run python tools/perf_database/parquet_diff.py \
  --base-ref origin/main \
  --head-ref HEAD \
  --output parquet-diff.md \
  --detail-dir parquet-diff-details
```

The summary checks row counts, column names, and Arrow table content hashes.
When a PR replaces `*_perf.txt` with `*.parquet`, the tool compares the new
parquet file against the base branch's legacy text file. A GitHub workflow also
uploads and comments this report on PRs that touch perf data. The artifact
bundle includes `parquet-diff-details/changed-files.csv`, full unified diffs for
every changed perf data file under `parquet-diff-details/diffs/`, and
`parquet-diff-details/summary.csv` with row-level CSV details when the tool can
classify added, removed, or modified rows.
