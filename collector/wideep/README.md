# WideEP Collectors

WideEP collectors live under this namespace so tooling can choose the right
runtime image separately from the normal framework collectors.

Each supported framework owns a WideEP-only `registry.py`. Normal framework
registries stay free of WideEP ops; `collect.py` appends a WideEP registry only
when the collector-v2 plan or explicit `--ops` requests those ops.

The authoritative framework versions and collector images are in
`collector/framework_manifest.json`. WideEP entries must use the same framework
version as their non-WideEP framework entry.
