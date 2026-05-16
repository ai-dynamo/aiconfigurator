# WideEP Collectors

WideEP collectors live under this namespace so tooling can choose the right
runtime image separately from the normal framework collectors.

The authoritative framework versions and collector images are in
`collector/framework_manifest.json`. WideEP entries must use the same framework
version as their non-WideEP framework entry.
