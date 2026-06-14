import yaml

from aiconfigurator.generator.emit import build_artifact_manifest
from aiconfigurator.generator.pipeline import run_pipeline
from tests.baseline.canary import CANARY_CASES


def test_manifest_indexes_artifacts_and_versions():
    case = CANARY_CASES[0]
    result = run_pipeline(case.params, case.backend, backend_version=case.backend_version)
    manifest_yaml = build_artifact_manifest(result, artifact_names=list(result.artifacts))
    manifest = yaml.safe_load(manifest_yaml)

    assert manifest["backend"]["name"] == case.backend
    assert manifest["backend"]["version"] == case.backend_version
    assert manifest["topology"]["mode"] == result.ir.mode
    indexed = {a["path"] for a in manifest["artifacts"]}
    assert indexed == set(result.artifacts)


def test_emit_writes_artifacts_and_manifest(tmp_path):
    case = CANARY_CASES[0]
    result = run_pipeline(case.params, case.backend, backend_version=case.backend_version)
    from aiconfigurator.generator.emit import emit
    emit(result, str(tmp_path))
    written = {p.name for p in tmp_path.iterdir() if p.is_file()}
    assert "artifact_manifest.yaml" in written
    for name in result.artifacts:
        # artifacts may include nested paths; check the top-level entry exists
        assert (tmp_path / name).exists()
