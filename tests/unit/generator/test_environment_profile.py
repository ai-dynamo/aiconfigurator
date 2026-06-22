from pathlib import Path

import yaml

from aiconfigurator.generator.environment.schema import (
    ENVIRONMENT_JSON_SCHEMA,
    EnvironmentProfile,
    load_environment_profile,
)

_DEFAULT = Path(__file__).resolve().parents[3] / "src/aiconfigurator/generator/environment/default.yaml"


def test_default_profile_loads_and_validates():
    profile = load_environment_profile(str(_DEFAULT))
    assert isinstance(profile, EnvironmentProfile)
    assert profile.namespace
    assert profile.storage_class_name is not None


def test_json_schema_published_and_default_conforms():
    import jsonschema

    assert ENVIRONMENT_JSON_SCHEMA["type"] == "object"
    jsonschema.validate(yaml.safe_load(_DEFAULT.read_text()), ENVIRONMENT_JSON_SCHEMA)


def test_load_applies_defaults_for_missing_fields(tmp_path):
    # A minimal profile (just namespace) still loads, other fields fall back to defaults.
    p = tmp_path / "min.yaml"
    p.write_text("namespace: my-ns\n")
    profile = load_environment_profile(str(p))
    assert profile.namespace == "my-ns"
    assert profile.storage_class_name is not None  # default applied
