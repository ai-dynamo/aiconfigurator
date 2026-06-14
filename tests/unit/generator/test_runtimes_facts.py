from pathlib import Path

from aiconfigurator.generator.utils import load_backend_version_matrix

_OLD = (
    Path(__file__).resolve().parents[3]
    / "src/aiconfigurator/generator/config/backend_version_matrix.yaml"
)
_NEW = (
    Path(__file__).resolve().parents[3]
    / "src/aiconfigurator/generator/facts/runtimes/dynamo.yaml"
)


def test_runtimes_facts_equal_legacy_matrix():
    assert load_backend_version_matrix(str(_NEW)) == load_backend_version_matrix(str(_OLD))


from aiconfigurator.generator.utils import get_default_dynamo_version_mapping


def test_default_mapping_unchanged_after_reroute():
    # get_default_dynamo_version_mapping() returns (dynamo_version, entry_dict) for
    # the first entry in the matrix.  Confirm the default path now resolves to the
    # same first entry that the legacy file contains.
    dynamo_ver, entry = get_default_dynamo_version_mapping()
    legacy_matrix = load_backend_version_matrix(str(_OLD))
    assert dynamo_ver in legacy_matrix
    assert entry == legacy_matrix[dynamo_ver]
