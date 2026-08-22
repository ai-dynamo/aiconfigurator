import pytest as _pytest


@_pytest.fixture(autouse=True)
def _allow_unlisted_versions_for_legacy_task_fixtures(monkeypatch):
    """These suites pin version-specific data coordinates through the Task
    surface (no allow parameter there). Transition escape until the
    fixture-discipline follow-up re-pins them onto version slots."""
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
