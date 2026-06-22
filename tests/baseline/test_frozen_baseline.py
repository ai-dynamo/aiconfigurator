import pytest

from tests.baseline.canary import CANARY_CASES
from tests.baseline.freeze_baseline import FROZEN_DIR, generate_case


@pytest.mark.parametrize("case", CANARY_CASES, ids=lambda c: c.name)
def test_generation_matches_frozen_baseline(case):
    # Dev-only fixtures live out of the repo (tests.baseline._paths). Skip where
    # the data is absent (CI / fresh checkout) -- run `freeze_baseline` to create.
    if not FROZEN_DIR.exists():
        pytest.skip(f"frozen baselines absent at {FROZEN_DIR} (set AIC_BASELINE_DIR)")
    artifacts = generate_case(case)
    case_dir = FROZEN_DIR / case.name
    assert case_dir.is_dir(), f"no frozen baseline for {case.name}; run freeze_baseline"
    frozen_names = {p.name for p in case_dir.iterdir() if p.is_file()}
    assert set(artifacts) == frozen_names, (
        f"artifact set drift for {case.name}: +{set(artifacts) - frozen_names} -{frozen_names - set(artifacts)}"
    )
    for name, content in artifacts.items():
        expected = (case_dir / name).read_text(encoding="utf-8")
        assert content == expected, f"byte drift in {case.name}/{name}"
