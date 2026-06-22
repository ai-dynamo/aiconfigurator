import pytest

from tests.baseline._paths import REF_DIR
from tests.baseline.canary import CANARY_CASES
from tests.baseline.freeze_baseline import generate_case
from tests.baseline.semantic import compare_artifact


@pytest.mark.parametrize("case", CANARY_CASES, ids=lambda c: c.name)
def test_artifacts_match_precutover_reference(case):
    # The pre-cutover reference is a dev-only, out-of-repo fixture
    # (tests.baseline._paths). Skip where absent (CI / fresh checkout).
    if not REF_DIR.exists():
        pytest.skip(f"pre-cutover ref absent at {REF_DIR} (set AIC_BASELINE_DIR)")
    arts = generate_case(case)
    ref_dir = REF_DIR / case.name
    assert ref_dir.is_dir(), f"no pre-cutover ref for {case.name}"
    ref_names = {p.name for p in ref_dir.iterdir() if p.is_file()}
    assert set(arts) == ref_names, f"artifact set drift for {case.name}"
    for name, content in arts.items():
        ok, msg = compare_artifact(name, content, (ref_dir / name).read_text())
        assert ok, f"{case.name}: {msg}"
