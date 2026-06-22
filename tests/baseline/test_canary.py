from tests.baseline.canary import CANARY_CASES


def test_canary_matrix_is_nonempty_and_well_formed():
    assert len(CANARY_CASES) >= 5
    seen = set()
    for case in CANARY_CASES:
        assert case.name and case.name not in seen, f"duplicate/empty name: {case.name}"
        seen.add(case.name)
        assert case.backend in {"vllm", "sglang", "trtllm"}
        assert isinstance(case.params, dict) and case.params
        assert "params" in case.params  # the role sub-tree (prefill/decode/agg)
