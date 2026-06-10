import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "diagnostics"))

from analyze_nsys_comm_overlap import _merge_intervals_ns, _overlap_ns, _union_ns  # noqa: E402


def test_interval_union_merges_touching_and_overlapping_ranges() -> None:
    assert _merge_intervals_ns([(10, 20), (20, 30), (5, 8), (25, 40)]) == [(5, 8), (10, 40)]
    assert _union_ns([(10, 20), (15, 30), (40, 50)]) == 30


def test_overlap_counts_only_intersection_of_merged_ranges() -> None:
    compute = [(0, 10), (20, 40), (35, 50)]
    comm = [(5, 25), (30, 32), (45, 60)]

    assert _overlap_ns(compute, comm) == 17
