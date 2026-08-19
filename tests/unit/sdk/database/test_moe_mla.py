# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer-policy resolution and provenance summary.

The per-call query facades this file used to exercise on the synthetic
comprehensive fixture (``query_moe`` / ``query_mla_bmm`` / ``query_mem_op`` /
``query_p2p`` interpolation, util grids, overflow/underflow holds, the
cross-shape/quant/profile transfer ladder, typed-miss error surfaces) retired
to the compiled engine with #1357 PR-5. Their behaviour is anchored by
the retired shim-migration baseline (pinned pre-deletion
cases on real databases) and the frozen parity goldens. What remains testable
from Python is the policy/provenance surface below.
"""

import pytest

from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit


class TestTransferPolicyAndProvenance:
    def test_resolve_transfer_policy(self):
        tk = common.TransferKind
        assert common.resolve_transfer_policy(None) == common.ALL_TRANSFERS
        assert common.resolve_transfer_policy("conservative") == common.TRANSFER_PRESETS["conservative"]
        assert common.resolve_transfer_policy(["xshape", "xquant"]) == frozenset({tk.XSHAPE, tk.XQUANT})
        assert common.resolve_transfer_policy(tk.XPROFILE) == frozenset({tk.XPROFILE})
        # comma-separated string (the CLI / flat-YAML form) splits into kinds
        assert common.resolve_transfer_policy("xshape,xquant") == frozenset({tk.XSHAPE, tk.XQUANT})
        assert common.resolve_transfer_policy(" xshape , xprofile ") == frozenset({tk.XSHAPE, tk.XPROFILE})
        with pytest.raises(ValueError):
            common.resolve_transfer_policy("not_a_kind")

    def test_worst_provenance_picks_least_confident(self):
        from aiconfigurator.sdk.operations import util_empirical as ue

        assert ue.worst_provenance(set()) == "silicon"  # nothing fired
        assert ue.worst_provenance({"empirical"}) == "empirical"
        # least-confident (latest in PROVENANCE_ORDER) wins over a mixed set
        assert ue.worst_provenance({"xshape", "xop", "empirical"}) == "xop"
        assert ue.worst_provenance({"xshape", "xquant"}) == "xquant"
        # capture round-trip: note inside the block, collected after
        with ue.capture_provenance() as tags:
            ue.note_provenance("xprofile")
            ue.note_provenance("xshape")
        assert tags == {"xprofile", "xshape"} and ue.worst_provenance(tags) == "xprofile"
        ue.note_provenance("xop")  # outside any capture -> no-op, no error
