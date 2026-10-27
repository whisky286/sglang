"""Membership validation for original-rank recovery; no model or device needed."""

import unittest
from dataclasses import FrozenInstanceError

from sglang.srt.fault_tolerance.recovery import RankRecoveryPlan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRankRecoveryPlan(CustomTestCase):
    def test_recovers_middle_rank_without_renumbering_survivors(self):
        plan = RankRecoveryPlan(2, (True, False, True, True), (1,))

        self.assertEqual(plan.target_active_mask, (True, True, True, True))
        self.assertEqual(plan.active_original_ranks, (0, 1, 2, 3))
        self.assertEqual(plan.store_prefix, "npu-ft/1111/2")
        self.assertEqual(plan.previous_active_mask, (True, False, True, True))

    def test_partial_recovery_keeps_other_removed_ranks_inactive(self):
        plan = RankRecoveryPlan(4, (True, False, True, False), (3,))

        self.assertEqual(plan.active_original_ranks, (0, 2, 3))
        self.assertEqual(plan.target_active_mask, (True, False, True, True))

    def test_attention_tp_requires_whole_dp_blocks(self):
        previous = (True, True, False, False, False, False)
        plan = RankRecoveryPlan(3, previous, (3, 2), ranks_per_dp=2)
        self.assertEqual(plan.recovering_ranks, (2, 3))
        self.assertEqual(
            plan.target_active_mask, (True, True, True, True, False, False)
        )
        with self.assertRaisesRegex(ValueError, "complete DP groups"):
            RankRecoveryPlan(3, previous, (2,), ranks_per_dp=2)

    def test_rejects_invalid_capacity_membership_and_rank_requests(self):
        cases = [
            (0, (True, False), (1,), 1),
            (True, (True, False), (1,), 1),
            (1, (True, False), (), 1),
            (1, (True, False), (0,), 1),
            (1, (True, False), (1, 1), 1),
            (1, (True, False), (-1,), 1),
            (1, (True, False), (2,), 1),
            (1, (True, False), (True,), 1),
            (1, (1, 0), (1,), 1),
            (1, (False, False), (1,), 1),
            (1, (True, False, False), (1,), 2),
            (1, (True, False, False, False), (2, 3), 2),
            (1, (True, False), (1,), 0),
        ]
        for epoch, mask, ranks, block_size in cases:
            with self.subTest(
                epoch=epoch, mask=mask, ranks=ranks, block_size=block_size
            ):
                with self.assertRaises(ValueError):
                    RankRecoveryPlan(epoch, mask, ranks, block_size)

    def test_copied_wire_lists_cannot_mutate_validated_plan(self):
        mask, ranks = [True, False], [1]
        plan = RankRecoveryPlan(2, mask, ranks)
        mask[0] = False
        ranks.clear()
        self.assertEqual(plan.previous_active_mask, (True, False))
        self.assertEqual(plan.recovering_ranks, (1,))
        with self.assertRaises(FrozenInstanceError):
            plan.membership_epoch = 3


if __name__ == "__main__":
    unittest.main()
