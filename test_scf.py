import math
import unittest

import numpy as np

from scf import (
    auc_credits,
    conformal_ranks,
    fisher_scores,
    raw_scores,
    reference_tail,
)


def pack(bags):
    return np.concatenate(bags), np.r_[0, np.cumsum([len(bag) for bag in bags])]


class ScfTests(unittest.TestCase):
    def test_source_balanced_reference_with_unequal_bags_and_ties(self):
        values, offsets = pack([[0.0], [0.0, 1.0, 2.0], [0.5, 1.0, 1.5, 2.0, 3.0]])
        q = reference_tail(values, offsets, np.array([0, 1]))
        expected = [7 / 15, 7 / 15, 1 / 3, 1 / 3, 1 / 5]
        np.testing.assert_allclose(q[4:], expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(q[:2], [1.0, 1.0])
        self.assertNotAlmostEqual(q[4], 3 / 5)

    def test_capping_changes_the_two_bag_comparison(self):
        q, offsets = pack([[0.15, 0.15], [0.01, 1.0]])
        scores = fisher_scores(q, offsets, np.array([0.0, 0.05]))
        expected = [[-math.log(0.15)] * 2, [-math.log(0.01) / 2, -math.log(0.05) / 2]]
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-15)
        self.assertLess(scores[0, 0], scores[1, 0])
        self.assertGreater(scores[0, 1], scores[1, 1])

    def test_source_means_preserve_constants_and_small_residuals(self):
        values, offsets = pack([[0.1], [0.1] * 3, [0.1] * 16, [-1e16, 1.0, 1e16]])
        scores = raw_scores(values, offsets)
        np.testing.assert_array_equal(scores[:3, 0], [0.1] * 3)
        self.assertEqual(scores[3, 0], 1 / 3)
        q, offsets = pack([[0.2], [0.2] * 3, [0.2] * 16])
        fisher = fisher_scores(q, offsets, np.array([0.0, 0.05]))
        np.testing.assert_array_equal(fisher, np.full((3, 2), -np.log(0.2)))

    def test_auroc_assigns_half_credit_to_ties(self):
        participants = np.array([[1.0, 1.0], [2.0, 3.0]])
        nulls = np.array([[1.0, 0.0], [2.0, 2.0]])
        np.testing.assert_array_equal(auc_credits(participants, nulls), [4, 6])

    def test_conformal_ranks_include_equal_calibration_scores(self):
        calibration = np.array([[1.0], [1.0], [2.0]])
        targets = np.array([[1.0], [1.5], [2.0], [3.0]])
        np.testing.assert_array_equal(conformal_ranks(calibration, targets), [[4], [2], [2], [1]])

    def test_ramia_retains_floor_three_quarters_of_each_bag(self):
        values, offsets = pack([[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0], [9.0]])
        np.testing.assert_array_equal(raw_scores(values, offsets)[:, 3], [3.0, 1.5, 9.0])


if __name__ == "__main__":
    unittest.main()
