"""Lightweight checks for the TASK-003B decision artifacts.

These tests use only the small committed JSON scorecard and pure-Python
geometry; they never require the WiLoR checkpoint, MANO assets, KArSL
videos, or any generated run directory.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORECARD = ROOT / "reports" / "evaluation" / "TASK-003B-scorecard.json"


class TestScorecardIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = json.loads(SCORECARD.read_text())

    def test_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(self.doc["weights"].values()), 1.0, places=9)

    def test_every_weighted_scenario_sums_to_one(self) -> None:
        for name, scenario in self.doc["sensitivity"].items():
            self.assertAlmostEqual(sum(scenario["weights"].values()), 1.0, places=9, msg=name)

    def test_totals_match_scores_times_weights(self) -> None:
        weights = self.doc["weights"]
        for system in ("mediapipe", "wilor"):
            expected = sum(self.doc["scores"][system][k] * weights[k] for k in weights)
            self.assertAlmostEqual(self.doc["totals"][system], expected, places=6, msg=system)

    def test_scores_are_in_range_and_complete(self) -> None:
        for system in ("mediapipe", "wilor"):
            scores = self.doc["scores"][system]
            self.assertEqual(set(scores), set(self.doc["weights"]), msg=system)
            for category, value in scores.items():
                self.assertGreaterEqual(value, 0.0, msg=f"{system}/{category}")
                self.assertLessEqual(value, 10.0, msg=f"{system}/{category}")

    def test_every_category_has_a_justification(self) -> None:
        self.assertEqual(set(self.doc["justifications"]), set(self.doc["weights"]))

    def test_declared_winner_agrees_with_arithmetic(self) -> None:
        totals = self.doc["totals"]
        winner = "WILOR" if totals["wilor"] > totals["mediapipe"] else "MEDIAPIPE"
        self.assertEqual(self.doc["decision"]["primary_pose_pipeline"], winner)

    def test_sensitivity_winners_are_recomputable(self) -> None:
        for name, scenario in self.doc["sensitivity"].items():
            w = scenario["weights"]
            mp = sum(self.doc["scores"]["mediapipe"][k] * w[k] for k in w)
            wl = sum(self.doc["scores"]["wilor"][k] * w[k] for k in w)
            self.assertAlmostEqual(scenario["mediapipe_total"], mp, places=6, msg=name)
            self.assertAlmostEqual(scenario["wilor_total"], wl, places=6, msg=name)
            self.assertEqual(scenario["winner"], "wilor" if wl > mp else "mediapipe", msg=name)

    def test_frozen_commits_are_recorded(self) -> None:
        commits = self.doc["source_commits"]
        self.assertEqual(commits["mediapipe_frozen"], "ed25d9f2814493f02e16848d23c3466b54f06d6e")
        self.assertEqual(commits["wilor_frozen"], "20e83afd7a54493523389fe02ca7077b1afc5866")
        self.assertEqual(commits["fairness_remediation_base"], "63c7e683eeab19624a00480b9e0525e25ca07c44")

    def test_dataset_contract_is_unchanged(self) -> None:
        dataset = self.doc["dataset"]
        self.assertEqual(dataset["videos"], 18)
        self.assertEqual(dataset["frames"], 894)
        self.assertEqual(
            dataset["manifest_sha256"],
            "4a8e0d24f587d9c54ad87ebd896c6df0290b3ebc80885774100d71b29dbc3c0c",
        )

    def test_required_disclosures_are_present(self) -> None:
        for key in (
            "thresholds_not_normalized",
            "compute_not_normalized",
            "absolute_3d_not_compared",
            "no_ground_truth",
        ):
            self.assertIn(key, self.doc["disclosures"])


class TestSideBySideProjection(unittest.TestCase):
    """The visual helper is rendering-only, but its projection must match the
    frozen pinhole model used by the WiLoR adapter."""

    def test_projection_places_a_centered_point_at_the_principal_point(self) -> None:
        from scripts.render_task003b_side_by_side import project_points

        pts = np.array([[0.0, 0.0, 0.0]])
        out = project_points(pts, np.array([0.0, 0.0, 10.0]), focal=5000.0, img_wh=(1920.0, 1080.0))
        self.assertAlmostEqual(out[0, 0], 960.0, places=6)
        self.assertAlmostEqual(out[0, 1], 540.0, places=6)

    def test_projection_scales_with_focal_over_depth(self) -> None:
        from scripts.render_task003b_side_by_side import project_points

        pts = np.array([[1.0, 0.0, 0.0]])
        out = project_points(pts, np.array([0.0, 0.0, 10.0]), focal=100.0, img_wh=(200.0, 200.0))
        # x_pixel = focal * (x / z) + cx = 100 * (1/10) + 100
        self.assertAlmostEqual(out[0, 0], 110.0, places=6)
        self.assertTrue(math.isfinite(float(out[0, 1])))


if __name__ == "__main__":
    unittest.main()
