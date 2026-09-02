"""TASK-006D integration tests using only synthetic data."""

from __future__ import annotations

import unittest

from virtual_glove.layout import layout_document
from evaluation.virtual_glove_integration import (
    run_gyro_convention_checks,
    run_invalid_fixture_validation,
    run_layout_reconciliation,
    run_valid_fixture_validation,
)
from evaluation.virtual_glove_qa.contract import parse_sensor_layout
from evaluation.virtual_glove_qa.validator import _adc_transfer, _metadata_normalization_check


class Task006DIntegrationTests(unittest.TestCase):
    def test_all_valid_fixtures_pass_production_adapter(self) -> None:
        result = run_valid_fixture_validation()
        self.assertEqual(result["fixture_count"], 167)
        self.assertEqual(result["passed"], 167)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["serialized_comparison_max_abs_error"], 0.0)

    def test_invalid_fixtures_are_rejected_at_neutral_boundary(self) -> None:
        result = run_invalid_fixture_validation()
        self.assertEqual(result["fixture_count"], 12)
        self.assertEqual(result["passed"], 12)
        self.assertEqual(result["rejected"], 12)

    def test_physical_template_expands_to_runtime_identities(self) -> None:
        result = run_layout_reconciliation()
        self.assertTrue(result["passed"])
        self.assertEqual(result["physical_template_definitions_per_hand"], 20)
        self.assertEqual(result["runtime_identity_count"], 40)
        self.assertEqual(result["runtime_hall_count"], 38)
        self.assertEqual(result["runtime_imu_count"], 2)
        self.assertEqual(result["markers"], {"hall": ["H"], "imu": ["IMU"]})

    def test_production_layout_is_accepted_as_template(self) -> None:
        result = parse_sensor_layout({"sensor_layout": layout_document()})
        self.assertTrue(result.passed, result.failures)
        self.assertEqual(result.representation, "physical_template_expanded")
        self.assertEqual(result.physical_template_count, 20)
        self.assertEqual(len(result.sensors), 40)

    def test_production_metadata_aliases_remain_strictly_validated(self) -> None:
        metadata = {
            "representations": {
                "normalized_ideal_sensor": {
                    "formula": "degrees / 180.0",
                    "range": [0.0, 1.0],
                    "fitted_to_dataset": False,
                },
                "optional_adc": {
                    "bits": 12,
                    "range": [0, 4095],
                    "formula": "normalized * 4095",
                    "rounding": "half-up, floor(x + 0.5)",
                    "invalid_sentinel": -1,
                },
            }
        }
        normalization = _metadata_normalization_check(metadata)
        transfer, failures = _adc_transfer(metadata)
        self.assertTrue(normalization["passed"], normalization["failures"])
        self.assertEqual(transfer["bits"], 12)
        self.assertEqual(transfer["min_code"], 0)
        self.assertEqual(transfer["max_code"], 4095)
        self.assertEqual(transfer["invalid_value"], -1)
        self.assertEqual(transfer["tolerance_codes"], 0.0)
        self.assertEqual(failures, [])

    def test_gyro_body_frame_convention_is_resolved(self) -> None:
        result = run_gyro_convention_checks()
        self.assertTrue(result["passed"])
        self.assertTrue(result["noncommuting_world_body_difference_observed"])
        self.assertEqual(len(result["cases"]), 4)


if __name__ == "__main__":
    unittest.main()
