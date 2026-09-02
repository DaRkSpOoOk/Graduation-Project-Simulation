"""TASK-005D adapter and frozen-input safety tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.kinematics import (
    PilotInputError,
    build_benchmark_catalog,
    extract_production_sequence,
    validate_frozen_pilot_inputs,
)


class TestTask005DProductionAdapter(unittest.TestCase):
    def test_adapter_preserves_contract_shapes(self) -> None:
        case = next(case for case in build_benchmark_catalog() if case.case_id == "neutral")
        sequence = case.generate()
        adapted = extract_production_sequence(sequence)
        self.assertEqual(adapted.result.flexion_deg.shape, (1, 2, 5, 3))
        self.assertEqual(adapted.result.adjacent_spread_deg.shape, (1, 2, 4))
        self.assertEqual(adapted.result.palm_rotation_matrix.shape, (1, 2, 3, 3))
        self.assertEqual(adapted.result.palm_quaternion_wxyz.shape, (1, 2, 4))
        self.assertEqual(adapted.valid_kinematics.shape, (1, 2))
        self.assertEqual(adapted.valid_palm_frame.shape, (1, 2))
        self.assertTrue(adapted.all_channels_finite)

    def test_adapter_keeps_production_partial_invalidity_visible(self) -> None:
        case = next(case for case in build_benchmark_catalog() if case.case_id == "single_index_joint0_90deg")
        adapted = extract_production_sequence(case.generate())
        self.assertFalse(bool(adapted.valid_kinematics[0, 0]))
        self.assertTrue(bool(adapted.valid_palm_frame[0, 0]))
        self.assertTrue(np.isfinite(adapted.result.flexion_deg[0, 0]).all())
        self.assertTrue(np.isnan(adapted.result.adjacent_spread_deg[0, 0]).any())


class TestTask005DFrozenInputSafety(unittest.TestCase):
    def test_phase_a_directory_is_not_selected_by_kinematics_npz_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase_a = root / "wilor_karsl_pilot"
            (phase_a / "sample_001").mkdir(parents=True)
            np.savez(phase_a / "sample_001" / "wilor_raw.npz", detector_only=np.array([True]))
            empty_kinematics = root / "kinematics"
            empty_kinematics.mkdir()

            with self.assertRaises(PilotInputError):
                validate_frozen_pilot_inputs(
                    phase_a,
                    empty_kinematics,
                    expected_sample_count=0,
                    expected_total_frames=0,
                )
