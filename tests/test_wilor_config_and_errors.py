import tempfile
import unittest
from pathlib import Path

from pose.wilor.config import WilorAssetPaths
from pose.wilor.errors import ManoAssetMissingError, WilorAssetMissingError
from pose.wilor.model_loader import check_assets


class TestWilorAssetPaths(unittest.TestCase):
    def test_resolve_respects_explicit_dir(self) -> None:
        assets = WilorAssetPaths.resolve("/tmp/example_wilor_assets")
        self.assertEqual(assets.assets_dir, Path("/tmp/example_wilor_assets"))
        self.assertEqual(
            assets.detector_checkpoint,
            Path("/tmp/example_wilor_assets/pretrained_models/detector.pt"),
        )
        self.assertEqual(
            assets.mano_right_pkl,
            Path("/tmp/example_wilor_assets/mano_data/MANO_RIGHT.pkl"),
        )


class TestCheckAssets(unittest.TestCase):
    def test_missing_detector_reported_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            assets = WilorAssetPaths.resolve(tmp)
            with self.assertRaises(WilorAssetMissingError) as ctx:
                check_assets(assets)
            self.assertEqual(ctx.exception.missing_path, assets.detector_checkpoint)

    def test_missing_mano_reported_as_mano_specific_error(self) -> None:
        """Reproduces this environment's actual, documented blocker: every
        other asset present, only the gated MANO file missing. See
        reports/pose/wilor/TASK-002-wilor-karsl-pilot.md."""
        with tempfile.TemporaryDirectory() as tmp:
            assets = WilorAssetPaths.resolve(tmp)
            assets.detector_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            assets.detector_checkpoint.write_bytes(b"stub")
            assets.wilor_checkpoint.write_bytes(b"stub")
            assets.model_config.write_text("stub: true\n")

            with self.assertRaises(ManoAssetMissingError) as ctx:
                check_assets(assets)
            self.assertEqual(ctx.exception.missing_path, assets.mano_right_pkl)
            self.assertIn("mano.is.tue.mpg.de", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
