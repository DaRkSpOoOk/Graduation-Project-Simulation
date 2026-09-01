import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/manifests/karsl_milestone1_pilot.csv"


class TestKArSLManifest(unittest.TestCase):
    def test_manifest_is_an_exact_three_signer_six_sign_pilot(self) -> None:
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 18)
        self.assertEqual({row["sign_id"] for row in rows}, {"0171", "0172", "0173", "0174", "0175", "0176"})
        self.assertEqual({row["signer_id"] for row in rows}, {"01", "02", "03"})
        self.assertEqual(len({row["sample_id"] for row in rows}), 18)
        self.assertTrue(all(row["split"] == "test" and row["modality"] == "RGB" for row in rows))
        self.assertTrue(all(row["source_archive_member"].endswith(".mp4") for row in rows))
        self.assertTrue(all(row["checksum_sha256"] for row in rows))


if __name__ == "__main__":
    unittest.main()
