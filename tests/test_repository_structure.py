import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestRepositoryStructure(unittest.TestCase):
    def test_required_files_exist(self) -> None:
        for rel_path in [
            "AGENTS.md",
            "README.md",
            "pyproject.toml",
            ".github/copilot-instructions.md",
            "reports/README.md",
            "reports/experiments/TASK-000-repository-bootstrap.md",
        ]:
            self.assertTrue((ROOT / rel_path).exists(), rel_path)

    def test_required_directories_exist(self) -> None:
        for rel_path in [
            "configs/datasets",
            "configs/pose",
            "configs/tracking",
            "configs/evaluation",
            "pose/common",
            "pose/mediapipe",
            "pose/wilor",
            "pose/hamer",
            "pose/omnihands",
            "reports/pose/mediapipe",
            "reports/pose/wilor",
            "reports/pose/hamer",
            "reports/pose/omnihands",
        ]:
            self.assertTrue((ROOT / rel_path).exists(), rel_path)


if __name__ == "__main__":
    unittest.main()
