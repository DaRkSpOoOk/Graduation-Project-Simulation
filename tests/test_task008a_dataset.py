import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluation.dataset.core28 import (
    CORE28_SIGN_IDS,
    EXTENDED_LETTER_SIGN_IDS,
    core28_records,
    extended_letter_records,
    load_label_records,
    validate_core28_records,
)
from evaluation.dataset.acquisition import (
    AcquisitionError,
    CatalogEntry,
    download_file,
    load_source_catalog,
    preflight_catalog,
)
from evaluation.dataset.manifest import (
    CORE28_MANIFEST_FIELDS,
    VideoRecord,
    build_manifest_from_video_root,
    validate_manifest_rows,
)
from evaluation.dataset.orchestrator import (
    RunPaths,
    StateStore,
    _stage_sidecar_payload,
    assign_shards,
    sample_ids_for_shard,
    stable_shard_index,
    validate_stage_artifact,
)
from evaluation.dataset.splits import build_loso_splits


ROOT = Path(__file__).resolve().parents[1]


def _row(sign_id: int, signer: str, partition: str, repetition: int = 1) -> dict[str, str]:
    label = core28_records()[sign_id - CORE28_SIGN_IDS[0]]
    return VideoRecord(
        sample_id=f"sample_{signer}_{sign_id:04d}_{partition}_{repetition:03d}",
        sign_id=sign_id,
        label_ar=label.label_ar,
        label_en=label.label_en,
        label_index=CORE28_SIGN_IDS.index(sign_id),
        signer_id=signer,
        official_partition=partition,
        repetition_id=f"rep{repetition:03d}",
        source_relative_path=f"raw/signer{signer}/{partition}/{sign_id:04d}/sample_{repetition:03d}.mp4",
        source_file_name=f"sample_{repetition:03d}.mp4",
        source_size_bytes=1,
    ).to_row()


class TestTask008ADatasetContracts(unittest.TestCase):
    def test_core28_and_extended_catalogs_are_exact(self) -> None:
        self.assertEqual(tuple(record.sign_id for record in core28_records()), CORE28_SIGN_IDS)
        self.assertEqual(len(core28_records()), 28)
        self.assertEqual(tuple(record.sign_id for record in extended_letter_records()), EXTENDED_LETTER_SIGN_IDS)
        self.assertTrue(all(record.is_core28 for record in core28_records()))
        committed = load_label_records(ROOT / "datasets/manifests/karsl_core28_labels.csv")
        self.assertEqual([record.sign_id for record in validate_core28_records(committed)], list(CORE28_SIGN_IDS))

    def test_committed_manifest_is_populated_and_portable(self) -> None:
        """TASK-008B populated this manifest from the official local source.

        It was schema-only while the source was unavailable; the schema and
        path-portability guarantees still hold, but emptiness no longer does.
        """

        manifest = ROOT / "datasets/manifests/karsl_core28.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(tuple(reader.fieldnames or ()), CORE28_MANIFEST_FIELDS)
            rows = list(reader)
        self.assertGreater(len(rows), 4000)
        for row in rows:
            value = row["source_relative_path"]
            self.assertFalse(value.startswith("/"), value)
            self.assertNotIn("/home/", value)
        loaded = __import__(
            "evaluation.dataset.manifest", fromlist=["load_manifest"]
        ).load_manifest(manifest)
        self.assertEqual(len(loaded), len(rows))

    def test_discovery_is_deterministic_and_uses_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_root = root / "raw"
            for sign_id in CORE28_SIGN_IDS:
                for signer in ("01", "02", "03"):
                    for partition in ("train", "test"):
                        directory = video_root / f"signer{signer}" / partition / f"{sign_id:04d}"
                        directory.mkdir(parents=True)
                        for filename in ("z.mp4", "a.mp4"):
                            (directory / filename).write_bytes(b"synthetic video")
            first = build_manifest_from_video_root(
                root, video_root, core28_records(), inspect=False, hash_files=False
            )
            second = build_manifest_from_video_root(
                root, video_root, core28_records(), inspect=False, hash_files=False
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first), 28 * 3 * 2 * 2)
            self.assertEqual(first[0]["repetition_id"], "rep001")
            self.assertEqual(first[0]["source_file_name"], "a.mp4")
            self.assertTrue(all(not Path(row["source_relative_path"]).is_absolute() for row in first))
            self.assertTrue(all("\\" not in row["source_relative_path"] for row in first))

    def test_manifest_rejects_duplicates_bad_labels_and_empty_paths(self) -> None:
        row = _row(32, "01", "train")
        with self.assertRaises(ValueError):
            validate_manifest_rows([row, dict(row)])
        bad_label = dict(row, label_ar="x")
        with self.assertRaises(ValueError):
            validate_manifest_rows([bad_label])
        with self.assertRaises(ValueError):
            validate_manifest_rows([dict(row, source_relative_path="")])
        with self.assertRaises(ValueError):
            validate_manifest_rows([dict(row, source_relative_path="../outside.mp4")])
        duplicate_source = dict(row, sample_id="different_sample")
        with self.assertRaises(ValueError):
            validate_manifest_rows([row, duplicate_source])
        with self.assertRaises(ValueError):
            validate_manifest_rows([dict(row, label_en_if_available="")])

    def test_loso_has_no_held_out_signer_leakage_and_is_deterministic(self) -> None:
        rows = [
            _row(sign_id, signer, partition)
            for sign_id in CORE28_SIGN_IDS
            for signer in ("01", "02", "03")
            for partition in ("train", "test")
        ]
        first = build_loso_splits(rows, "02")
        second = build_loso_splits(rows, "02")
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(rows))
        self.assertEqual(len({row["sample_id"] for row in first}), len(rows))
        self.assertTrue(all(row["signer_id"] == "02" for row in first if row["role"] == "test"))
        self.assertTrue(all(row["signer_id"] != "02" for row in first if row["role"] != "test"))
        self.assertEqual(
            {int(row["sign_id"]) for row in first if row["role"] == "train"},
            set(CORE28_SIGN_IDS),
        )
        self.assertEqual(
            {int(row["sign_id"]) for row in first if row["role"] == "validation"},
            set(CORE28_SIGN_IDS),
        )

    def test_shards_are_stable_and_disjoint(self) -> None:
        rows = [_row(32, "01", "train", index) for index in range(1, 10)]
        first = assign_shards(rows, 4)
        second = assign_shards(rows, 4)
        self.assertEqual(first, second)
        self.assertEqual(
            {row["sample_id"] for shard in first.values() for row in shard},
            {row["sample_id"] for row in rows},
        )
        self.assertEqual(sum(len(shard) for shard in first.values()), len(rows))
        self.assertEqual(stable_shard_index(7, 4), 3)
        self.assertEqual(sample_ids_for_shard(rows, 4, 3), [rows[3]["sample_id"], rows[7]["sample_id"]])
        with self.assertRaises(ValueError):
            sample_ids_for_shard(rows, 4, 4)

    def test_git_exclusion_guard_covers_data_but_not_manifests(self) -> None:
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", "datasets/raw/dummy/video.mp4"],
            cwd=ROOT,
            check=False,
        )
        tracked_schema = subprocess.run(
            ["git", "check-ignore", "-q", "--", "datasets/manifests/dummy.csv"],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(ignored.returncode, 0)
        self.assertNotEqual(tracked_schema.returncode, 0)

    def test_corrupt_or_detector_only_pose_is_not_resume_valid(self) -> None:
        row = _row(32, "01", "train")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = RunPaths(root)
            sample_dir = paths.stage_dir("POSE", row["sample_id"])
            sample_dir.mkdir(parents=True)
            np.savez(
                sample_dir / "wilor_raw.npz",
                frame_index=np.arange(2, dtype=np.int32),
                run_metadata_json=np.array(json.dumps({"mode": "full"})),
            )
            sidecar = _stage_sidecar_payload(
                "POSE", row, "a" * 64, "", frames=2
            )
            paths.stage_sidecar("POSE", row["sample_id"]).write_text(json.dumps(sidecar), encoding="utf-8")
            self.assertIsNotNone(validate_stage_artifact(paths, "POSE", row, "a" * 64))
            np.savez(
                sample_dir / "wilor_raw.npz",
                frame_index=np.arange(2, dtype=np.int32),
                run_metadata_json=np.array(json.dumps({"mode": "detector_only"})),
            )
            self.assertIsNone(validate_stage_artifact(paths, "POSE", row, "a" * 64))

    def test_state_is_atomic_and_reloadable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = RunPaths(Path(temporary))
            state = StateStore(paths, 2, 4, "b" * 64)
            state.update("sample", status="POSE_DONE", frames_processed=12)
            reloaded = StateStore(paths, 2, 4, "b" * 64)
            self.assertEqual(reloaded.get("sample")["status"], "POSE_DONE")
            self.assertEqual(reloaded.get("sample")["frames_processed"], 12)
            with self.assertRaises(ValueError):
                StateStore(paths, 2, 4, "c" * 64)

    def test_failed_sample_is_recorded_and_retry_failed_reprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            source = data_root / "raw/signer01/train/0032/sample.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"synthetic video")
            import evaluation.dataset.orchestrator as orchestration

            row = dict(_row(32, "01", "train"), source_relative_path="raw/signer01/train/0032/sample.mp4")
            row["source_sha256"] = orchestration.sha256_file(source)
            paths = RunPaths(root / "run")
            state = StateStore(paths, 0, 1, "d" * 64)
            with patch.object(orchestration, "_run_pose", side_effect=RuntimeError("synthetic failure")):
                failed = orchestration.process_sample(
                    row,
                    paths=paths,
                    data_root=data_root,
                    manifest_hash="d" * 64,
                    state=state,
                    pipeline=object(),
                    resume=False,
                    retry_failed=False,
                    requested_stage="POSE",
                )
            self.assertEqual(failed["status"], "FAILED")
            self.assertTrue((paths.failures / "shard-00.jsonl").is_file())

            with patch.object(
                orchestration,
                "_run_pose",
                return_value={"status": "POSE_DONE", "frames": 1},
            ) as run_pose, patch.object(
                orchestration,
                "_run_tracking",
                return_value={"status": "TRACKING_DONE", "frames": 1},
            ), patch.object(
                orchestration,
                "_run_kinematics",
                return_value={"status": "KINEMATICS_DONE", "frames": 1},
            ), patch.object(
                orchestration,
                "_run_virtual_glove",
                return_value={"status": "VIRTUAL_GLOVE_DONE", "frames": 1},
            ):
                retried = orchestration.process_sample(
                    row,
                    paths=paths,
                    data_root=data_root,
                    manifest_hash="d" * 64,
                    state=state,
                    pipeline=object(),
                    resume=False,
                    retry_failed=True,
                    requested_stage="POSE",
                )
            self.assertEqual(retried["status"], "VIRTUAL_GLOVE_DONE")
            run_pose.assert_called_once()

    def test_status_mode_is_read_only_and_does_not_need_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            before = sorted(run_root.rglob("*")) if run_root.exists() else []
            command = [
                sys.executable,
                str(ROOT / "scripts/run_task008a_karsl_core28.py"),
                "--manifest",
                str(ROOT / "datasets/manifests/karsl_core28.csv"),
                "--run-root",
                str(run_root),
                "--status",
            ]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("TASK-008A CORE-28", completed.stdout)
            after = sorted(run_root.rglob("*")) if run_root.exists() else []
            self.assertEqual(before, after)

    def test_catalog_rejects_unsafe_or_duplicate_destinations_and_reuses_verified_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.csv"
            catalog.write_text(
                "source_url,destination_relative_path,expected_sha256,expected_size_bytes,sample_id\n"
                "file:///tmp/source,../escape,,1,x\n",
                encoding="utf-8",
            )
            with self.assertRaises(AcquisitionError):
                load_source_catalog(catalog)
            catalog.write_text(
                "source_url,destination_relative_path,expected_sha256,expected_size_bytes,sample_id\n"
                "file:///tmp/source,out.bin,,1,x\n"
                "file:///tmp/source,other.bin,,1,x\n",
                encoding="utf-8",
            )
            with self.assertRaises(AcquisitionError):
                load_source_catalog(catalog)
            source = root / "source.bin"
            source.write_bytes(b"source")
            import hashlib

            destination = root / "data/out.bin"
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            first = download_file(
                source.as_uri(), destination, expected_sha256=expected, expected_size_bytes=source.stat().st_size
            )
            second = download_file(
                source.as_uri(), destination, expected_sha256=expected, expected_size_bytes=source.stat().st_size
            )
            self.assertFalse(first["skipped_existing"] if "skipped_existing" in first else False)
            self.assertTrue(second["skipped_existing"])
            self.assertTrue(preflight_catalog([CatalogEntry(source.as_uri(), "out.bin", expected, 6)], root / "data")["sufficient_for_known_bytes"])


if __name__ == "__main__":
    unittest.main()
