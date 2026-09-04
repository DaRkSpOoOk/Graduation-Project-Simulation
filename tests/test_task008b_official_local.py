"""Tests for TASK-008B official-workbook verification and local-source discovery.

Every fixture is synthetic and built in a temporary directory. Nothing here
reads /home/hatim/datasets, requires a GPU, or downloads anything. GPU
throughput itself is measured in the report, not asserted here.
"""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from evaluation.dataset.core28 import (
    CORE28_SIGN_IDS,
    EXTENDED_LETTER_SIGN_IDS,
    LabelRecord,
    core28_records,
    extended_letter_records,
)
from evaluation.dataset.local_source import (
    FILENAME_RE,
    LocalSourceError,
    discover_local_videos,
    parse_video_filename,
    select_smoke_subset,
)
from evaluation.dataset.manifest import _locate_layout
from evaluation.dataset.official import (
    CATEGORY_CORE28_LETTER,
    CATEGORY_EXTENDED_LETTER,
    CATEGORY_NUMBER,
    LETTER_SIGN_IDS,
    NUMBER_SIGN_IDS,
    OFFICIAL_MAPPING_VERSION,
    category_breakdown,
    normalize_for_comparison,
    read_official_workbook,
    verify_candidate_mapping,
)
from evaluation.dataset.splits import build_loso_splits

REPO = Path(__file__).resolve().parents[1]

# The 28 standard letters and 11 extended forms exactly as the official
# workbook stores them. Duplicated here on purpose: a test that imported the
# same constant it is checking would prove nothing.
OFFICIAL_CORE28 = [
    (32, "ا", "alif"), (33, "ب", "baa"), (34, "ت", "ta"), (35, "ث", "tha"),
    (36, "ج", "Jiim"), (37, "ح", "Haa"), (38, "خ", "kha"), (39, "د", "daal"),
    (40, "ذ", "thal"), (41, "ر", "raa"), (42, "ز", "zay"), (43, "س", "siin"),
    (44, "ش", "shiin"), (45, "ص", "Saad"), (46, "ض", "Daad"), (47, "ط", "Taa"),
    (48, "ظ", "Zaa"), (49, "ع", "Ayn"), (50, "غ", "ghayn"), (51, "ف", "faa"),
    (52, "ق", "qaaf"), (53, "ك", "kaaf"), (54, "ل", "laam"), (55, "م", "miim"),
    (56, "ن", "noon"), (57, "ه", "haa"), (58, "و", "waaw"), (59, "ي", "yaa"),
]
OFFICIAL_EXTENDED = [
    (60, "ة", "taa marbuuTa"), (61, "أ", "alif with hamza above"),
    (62, "ؤ", "Waaw with hamza"), (63, "ئ", "Alif maqsoura with hamza"),
    (64, "ئـ", "hamza on line"), (65, "ء", "hamza"),
    (66, "إ", "alif with hamza below"), (67, "آ", "ALif with maad"),
    (68, "ى", "Alif maqsoura"), (69, "لا", "laam Alif"), (70, "ال", "Al"),
]
# SignIDs 1-31 are numeric magnitudes stored as spreadsheet integers.
OFFICIAL_NUMBERS = [
    *range(0, 10), 10, 20, 30, 40, 50, 60, 70, 80, 90,
    100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1000000, 10000000,
]


def _write_workbook(path: Path, rows: list[tuple[int, object, object]]) -> None:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["SignID", "Sign-Arabic", "Sign-English"])
    for row in rows:
        sheet.append(list(row))
    workbook.save(path)


def _official_like_rows() -> list[tuple[int, object, object]]:
    rows: list[tuple[int, object, object]] = []
    for index, value in enumerate(OFFICIAL_NUMBERS, start=1):
        rows.append((index, value, value))
    for sign_id, arabic, english in OFFICIAL_CORE28:
        rows.append((sign_id, arabic, english))
    for sign_id, arabic, english in OFFICIAL_EXTENDED:
        rows.append((sign_id, arabic, english))
    return rows


class TestOfficialWorkbookParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "KARSL-502_Labels.xlsx"
        _write_workbook(self.path, _official_like_rows())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_reads_header_rows_and_hash(self) -> None:
        result = read_official_workbook(self.path)
        self.assertEqual(result.header[:3], ["SignID", "Sign-Arabic", "Sign-English"])
        self.assertEqual(result.data_rows, 70)
        self.assertEqual(result.sign_id_min, 1)
        self.assertEqual(result.sign_id_max, 70)
        self.assertEqual(len(result.sha256), 64)
        self.assertGreater(result.size_bytes, 0)
        self.assertEqual(result.issues, [])

    def test_missing_workbook_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_official_workbook(Path(self.tmp.name) / "absent.xlsx")

    def test_duplicate_sign_id_is_reported_not_silently_kept(self) -> None:
        path = Path(self.tmp.name) / "dupe.xlsx"
        _write_workbook(path, [(32, "ا", "alif"), (32, "ب", "baa")])
        result = read_official_workbook(path)
        self.assertTrue(any("duplicate" in issue for issue in result.issues))
        self.assertEqual(result.data_rows, 1)

    def test_numeric_cells_keep_their_recorded_type(self) -> None:
        """Numbers are stored as integers; nothing is reinterpreted as text digits."""

        entries = read_official_workbook(self.path).by_id()
        self.assertEqual(entries[1].arabic_cell_type, "int")
        self.assertEqual(entries[32].arabic_cell_type, "str")

    def test_labels_are_preserved_verbatim(self) -> None:
        entries = read_official_workbook(self.path).by_id()
        self.assertEqual(entries[32].label_ar, "ا")
        self.assertEqual(entries[59].label_ar, "ي")

    def test_nfc_normalization_is_comparison_only(self) -> None:
        decomposed = "آ"  # alif + combining maddah
        self.assertNotEqual(decomposed, "آ")
        self.assertEqual(normalize_for_comparison(decomposed), normalize_for_comparison("آ"))


class TestCore28OfficialVerification(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "labels.xlsx"
        _write_workbook(self.path, _official_like_rows())
        self.workbook = read_official_workbook(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_core28_is_exactly_28_classes_0032_to_0059(self) -> None:
        self.assertEqual(tuple(CORE28_SIGN_IDS), tuple(range(32, 60)))
        self.assertEqual(len(CORE28_SIGN_IDS), 28)

    def test_repository_mapping_matches_the_workbook(self) -> None:
        result = verify_candidate_mapping(self.workbook, core28_records(), tuple(CORE28_SIGN_IDS))
        self.assertTrue(result["passed"], result["label_mismatches"])
        self.assertEqual(result["class_count"], 28)
        self.assertEqual(result["label_mismatches"], [])

    def test_class_indices_are_dense_and_ordered(self) -> None:
        records = core28_records()
        self.assertEqual([r.sign_id for r in records], list(range(32, 60)))
        indices = [CORE28_SIGN_IDS.index(r.sign_id) for r in records]
        self.assertEqual(indices, list(range(28)))

    def test_a_wrong_candidate_label_is_detected(self) -> None:
        wrong = [
            LabelRecord(sign_id=r.sign_id,
                        label_ar="X" if r.sign_id == 40 else r.label_ar,
                        label_en=r.label_en, source_row=r.source_row)
            for r in core28_records()
        ]
        result = verify_candidate_mapping(self.workbook, wrong, tuple(CORE28_SIGN_IDS))
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["label_mismatches"]), 1)
        self.assertEqual(result["label_mismatches"][0]["sign_id"], 40)

    def test_a_wrong_candidate_sign_id_range_is_detected(self) -> None:
        result = verify_candidate_mapping(self.workbook, core28_records(), tuple(range(33, 61)))
        self.assertFalse(result["passed"])
        self.assertFalse(result["sign_ids_match"])

    def test_mapping_version_is_frozen(self) -> None:
        self.assertEqual(OFFICIAL_MAPPING_VERSION, "karsl-core28-v2-official")


class TestExtendedAndNumberBoundaries(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "labels.xlsx"
        _write_workbook(path, _official_like_rows())
        self.workbook = read_official_workbook(path)
        self.breakdown = category_breakdown(self.workbook)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_extended_letters_are_eleven_classes_0060_to_0070(self) -> None:
        block = self.breakdown[CATEGORY_EXTENDED_LETTER]
        self.assertEqual(block["class_count"], 11)
        self.assertEqual(block["sign_ids"], list(range(60, 71)))
        self.assertEqual(tuple(EXTENDED_LETTER_SIGN_IDS), tuple(range(60, 71)))

    def test_extended_letters_are_letter_forms_not_numbers(self) -> None:
        for entry in self.breakdown[CATEGORY_EXTENDED_LETTER]["entries"]:
            self.assertEqual(entry["arabic_cell_type"], "str")
            self.assertFalse(str(entry["label_ar"]).isdigit())

    def test_repository_extended_mapping_matches_the_workbook(self) -> None:
        result = verify_candidate_mapping(
            self.workbook, extended_letter_records(), tuple(EXTENDED_LETTER_SIGN_IDS)
        )
        self.assertTrue(result["passed"], result["label_mismatches"])

    def test_letters_total_thirty_nine_classes(self) -> None:
        self.assertEqual(len(LETTER_SIGN_IDS), 39)
        total = (self.breakdown[CATEGORY_CORE28_LETTER]["class_count"]
                 + self.breakdown[CATEGORY_EXTENDED_LETTER]["class_count"])
        self.assertEqual(total, 39)

    def test_numbers_are_thirty_one_classes_not_thirty(self) -> None:
        """The workbook is authoritative over the literature's approximate count."""

        block = self.breakdown[CATEGORY_NUMBER]
        self.assertEqual(block["class_count"], 31)
        self.assertEqual(block["sign_ids"], list(range(1, 32)))
        self.assertEqual(len(NUMBER_SIGN_IDS), 31)

    def test_sign_id_0031_is_the_numeric_magnitude_ten_million(self) -> None:
        entry = self.workbook.by_id()[31]
        self.assertEqual(entry.category, CATEGORY_NUMBER)
        self.assertEqual(entry.label_ar, "10000000")
        self.assertEqual(entry.arabic_cell_type, "int")

    def test_number_and_letter_regions_do_not_overlap(self) -> None:
        self.assertEqual(set(NUMBER_SIGN_IDS) & set(LETTER_SIGN_IDS), set())
        self.assertEqual(max(NUMBER_SIGN_IDS) + 1, min(LETTER_SIGN_IDS))


class TestFilenameAndLayoutParsing(unittest.TestCase):
    def test_official_filename_fields(self) -> None:
        parsed = parse_video_filename("02_01_0041_(17_11_16_17_32_38)_c.mp4")
        self.assertEqual(parsed["chapter"], "02")
        self.assertEqual(parsed["signer"], "01")
        self.assertEqual(parsed["sign_id"], 41)
        self.assertEqual(parsed["modality"], "c")
        self.assertEqual(parsed["timestamp_token"], "17_11_16_17_32_38")

    def test_unrecognized_filename_returns_none(self) -> None:
        for name in ("random.mp4", "02_01_0041.mp4", "02_01_41_(x)_c.mp4"):
            self.assertIsNone(parse_video_filename(name), name)

    def test_layout_parser_accepts_the_official_videos_directory(self) -> None:
        self.assertEqual(
            _locate_layout(Path("01/train/videos/0041/02_01_0041_(x_x_x_x_x_x)_c.mp4")),
            ("01", "train", 41),
        )

    def test_layout_parser_still_accepts_a_flat_layout(self) -> None:
        self.assertEqual(_locate_layout(Path("02/test/0059/clip.mp4")), ("02", "test", 59))

    def test_non_core28_paths_are_skipped_not_fatal(self) -> None:
        self.assertIsNone(_locate_layout(Path("01/train/videos/0007/clip.mp4")))
        self.assertIsNone(_locate_layout(Path("unrelated/file.mp4")))

    def test_ambiguous_layout_still_raises(self) -> None:
        with self.assertRaises(ValueError):
            _locate_layout(Path("01/train/0041/02/test/0059/clip.mp4"))


def _make_dataset(root: Path, *, sign_ids=(32, 33), signers=("01", "02", "03"), count=2) -> None:
    for signer in signers:
        for partition in ("train", "test"):
            for sign_id in sign_ids:
                directory = root / signer / partition / "videos" / f"{sign_id:04d}"
                directory.mkdir(parents=True, exist_ok=True)
                for index in range(count):
                    chapter = "01" if sign_id <= 31 else "02"
                    name = (f"{chapter}_{signer}_{sign_id:04d}_"
                            f"(01_01_16_10_00_{index:02d})_c.mp4")
                    (directory / name).write_bytes(b"\x00" * (100 + index))


class TestLocalDiscovery(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _make_dataset(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_discovers_every_video_with_portable_paths(self) -> None:
        videos = discover_local_videos(self.root)
        self.assertEqual(len(videos), 3 * 2 * 2 * 2)
        for video in videos:
            self.assertFalse(video.relative_path.startswith("/"))
            self.assertNotIn(str(self.root), video.relative_path)

    def test_discovery_is_deterministic(self) -> None:
        first = [v.relative_path for v in discover_local_videos(self.root)]
        second = [v.relative_path for v in discover_local_videos(self.root)]
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))

    def test_sign_id_filter(self) -> None:
        videos = discover_local_videos(self.root, sign_ids={32})
        self.assertTrue(all(v.sign_id == 32 for v in videos))
        self.assertEqual(len(videos), 3 * 2 * 2)

    def test_directory_wins_over_filename_and_mismatch_is_flagged(self) -> None:
        directory = self.root / "01" / "train" / "videos" / "0035"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "02_01_0032_(01_01_16_10_00_00)_c.mp4").write_bytes(b"\x00")
        video = next(v for v in discover_local_videos(self.root) if v.sign_id == 35)
        self.assertEqual(video.sign_id, 35)
        self.assertEqual(video.filename_sign_id, 32)
        self.assertFalse(video.path_sign_id_matches_filename)

    def test_unrecognized_filename_raises(self) -> None:
        bad = self.root / "01" / "train" / "videos" / "0032" / "not-official.mp4"
        bad.write_bytes(b"\x00")
        with self.assertRaises(LocalSourceError):
            discover_local_videos(self.root)

    def test_missing_root_raises(self) -> None:
        with self.assertRaises(LocalSourceError):
            discover_local_videos(self.root / "absent")

    def test_no_download_is_attempted(self) -> None:
        """Discovery is purely local: the module imports no network client."""

        source = (REPO / "evaluation/dataset/local_source.py").read_text()
        for token in ("urllib.request", "requests", "http://", "https://", "urlopen"):
            self.assertNotIn(token, source)


_LABEL_AR = {sign_id: arabic for sign_id, arabic, _ in OFFICIAL_CORE28}
_LABEL_EN = {sign_id: english for sign_id, _, english in OFFICIAL_CORE28}


def _manifest_rows(count_per_cell: int = 3) -> list[dict[str, str]]:
    """Rows carrying the full committed manifest schema.

    Every (signer, partition, class) cell is populated so the LOSO builder's
    class-coverage rebalancing has something to work with, as it does on the
    real manifest.
    """

    sign_ids = tuple(range(32, 60))  # the splitter requires full Core-28 coverage
    rows = []
    for signer in ("01", "02", "03"):
        for partition in ("train", "test"):
            for sign_id in sign_ids:
                for index in range(count_per_cell):
                    frames = 10 + (sign_id - 32) * 5 + index * 7
                    rows.append({
                        "sample_id": (f"karsl_core28_s{signer}_sign{sign_id:04d}"
                                      f"_{partition}_rep{index:03d}"),
                        "source_dataset": "KArSL-502", "dataset_version": "v1", "modality": "rgb",
                        "sign_id": f"{sign_id:04d}",
                        "label_ar": _LABEL_AR[sign_id],
                        "label_en_if_available": _LABEL_EN[sign_id],
                        "label_index": str(sign_id - 32),
                        "signer_id": signer, "official_partition": partition,
                        "repetition_id": f"rep{index:03d}",
                        "source_relative_path": (f"{signer}/{partition}/videos/"
                                                 f"{sign_id:04d}/f{index}.mp4"),
                        "source_file_name": f"f{index}.mp4", "source_url": "",
                        "source_sha256": f"{sign_id:032d}{index:032d}",
                        "source_size_bytes": "1024",
                        "container": "mp4", "width": "1920", "height": "1080",
                        "fps": "30.000000", "frame_count": str(frames),
                        "duration_seconds": f"{frames / 30:.6f}",
                        "skeleton_available": "unknown",
                    })
    return rows


class TestDeterministicSmokeSelection(unittest.TestCase):
    def test_selection_is_deterministic(self) -> None:
        rows = _manifest_rows()
        first = [r["sample_id"] for r in select_smoke_subset(rows, target=18)]
        second = [r["sample_id"] for r in select_smoke_subset(list(reversed(rows)), target=18)]
        self.assertEqual(first, second)

    def test_covers_all_signers_and_both_partitions(self) -> None:
        selected = select_smoke_subset(_manifest_rows(), target=18)
        self.assertEqual(sorted({r["signer_id"] for r in selected}), ["01", "02", "03"])
        self.assertEqual(sorted({r["official_partition"] for r in selected}), ["test", "train"])

    def test_includes_shortest_and_longest_sequences(self) -> None:
        rows = _manifest_rows()
        selected = select_smoke_subset(rows, target=18)
        frames = [int(r["frame_count"]) for r in rows]
        chosen = [int(r["frame_count"]) for r in selected]
        self.assertEqual(min(chosen), min(frames))
        self.assertEqual(max(chosen), max(frames))

    def test_respects_the_target_size(self) -> None:
        for target in (18, 20, 24):
            self.assertLessEqual(len(select_smoke_subset(_manifest_rows(), target=target)), target)

    def test_rows_without_a_frame_count_are_excluded(self) -> None:
        rows = _manifest_rows()
        for row in rows:
            row["frame_count"] = ""
        self.assertEqual(select_smoke_subset(rows, target=18), [])

    def test_output_order_is_stable_and_sorted(self) -> None:
        selected = select_smoke_subset(_manifest_rows(), target=18)
        keys = [(r["signer_id"], r["official_partition"], int(r["frame_count"]), r["sample_id"])
                for r in selected]
        self.assertEqual(keys, sorted(keys))


class TestRealLosoGeneration(unittest.TestCase):
    def test_held_out_signer_never_appears_in_train_or_validation(self) -> None:
        rows = _manifest_rows(count_per_cell=4)
        for signer in ("01", "02", "03"):
            assignments = build_loso_splits(rows, signer)
            for row in assignments:
                if row["role"] in {"train", "validation"}:
                    self.assertNotEqual(row["signer_id"], signer)

    def test_test_role_is_exactly_the_held_out_signer(self) -> None:
        rows = _manifest_rows(count_per_cell=4)
        assignments = build_loso_splits(rows, "02")
        test = [r for r in assignments if r["role"] == "test"]
        self.assertTrue(test)
        self.assertEqual({r["signer_id"] for r in test}, {"02"})

    def test_every_sample_is_assigned_exactly_once(self) -> None:
        rows = _manifest_rows(count_per_cell=4)
        assignments = build_loso_splits(rows, "01")
        ids = [r["sample_id"] for r in assignments]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {r["sample_id"] for r in rows})

    def test_an_invalid_held_out_signer_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_loso_splits(_manifest_rows(), "07")


class TestCommittedArtifacts(unittest.TestCase):
    """The committed manifests/splits must stay portable and populated."""

    def test_core28_manifest_is_populated_and_portable(self) -> None:
        path = REPO / "datasets/manifests/karsl_core28.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreater(len(rows), 4000, "manifest must not be schema-only")
        ids = [r["sample_id"] for r in rows]
        paths = [r["source_relative_path"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(len({r["sign_id"] for r in rows}), 28)
        for value in paths:
            self.assertFalse(value.startswith("/"), value)
            self.assertNotIn("/home/", value)

    def test_smoke_manifest_is_deterministic_and_diverse(self) -> None:
        path = REPO / "datasets/manifests/karsl_core28_smoke.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 18)
        self.assertLessEqual(len(rows), 24)
        self.assertEqual(sorted({r["signer_id"] for r in rows}), ["01", "02", "03"])
        self.assertEqual(sorted({r["official_partition"] for r in rows}), ["test", "train"])
        frames = [int(r["frame_count"]) for r in rows]
        self.assertGreater(max(frames), min(frames) * 2, "benchmark subset needs length spread")

    def test_committed_splits_have_no_signer_leakage(self) -> None:
        for signer in ("01", "02", "03"):
            path = REPO / f"datasets/splits/karsl_core28_loso_s{signer}.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 4000)
            for row in rows:
                if row["role"] in {"train", "validation"}:
                    self.assertNotEqual(row["signer_id"], signer)
            test = {r["signer_id"] for r in rows if r["role"] == "test"}
            self.assertEqual(test, {signer})


class TestDatasetMediaExcludedFromGit(unittest.TestCase):
    def test_media_and_bulk_artifacts_are_ignored(self) -> None:
        candidates = [
            "datasets/raw/clip.mp4", "datasets/raw/clip.avi", "datasets/raw/clip.mkv",
            "datasets/raw/0001-0070.7z", "runs/pose/sample/wilor_raw.npz",
            "runs/tracking/sample/wilor_tracked.npz",
        ]
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", *candidates],
            cwd=REPO, capture_output=True, text=True,
        )
        ignored = set(result.stdout.split())
        for candidate in candidates:
            self.assertIn(candidate, ignored, f"{candidate} is not git-ignored")

    def test_manifests_splits_and_reports_are_not_ignored(self) -> None:
        keep = [
            "datasets/manifests/karsl_core28.csv",
            "datasets/manifests/karsl_core28_smoke.csv",
            "datasets/splits/karsl_core28_loso_s01.csv",
            "reports/dataset/TASK-008B-results.json",
        ]
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", *keep],
            cwd=REPO, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout.strip(), "", "tracked artifacts must not be ignored")


if __name__ == "__main__":
    unittest.main()


class TestResumeArtifactValidation(unittest.TestCase):
    """Regression tests for the two resume defects TASK-008B found.

    Both were only visible on a real interrupted run, and both would have made
    a multi-hour extraction far more expensive or outright fragile.
    """

    def setUp(self) -> None:
        import numpy as np

        from evaluation.dataset.orchestrator import RunPaths, _write_stage_sidecar

        self.np = np
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = RunPaths(Path(self.tmp.name))
        self.sample = "karsl_core28_s01_sign0032_train_rep001"
        self.manifest_hash = "a" * 64
        self.source_sha = "b" * 64
        self.row = {"sample_id": self.sample, "source_sha256": self.source_sha}

        # A two-hand video: 3 frames, 2 rows per frame.
        directory = self.paths.stage_dir("POSE", self.sample)
        directory.mkdir(parents=True, exist_ok=True)
        self.artifact = directory / "wilor_raw.npz"
        np.savez_compressed(
            self.artifact,
            frame_index=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
            run_metadata_json=np.array(json.dumps({"mode": "full"})),
        )
        _write_stage_sidecar(self.paths, "POSE", self.sample, {
            "schema_version": "1", "stage": "pose", "status": "POSE_DONE",
            "sample_id": self.sample, "manifest_sha256": self.manifest_hash,
            "source_sha256": self.source_sha, "frames": 3,
            "artifact": str(self.artifact),
        })

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _validate(self):
        from evaluation.dataset.orchestrator import validate_stage_artifact

        return validate_stage_artifact(
            self.paths, "POSE", self.row, self.manifest_hash, source_sha256=self.source_sha
        )

    def test_two_hand_pose_artifact_is_accepted_on_resume(self) -> None:
        """One row per detected hand must not be mistaken for one row per frame.

        Before the fix this comparison failed for every two-hand video, so
        --resume silently recomputed POSE -- 94.9% of the pipeline cost -- for
        work that was already complete.
        """

        result = self._validate()
        self.assertIsNotNone(result, "a complete two-hand POSE stage must be skippable")
        self.assertEqual(result["frames"], 3)

    def test_genuine_frame_count_disagreement_is_still_rejected(self) -> None:
        from evaluation.dataset.orchestrator import _write_stage_sidecar

        _write_stage_sidecar(self.paths, "POSE", self.sample, {
            "schema_version": "1", "stage": "pose", "status": "POSE_DONE",
            "sample_id": self.sample, "manifest_sha256": self.manifest_hash,
            "source_sha256": self.source_sha, "frames": 99,
            "artifact": str(self.artifact),
        })
        self.assertIsNone(self._validate())

    def test_truncated_artifact_is_recomputed_not_fatal(self) -> None:
        """An interrupted write leaves a corrupt archive; it must not kill the run."""

        self.artifact.write_bytes(b"trunc")
        self.assertIsNone(self._validate())

    def test_empty_artifact_is_recomputed_not_fatal(self) -> None:
        self.artifact.write_bytes(b"")
        self.assertIsNone(self._validate())

    def test_missing_artifact_is_recomputed(self) -> None:
        self.artifact.unlink()
        self.assertIsNone(self._validate())

    def test_a_different_source_video_invalidates_the_stage(self) -> None:
        from evaluation.dataset.orchestrator import validate_stage_artifact

        self.assertIsNone(validate_stage_artifact(
            self.paths, "POSE", self.row, self.manifest_hash, source_sha256="c" * 64
        ))

    def test_a_different_manifest_invalidates_the_stage(self) -> None:
        from evaluation.dataset.orchestrator import validate_stage_artifact

        self.assertIsNone(validate_stage_artifact(
            self.paths, "POSE", self.row, "d" * 64, source_sha256=self.source_sha
        ))


class TestProgressDisplay(unittest.TestCase):
    """The long-run progress display must carry every required field."""

    def test_progress_line_contains_the_required_information(self) -> None:
        import io
        from contextlib import redirect_stdout

        from evaluation.dataset.orchestrator import ProgressDisplay

        display = ProgressDisplay(0, 1, 10)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            display.update({"sample_id": "sample_a", "status": "VIRTUAL_GLOVE_DONE",
                            "stage": "VIRTUAL_GLOVE", "frames": 25})
        line = buffer.getvalue()
        for token in ("1/10", "%", "current=sample_a", "stage=", "success=",
                      "failed=", "frames=", "rolling_fps=", "elapsed=", "eta="):
            self.assertIn(token, line, f"progress line is missing {token!r}")

    def test_failures_are_counted_separately(self) -> None:
        import io
        from contextlib import redirect_stdout

        from evaluation.dataset.orchestrator import ProgressDisplay

        display = ProgressDisplay(0, 1, 2)
        with redirect_stdout(io.StringIO()):
            display.update({"sample_id": "a", "status": "FAILED", "frames": 0})
            display.update({"sample_id": "b", "status": "VIRTUAL_GLOVE_DONE", "frames": 5})
        self.assertEqual(display.failed, 1)
        self.assertEqual(display.success, 1)
        self.assertEqual(display.frames, 5)
