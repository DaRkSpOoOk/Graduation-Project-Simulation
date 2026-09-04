#!/usr/bin/env python3
"""TASK-009A: audit all 4,222 finalized sequences through the new data contract.

No model is built and no training happens. Every sequence is loaded through the
production dataset path, and the resulting tensor-level statistics are
reconciled against the TASK-008C dataset-level statistics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.data import (  # noqa: E402
    CONTRACT_VERSION,
    FEATURE_SETS,
    FOLD_SIGNERS,
    SequenceContractError,
    SequenceInputConfig,
    VirtualGloveSequenceDataset,
    channel_names,
    collate_sequences,
    family_slice,
    feature_dimension,
    load_all_folds,
    load_index,
)

# TASK-008C dataset-level results, quoted for reconciliation only.
TASK008C_REFERENCE = {"bend": 0.9831637958856788, "spread": 0.7784189387872196,
                      "imu": 0.9831637958856788}


def _percentiles(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size), "min": int(array.min()), "max": int(array.max()),
        "mean": float(array.mean()),
        **{name: float(np.percentile(array, p))
           for name, p in (("p5", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95))},
    }



def _oracle_accuracy(lengths: list[int], targets: list) -> float:
    """Best achievable accuracy using sequence length as the ONLY feature.

    An in-sample upper bound, deliberately optimistic: it assigns every distinct
    length its own majority target. If this is near chance, length carries no
    usable information about the target; if it is far above chance, it does.
    """

    best: dict[int, Counter] = defaultdict(Counter)
    for length, target in zip(lengths, targets):
        best[length][target] += 1
    return sum(c.most_common(1)[0][1] for c in best.values()) / len(lengths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "datasets/manifests/karsl_core28_virtual_glove.csv")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "datasets/splits")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/recognition/TASK-009A-audit.json")
    parser.add_argument("--verify-layout", default="all", choices=("first", "all", "none"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    records = load_index(args.index)
    config = SequenceInputConfig(feature_set="full", verify_layout=args.verify_layout)
    dataset = VirtualGloveSequenceDataset(records, args.run_root, config)

    lengths: list[int] = []
    by_signer: dict[str, list[int]] = defaultdict(list)
    by_class: dict[str, list[int]] = defaultdict(list)
    by_partition: dict[str, list[int]] = defaultdict(list)
    rejected: list[dict[str, str]] = []
    loaded = 0

    bend_valid = bend_total = spread_valid = spread_total = imu_valid = imu_total = 0
    left_present = right_present = both_present = neither_present = 0
    hand_frames = 0
    state_counts: Counter = Counter()
    valid_zero_observations = 0
    invalid_placeholders = 0
    smallest_valid_magnitude = float("inf")
    near_zero_valid = 0
    valid_observations = 0
    finite_everywhere = True

    bend_slices = [family_slice("full", hand, "bend") for hand in ("LEFT", "RIGHT")]
    spread_slices = [family_slice("full", hand, "spread") for hand in ("LEFT", "RIGHT")]
    quat_slices = [family_slice("full", hand, "quaternion") for hand in ("LEFT", "RIGHT")]

    for position in range(len(dataset)):
        record = dataset.records[position]
        try:
            item = dataset[position]
        except SequenceContractError as error:
            rejected.append({"sample_id": record.sample_id, "error": str(error)})
            continue
        loaded += 1
        length = item["length"]
        lengths.append(length)
        by_signer[record.signer_id].append(length)
        by_class[record.sign_id].append(length)
        by_partition[record.official_partition].append(length)

        values = item["values"]
        valid = item["feature_valid"]
        if not np.isfinite(values).all():
            finite_everywhere = False
        # A stored 0.0 that is still marked valid: the case the contract must
        # keep distinguishable from a zero-filled placeholder.
        valid_zero_observations += int(((values == 0.0) & valid).sum())
        invalid_placeholders += int((~valid).sum())
        observed = np.abs(values[valid])
        if observed.size:
            smallest_valid_magnitude = min(smallest_valid_magnitude, float(observed.min()))
            near_zero_valid += int((observed < 1e-6).sum())
            valid_observations += int(observed.size)

        for block in bend_slices:
            bend_valid += int(valid[:, block].sum()); bend_total += valid[:, block].size
        for block in spread_slices:
            spread_valid += int(valid[:, block].sum()); spread_total += valid[:, block].size
        for block in quat_slices:
            # One IMU package per hand: count packages, not the 4 broadcast columns.
            imu_valid += int(valid[:, block.start].sum()); imu_total += valid.shape[0]

        present = item["hand_present"]
        left, right = present[:, 0], present[:, 1]
        left_present += int(left.sum()); right_present += int(right.sum())
        both_present += int((left & right).sum())
        neither_present += int((~left & ~right).sum())
        hand_frames += int(present.size)
        for code, count in zip(*np.unique(item["tracking_state_code"], return_counts=True)):
            state_counts[int(code)] += int(count)

    folds = load_all_folds(args.splits_dir, records)
    fold_report = {}
    for signer, fold in folds.items():
        role_lengths = {role: [r.sequence_length for r in fold.roles[role]] for role in fold.roles}
        fold_report[signer] = {
            "counts": fold.counts(),
            "signers_per_role": {role: sorted(fold.signers(role)) for role in fold.roles},
            "classes_per_role": {role: len(fold.classes(role)) for role in fold.roles},
            "held_out_signer_in_train_or_validation": any(
                signer in fold.signers(role) for role in ("train", "validation")
            ),
            "sequence_length_by_role": {r: _percentiles(v) for r, v in role_lengths.items() if v},
        }

    # A representative batch, to measure real padding overhead rather than guess.
    batch_report = {}
    position_of = {record.sample_id: i for i, record in enumerate(dataset.records)}
    for signer, fold in folds.items():
        items = [dataset[position_of[record.sample_id]] for record in fold.roles["train"][:32]]
        batch = collate_sequences(items, config)
        real = int(batch["frame_valid"].sum())
        cells = int(batch["frame_valid"].numel())
        batch_report[f"S{signer}_train_first32"] = {
            "values_shape": list(batch["values"].shape),
            "feature_valid_shape": list(batch["feature_valid"].shape),
            "hand_present_shape": list(batch["hand_present"].shape),
            "frame_valid_shape": list(batch["frame_valid"].shape),
            "lengths_shape": list(batch["lengths"].shape),
            "labels_shape": list(batch["labels"].shape),
            "min_length": int(batch["lengths"].min()),
            "max_length": int(batch["lengths"].max()),
            "real_timesteps": real,
            "padded_timesteps": cells - real,
            "padding_fraction": (cells - real) / cells,
        }

    # Phase J: does duration alone identify the signer, and does it identify the
    # letter? The gap between those two answers is what makes duration a
    # nuisance variable rather than signal.
    all_lengths = [record.sequence_length for record in records]
    signers = [record.signer_id for record in records]
    classes = [record.label_index for record in records]
    duration_audit = {
        "signer_from_length_oracle_accuracy": _oracle_accuracy(all_lengths, signers),
        "signer_majority_baseline": max(Counter(signers).values()) / len(signers),
        "class_from_length_oracle_accuracy": _oracle_accuracy(all_lengths, classes),
        "class_majority_baseline": max(Counter(classes).values()) / len(classes),
        "per_fold": {},
    }
    length_array = np.asarray(all_lengths, dtype=np.float64)
    signer_array = np.asarray(signers)
    for signer in FOLD_SIGNERS:
        train_lengths = length_array[signer_array != signer]
        test_lengths = length_array[signer_array == signer]
        low, high = np.percentile(train_lengths, [5, 95])
        duration_audit["per_fold"][signer] = {
            "train_mean": float(train_lengths.mean()),
            "test_mean": float(test_lengths.mean()),
            "test_over_train_mean_ratio": float(test_lengths.mean() / train_lengths.mean()),
            "train_p5_p95": [float(low), float(high)],
            "test_fraction_inside_train_p5_p95": float(
                ((test_lengths >= low) & (test_lengths <= high)).mean()
            ),
        }

    payload = {
        "contract_version": CONTRACT_VERSION,
        "signer_duration_audit": duration_audit,
        "index": str(args.index),
        "run_root": str(args.run_root.resolve()),
        "coverage": {
            "total_indexed": len(records),
            "successfully_loaded": loaded,
            "rejected": len(rejected),
            "rejections": rejected[:20],
        },
        "feature_dimensions": {name: feature_dimension(name) for name in FEATURE_SETS},
        "channel_names_full": channel_names("full"),
        "sequence_lengths": {
            "overall": _percentiles(lengths),
            "by_signer": {k: _percentiles(v) for k, v in sorted(by_signer.items())},
            "by_partition": {k: _percentiles(v) for k, v in sorted(by_partition.items())},
            "by_class": {k: _percentiles(v) for k, v in sorted(by_class.items())},
        },
        "validity": {
            "bend": {"valid": bend_valid, "total": bend_total, "fraction": bend_valid / bend_total},
            "spread": {"valid": spread_valid, "total": spread_total, "fraction": spread_valid / spread_total},
            "imu_packages": {"valid": imu_valid, "total": imu_total, "fraction": imu_valid / imu_total},
            "task008c_reference": TASK008C_REFERENCE,
        },
        "hand_availability": {
            "hand_instances": hand_frames,
            "left_present": left_present, "right_present": right_present,
            "both_present": both_present, "neither_present": neither_present,
            "left_only": left_present - both_present,
            "right_only": right_present - both_present,
        },
        "tracking_states": {str(code): count for code, count in sorted(state_counts.items())},
        "value_integrity": {
            "all_tensor_values_finite": finite_everywhere,
            "valid_zero_observations": valid_zero_observations,
            "invalid_zero_filled_placeholders": invalid_placeholders,
            "valid_observations": valid_observations,
            "smallest_valid_magnitude": smallest_valid_magnitude,
            "valid_observations_within_1e-6_of_zero": near_zero_valid,
            "distinguishable": True,
            "evidence": (
                "both populations hold the value 0.0; they are separated only by "
                "feature_valid, which is why the mask is part of the contract"
            ),
        },
        "loso": fold_report,
        "representative_batches": batch_report,
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True, check=False).stdout.strip(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"Wrote {args.output} in {payload['elapsed_seconds']:.1f}s "
          f"({loaded}/{len(records)} loaded, {len(rejected)} rejected)")
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
