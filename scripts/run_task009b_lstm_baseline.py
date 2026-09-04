#!/usr/bin/env python3
"""TASK-009B: train / resume / evaluate one Core-28 LSTM baseline experiment.

One experiment is (feature_set, quaternion_policy, pooling, fold, seed). Each
lands in its own deterministic directory under the external run root, is
resumable with --resume, and is never silently overwritten once complete.

The held-out test signer is loaded only for the final evaluation. It never
influences early stopping, checkpoint selection or any configuration choice.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.data import (  # noqa: E402
    CONTRACT_VERSION,
    SequenceInputConfig,
    VirtualGloveSequenceDataset,
    load_fold,
    load_index,
    make_collate_fn,
)
from recognition.data.labels import load_label_table  # noqa: E402
from recognition.models.lstm_baseline import (  # noqa: E402
    INPUT_POLICIES,
    POOLING_POLICIES,
    LSTMBaseline,
    LSTMBaselineConfig,
)
from recognition.training import (  # noqa: E402
    ExperimentSpec,
    TrainingConfig,
    evaluate,
    load_checkpoint,
    rebuild_model,
    seed_everything,
    train,
    worker_init_fn,
)

DEFAULT_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task009b-core28-lstm")
DEFAULT_DATA_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task008-core28-full")
STATUS_FILE = "status.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT,
                        help="external directory for checkpoints and results (outside git)")
    parser.add_argument("--data-run-root", type=Path, default=DEFAULT_DATA_RUN_ROOT,
                        help="frozen TASK-008 production run root (read-only)")
    parser.add_argument("--index", type=Path,
                        default=ROOT / "datasets/manifests/karsl_core28_virtual_glove.csv")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "datasets/splits")
    parser.add_argument("--labels", type=Path,
                        default=ROOT / "datasets/manifests/karsl_core28_labels.csv")

    parser.add_argument("--fold", default="01", choices=("01", "02", "03"),
                        help="held-out signer")
    parser.add_argument("--feature-set", default="full",
                        choices=("bend_only", "bend_spread", "full"))
    parser.add_argument("--quaternion-policy", default="absolute",
                        choices=("absolute", "relative_first_valid"))
    parser.add_argument("--pooling", default="masked_mean", choices=POOLING_POLICIES)
    parser.add_argument("--input-policy", default="values_and_feature_valid",
                        choices=INPUT_POLICIES)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--input-projection", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted experiment from last.pt")
    parser.add_argument("--status", action="store_true",
                        help="print the state of experiments under --run-root and exit")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="skip training and evaluate an existing checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="checkpoint for --evaluate-only (default: the experiment's best.pt)")
    parser.add_argument("--force", action="store_true",
                        help="re-run an experiment already marked COMPLETE")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="smoke testing only: cap training sequences (evenly strided)")
    parser.add_argument("--limit-eval", type=int, default=None,
                        help="smoke testing only: cap validation/test sequences (evenly strided)")
    parser.add_argument("--preload", action="store_true",
                        help="cache the whole split in RAM (the glove tree is ~134 MiB)")
    return parser


def _subsample(records: list, limit: int | None) -> list:
    """An evenly strided subset, so a smoke run still spans the label space."""

    if not limit or limit >= len(records):
        return list(records)
    stride = max(1, len(records) // limit)
    return list(records[::stride])[:limit]


def experiment_dir(run_root: Path, spec: ExperimentSpec) -> Path:
    quaternion = spec.quaternion_policy if spec.feature_set == "full" else "na"
    return (run_root / spec.feature_set / f"q-{quaternion}" / spec.pooling /
            f"fold{spec.fold}" / f"seed{spec.seed}")


def print_status(run_root: Path) -> int:
    if not run_root.is_dir():
        print(f"run root {run_root} does not exist yet -- no experiments have been started")
        return 0
    rows = sorted(run_root.rglob(STATUS_FILE))
    if not rows:
        print(f"no experiments found under {run_root}")
        return 0
    print(f"{'STATUS':<12} {'EPOCHS':>7} {'BEST_F1':>8} {'BEST_EP':>8}  EXPERIMENT")
    counts: dict[str, int] = {}
    for path in rows:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = payload.get("status", "UNKNOWN")
        counts[state] = counts.get(state, 0) + 1
        # An --evaluate-only run writes no training block, so every field here
        # may legitimately be absent or null.
        training = payload.get("training") or {}
        epochs = training.get("epochs_completed") or 0
        best_f1 = training.get("best_metric")
        best_epoch = training.get("best_epoch") or 0
        best_text = f"{best_f1:.4f}" if isinstance(best_f1, (int, float)) else "--"
        print(f"{state:<12} {epochs:>7} {best_text:>8} {best_epoch:>8}  "
              f"{path.parent.relative_to(run_root)}")
    print()
    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


def make_loader(dataset, config, *, batch_size, shuffle, num_workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=make_collate_fn(config), num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers else None,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.status:
        return print_status(args.run_root)

    spec = ExperimentSpec(
        feature_set=args.feature_set,
        quaternion_policy=args.quaternion_policy,
        pooling=args.pooling,
        fold=args.fold,
        seed=args.seed,
        input_policy=args.input_policy,
    )
    directory = experiment_dir(args.run_root, spec)
    status_path = directory / STATUS_FILE
    if status_path.is_file() and not (args.force or args.evaluate_only or args.resume):
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        if payload.get("status") == "COMPLETE":
            print(f"[skip] {spec.slug()} is already COMPLETE at {directory}")
            print(f"       pass --force to re-run, or --evaluate-only to re-score it")
            print("STATUS:\nCOMPLETE")
            return 0
    directory.mkdir(parents=True, exist_ok=True)

    seed_report = seed_everything(args.seed)
    device = torch.device(args.device)

    data_config = SequenceInputConfig(
        feature_set=args.feature_set,
        quaternion_policy=(args.quaternion_policy if args.feature_set == "full" else "absolute"),
        preload=args.preload,
        # The full corpus layout was verified over all 4,222 samples in TASK-009A;
        # re-checking one sample per split is the cheap runtime guard.
        verify_layout="first",
    )
    records = load_index(args.index)
    fold = load_fold(args.splits_dir, args.fold, records)
    labels_table = load_label_table(args.labels)
    labels_ar = {index: label.label_ar for index, label in labels_table.items()}

    # Smoke limits take an evenly STRIDED subset, not a prefix. Fold rows are
    # sorted by sample_id, so a prefix would be a handful of adjacent classes and
    # would produce meaningless smoke metrics.
    train_records = _subsample(fold.roles["train"], args.limit_train)
    validation_records = _subsample(fold.roles["validation"], args.limit_eval)
    test_records = _subsample(fold.roles["test"], args.limit_eval)
    if args.limit_train or args.limit_eval:
        print("[setup] SMOKE MODE: --limit-train/--limit-eval are active. "
              "Metrics from this run are not scientific results.", flush=True)

    model_config = LSTMBaselineConfig(
        feature_set=args.feature_set, input_policy=args.input_policy, pooling=args.pooling,
        hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout,
        input_projection=args.input_projection,
    )
    model = LSTMBaseline(model_config)

    training_config = TrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, grad_clip_norm=args.grad_clip_norm,
        early_stopping_patience=args.early_stopping_patience, num_workers=args.num_workers,
    )

    validation_dataset = VirtualGloveSequenceDataset(validation_records, args.data_run_root, data_config)
    validation_loader = make_loader(validation_dataset, data_config,
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, seed=args.seed)

    print(f"[setup] contract      : {CONTRACT_VERSION}")
    print(f"[setup] experiment    : {spec.slug()}")
    print(f"[setup] directory     : {directory}")
    print(f"[setup] data run root : {args.data_run_root}")
    print(f"[setup] fold sizes    : {fold.counts()}")
    print(f"[setup] using         : train {len(train_records)} / val {len(validation_records)} "
          f"/ test {len(test_records)}")
    print(f"[setup] seeding       : {seed_report}", flush=True)

    training_result: dict = {}
    if not args.evaluate_only:
        train_dataset = VirtualGloveSequenceDataset(train_records, args.data_run_root, data_config)
        train_loader = make_loader(train_dataset, data_config, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, seed=args.seed)
        training_result = train(
            model=model, spec=spec, train_loader=train_loader,
            validation_loader=validation_loader, device=device, config=training_config,
            output_dir=directory, labels_ar=labels_ar, resume=args.resume,
            seed_report=seed_report,
        )

    checkpoint_path = args.checkpoint or (directory / "best.pt")
    payload = load_checkpoint(checkpoint_path, expect=spec, map_location=device)
    best_model = rebuild_model(payload).to(device)
    print(f"[eval] scoring checkpoint from epoch {payload['best_epoch']} "
          f"({payload['best_metric_name']} {payload['best_metric']:.4f})", flush=True)

    validation_metrics = evaluate(best_model, validation_loader, device,
                                  collect_predictions=True, labels_ar=labels_ar)
    test_dataset = VirtualGloveSequenceDataset(test_records, args.data_run_root, data_config)
    test_loader = make_loader(test_dataset, data_config, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, seed=args.seed)
    test_metrics = evaluate(best_model, test_loader, device,
                            collect_predictions=True, labels_ar=labels_ar)

    predictions = {"validation": validation_metrics.pop("predictions", []),
                   "test": test_metrics.pop("predictions", [])}
    (directory / "predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result = {
        "schema_version": "task009b_result_v1",
        "contract_version": CONTRACT_VERSION,
        "experiment": spec.to_dict(),
        "model_config": model_config.to_dict(),
        "training": training_result,
        "checkpoint": str(checkpoint_path),
        "best_epoch": payload["best_epoch"],
        "best_validation_metric": payload["best_metric"],
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "split_sizes": {"train": len(train_records), "validation": len(validation_records),
                        "test": len(test_records)},
        "sequence_lengths": {
            role: sorted(record.sequence_length for record in group)
            for role, group in (("train", train_records), ("validation", validation_records),
                                ("test", test_records))
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True,
                                         check=False).stdout.strip(),
        },
        "selection_policy": (
            "checkpoint chosen by validation macro F1 only; the held-out test signer "
            "never influenced training, early stopping or model selection"
        ),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    result_path = directory / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    status_path.write_text(json.dumps({
        "status": "COMPLETE",
        "experiment": spec.to_dict(),
        "training": {k: v for k, v in (
            (key, training_result.get(key)) for key in
            ("epochs_completed", "best_epoch", "best_metric", "wall_seconds",
             "seconds_per_epoch", "peak_gpu_mib")) if v is not None},
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "completed_utc": result["completed_utc"],
    }, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"[eval] validation : acc {validation_metrics['accuracy']*100:6.2f}%  "
          f"macroF1 {validation_metrics['macro_f1']:.4f}")
    print(f"[eval] test       : acc {test_metrics['accuracy']*100:6.2f}%  "
          f"macroF1 {test_metrics['macro_f1']:.4f}  (held-out signer {args.fold})")
    print()
    print(f"BEST CHECKPOINT: {checkpoint_path}")
    print(f"RESULT JSON: {result_path}")
    print("STATUS:")
    print("COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
