#!/usr/bin/env python3
"""TASK-009C: prepare and run the all-signers Core-28 deployment training.

Two modes:

  --prepare   inspect the completed TASK-009B primary LOSO runs, derive the frozen
              epoch budget, audit all 4,222 sequences, and write deployment_plan.json.
              Performs no training.

  (default)   read the frozen plan and fit one model on every sequence for exactly
              the planned number of epochs, then write deployment.pt.

This is POST-EVALUATION deployment training, not a new scientific evaluation. The
resulting model has no held-out data; its training accuracy is not a performance
estimate. The recognition result for this project remains the TASK-009B LOSO
evidence, which is copied into the plan and into the checkpoint metadata.
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
    load_index,
    make_collate_fn,
)
from recognition.deployment import (  # noqa: E402
    DEPLOYMENT_CHECKPOINT,
    DeploymentPlanError,
    build_deployment_plan,
    load_plan,
    train_deployment_model,
    write_plan,
)
from recognition.models.lstm_baseline import LSTMBaseline, LSTMBaselineConfig  # noqa: E402
from recognition.training import seed_everything, worker_init_fn  # noqa: E402

DEFAULT_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task009c-core28-deployment")
DEFAULT_DATA_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task008-core28-full")
DEFAULT_LOSO_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task009b-core28-lstm")
PLAN_NAME = "deployment_plan.json"
STATUS_NAME = "status.json"
SUMMARY_NAME = "training_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT,
                        help="external deployment output directory (outside git)")
    parser.add_argument("--data-run-root", type=Path, default=DEFAULT_DATA_RUN_ROOT,
                        help="frozen TASK-008 production run root (read-only)")
    parser.add_argument("--loso-run-root", type=Path, default=DEFAULT_LOSO_RUN_ROOT,
                        help="completed TASK-009B sweep, read to derive the epoch budget")
    parser.add_argument("--index", type=Path,
                        default=ROOT / "datasets/manifests/karsl_core28_virtual_glove.csv")
    parser.add_argument("--labels", type=Path,
                        default=ROOT / "datasets/manifests/karsl_core28_labels.csv")
    parser.add_argument("--deployment-plan", type=Path, default=None,
                        help=f"plan path (default: <run-root>/{PLAN_NAME})")

    parser.add_argument("--prepare", action="store_true",
                        help="derive and freeze the deployment plan; no training")
    parser.add_argument("--force-plan", action="store_true",
                        help="overwrite an existing frozen plan (deliberate act)")
    parser.add_argument("--status", action="store_true",
                        help="print deployment state and exit")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted deployment run")
    parser.add_argument("--force", action="store_true",
                        help="re-run a deployment training already marked COMPLETE")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the plan seed (normally left alone)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override the plan batch size (normally left alone)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None,
                        help="smoke testing only: evenly strided subset of the training data")
    parser.add_argument("--smoke-epochs", type=int, default=None,
                        help="smoke testing only: override the frozen epoch budget")
    return parser


def _plan_path(args: argparse.Namespace) -> Path:
    return args.deployment_plan or (args.run_root / PLAN_NAME)


def _subsample(records: list, limit: int | None) -> list:
    """Evenly strided subset, so a smoke run still spans the label space."""

    if not limit or limit >= len(records):
        return list(records)
    stride = max(1, len(records) // limit)
    return list(records[::stride])[:limit]


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=False).stdout.strip()


def do_prepare(args: argparse.Namespace) -> int:
    path = _plan_path(args)
    print(f"[prepare] reading TASK-009B primary results from {args.loso_run_root}", flush=True)
    print(f"[prepare] auditing all deployment sequences (loads every sequence)", flush=True)
    plan = build_deployment_plan(
        loso_run_root=args.loso_run_root,
        index_path=args.index,
        data_run_root=args.data_run_root,
        seed=args.seed if args.seed is not None else 1337,
        batch_size=args.batch_size if args.batch_size is not None else 32,
        task009b_analysis_commit=_git_commit(),
    )
    write_plan(path, plan, force=args.force_plan)

    budget = plan["epoch_budget"]
    audit = plan["dataset_audit"]
    reference = plan["loso_reference"]
    print()
    print("=" * 68)
    print("TASK-009C DEPLOYMENT PLAN")
    print("=" * 68)
    print(f"  contract            : {plan['contract_version']}")
    print(f"  configuration       : {plan['configuration']['feature_set']} / "
          f"q={plan['configuration']['quaternion_policy']} / "
          f"{plan['configuration']['pooling']} / {plan['configuration']['input_policy']}")
    print(f"  model input dim     : {plan['configuration']['model_input_dimension']}")
    print()
    print("  primary LOSO best epochs (validation-selected, pre-test):")
    for fold, value in budget["source_best_epochs"].items():
        print(f"      S{fold} = {value}")
    print(f"  median              : {budget['raw_median']}")
    print(f"  FROZEN EPOCHS       : {budget['deployment_epochs']}   ({budget['policy']})")
    print()
    print(f"  training samples    : {audit['indexed_samples']} "
          f"(loaded {audit['loaded_samples']}, rejected {audit['rejected_samples']})")
    print(f"  signers             : {dict(audit['signers'])}")
    print(f"  classes             : {audit['classes']}")
    print()
    print(f"  LOSO reference      : accuracy {reference['mean_test_accuracy']*100:.2f}%  "
          f"macro F1 {reference['mean_test_macro_f1']:.4f}")
    print(f"  scientific status   : POST-EVALUATION DEPLOYMENT TRAINING (no held-out data)")
    print("=" * 68)
    print()
    print(f"DEPLOYMENT PLAN: {path}")
    print("STATUS:")
    print("PREPARED", flush=True)
    return 0


def do_status(args: argparse.Namespace) -> int:
    root = args.run_root
    plan_path = _plan_path(args)
    print(f"run root       : {root}")
    if not root.is_dir():
        print("state          : NOT STARTED (run root does not exist)")
        return 0
    if plan_path.is_file():
        plan = load_plan(plan_path)
        print(f"plan           : {plan_path}")
        print(f"frozen epochs  : {plan['epoch_budget']['deployment_epochs']} "
              f"({plan['epoch_budget']['policy']})")
        print(f"configuration  : {plan['configuration']['feature_set']} / "
              f"q={plan['configuration']['quaternion_policy']} / "
              f"{plan['configuration']['pooling']}")
    else:
        print("plan           : ABSENT -- run with --prepare first")

    status_path = root / STATUS_NAME
    if status_path.is_file():
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        print(f"state          : {payload.get('status')}")
        print(f"epochs         : {payload.get('epochs_completed')}/{payload.get('epochs_planned')}")
        print(f"wall seconds   : {payload.get('wall_seconds')}")
        print(f"checkpoint     : {payload.get('deployment_checkpoint')}")
    else:
        last = root / "last.pt"
        if last.is_file():
            from recognition.training import load_checkpoint
            payload = load_checkpoint(last)
            print(f"state          : IN PROGRESS (epoch {payload['epoch']})")
        else:
            print("state          : NOT STARTED (no checkpoint yet)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.status:
        return do_status(args)
    if args.prepare:
        return do_prepare(args)

    plan_path = _plan_path(args)
    plan = load_plan(plan_path)
    status_path = args.run_root / STATUS_NAME
    if status_path.is_file() and not (args.force or args.resume):
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        if payload.get("status") == "COMPLETE":
            print(f"[skip] deployment training is already COMPLETE at {args.run_root}")
            print(f"       checkpoint: {payload.get('deployment_checkpoint')}")
            print("       pass --force to retrain from scratch")
            print("STATUS:\nCOMPLETE")
            return 0

    seed = args.seed if args.seed is not None else int(plan["training_config"]["seed"])
    batch_size = (args.batch_size if args.batch_size is not None
                  else int(plan["training_config"]["batch_size"]))
    if args.smoke_epochs is not None:
        plan = dict(plan)
        plan["epoch_budget"] = {**plan["epoch_budget"], "deployment_epochs": args.smoke_epochs}
        print("[setup] SMOKE MODE: the frozen epoch budget is overridden. "
              "This run is not a deployment model.", flush=True)

    seed_report = seed_everything(seed)
    device = torch.device(args.device)
    configuration = plan["configuration"]

    data_config = SequenceInputConfig(
        feature_set=configuration["feature_set"],
        quaternion_policy=configuration["quaternion_policy"],
        verify_layout="first",
    )
    records = load_index(args.index)
    if args.limit_samples:
        records = _subsample(records, args.limit_samples)
        print("[setup] SMOKE MODE: --limit-samples is active. "
              "This run is not a deployment model.", flush=True)
    elif len(records) != plan["dataset_audit"]["indexed_samples"]:
        raise DeploymentPlanError(
            f"index holds {len(records)} sequences but the frozen plan audited "
            f"{plan['dataset_audit']['indexed_samples']}; the data changed since the plan "
            "was frozen")

    dataset = VirtualGloveSequenceDataset(records, args.data_run_root, data_config)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=make_collate_fn(data_config),
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers else None,
        generator=generator, drop_last=False,
    )

    model = LSTMBaseline(LSTMBaselineConfig(
        feature_set=configuration["feature_set"],
        input_policy=configuration["input_policy"],
        pooling=configuration["pooling"],
        hidden_size=int(configuration["hidden_size"]),
        num_layers=int(configuration["num_layers"]),
        dropout=float(configuration["dropout"]),
        input_projection=configuration["input_projection"],
        bidirectional=bool(configuration["bidirectional"]),
    ))

    print(f"[setup] contract      : {CONTRACT_VERSION}")
    print(f"[setup] plan          : {plan_path}")
    print(f"[setup] run root      : {args.run_root}")
    print(f"[setup] data run root : {args.data_run_root}")
    print(f"[setup] sequences     : {len(records)}")
    print(f"[setup] seeding       : {seed_report}", flush=True)

    summary = train_deployment_model(
        model=model, plan=plan, loader=loader, device=device, output_dir=args.run_root,
        resume=args.resume, seed_report=seed_report, training_samples=len(records),
    )
    summary["environment"] = {
        "python": platform.python_version(), "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "git_commit": _git_commit(),
    }
    summary["plan"] = str(plan_path)
    summary["loso_reference"] = plan["loso_reference"]
    summary["scientific_status"] = plan["scientific_status"]
    summary["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["smoke_mode"] = bool(args.limit_samples or args.smoke_epochs)

    (args.run_root / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    status_path.write_text(json.dumps({
        "status": "COMPLETE",
        "training_role": summary["training_role"],
        "epochs_planned": summary["epochs_planned"],
        "epochs_completed": summary["epochs_completed"],
        "wall_seconds": summary["wall_seconds"],
        "deployment_checkpoint": summary["deployment_checkpoint"],
        "smoke_mode": summary["smoke_mode"],
        "completed_utc": summary["completed_utc"],
    }, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"[deployment] final in-sample train loss {summary['final_train_loss']:.4f}, "
          f"accuracy {summary['final_train_accuracy']*100:.2f}% "
          f"(IN-SAMPLE, not a performance estimate)")
    print(f"[deployment] generalization reference remains TASK-009B LOSO: "
          f"{plan['loso_reference']['mean_test_accuracy']*100:.2f}% accuracy / "
          f"{plan['loso_reference']['mean_test_macro_f1']:.4f} macro F1")
    print()
    print(f"DEPLOYMENT CHECKPOINT: {summary['deployment_checkpoint']}")
    print(f"TRAINING SUMMARY: {args.run_root / SUMMARY_NAME}")
    print("STATUS:")
    print("COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
