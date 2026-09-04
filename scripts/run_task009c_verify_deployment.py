#!/usr/bin/env python3
"""TASK-009C: verify and finalize the persisted all-signers deployment artifact.

Reads only the completed deployment run. Trains nothing, modifies nothing, and
produces no performance estimate -- a model fitted on every available sequence has
no held-out data from which one could be produced.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.data.contract import CONTRACT_VERSION  # noqa: E402
from recognition.data.labels import load_label_table  # noqa: E402
from recognition.data.sequence_dataset import load_sequence_arrays  # noqa: E402
from recognition.data import load_index  # noqa: E402
from recognition.deployment import (  # noqa: E402
    DEPLOYMENT_CHECKPOINT,
    HASHED_FILES,
    RESUME_CHECKPOINT,
    sha256_file,
    verify_checkpoint_metadata,
    verify_files,
    verify_history,
    verify_plan,
    verify_status,
)
from recognition.models import SequenceRecognizer  # noqa: E402
from recognition.training import load_checkpoint  # noqa: E402

DEFAULT_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task009c-core28-deployment")
DEFAULT_DATA_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task008-core28-full")
DEFAULT_RESEARCH_CHECKPOINT = Path(
    "/home/hatim/graduation-project-runs/task009b-core28-lstm/full/q-absolute/"
    "masked_mean/fold01/seed1337/best.pt")
# Deterministic, preselected before any prediction was seen: the first row of the
# frozen index. Not chosen because it classifies correctly.
PROBE_INDEX_ROW = 0


def plot_training(history, path: Path) -> None:
    epochs = [record["epoch"] for record in history]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    axes[0].plot(epochs, [r["train_loss"] for r in history], "o-", color="#4C78A8", markersize=3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("IN-SAMPLE deployment training loss", fontsize=9.5)
    axes[1].plot(epochs, [r["train_accuracy"] * 100 for r in history], "o-",
                 color="#54A24B", markersize=3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("IN-SAMPLE deployment training accuracy", fontsize=9.5)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "TASK-009C all-signers deployment fit — IN-SAMPLE TRAINING CURVES\n"
        "these are NOT validation or generalization curves; the model has no held-out data",
        fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--data-run-root", type=Path, default=DEFAULT_DATA_RUN_ROOT)
    parser.add_argument("--index", type=Path,
                        default=ROOT / "datasets/manifests/karsl_core28_virtual_glove.csv")
    parser.add_argument("--labels", type=Path,
                        default=ROOT / "datasets/manifests/karsl_core28_labels.csv")
    parser.add_argument("--research-checkpoint", type=Path,
                        default=DEFAULT_RESEARCH_CHECKPOINT)
    parser.add_argument("--committed-plan", type=Path,
                        default=ROOT / "reports/recognition/TASK-009C-deployment-plan.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/recognition/TASK-009C-deployment-artifact.json")
    parser.add_argument("--figure", type=Path,
                        default=ROOT / "reports/recognition/figures/task009c-deployment-training.png")
    args = parser.parse_args(argv)

    print(f"[verify] run root: {args.run_root}", flush=True)
    files = verify_files(args.run_root)
    plan = verify_plan(args.run_root, args.committed_plan)
    epochs = plan["deployment_epochs"]
    status = verify_status(args.run_root, epochs)
    history_summary = verify_history(args.run_root, epochs)

    payload = load_checkpoint(args.run_root / DEPLOYMENT_CHECKPOINT)
    metadata = verify_checkpoint_metadata(payload, epochs)

    hashes = {name: sha256_file(args.run_root / name) for name in HASHED_FILES
              if (args.run_root / name).is_file()}
    sizes = {name: (args.run_root / name).stat().st_size
             for name in (DEPLOYMENT_CHECKPOINT, RESUME_CHECKPOINT)
             if (args.run_root / name).is_file()}

    # Compatibility: the deployment checkpoint and an older research checkpoint
    # must both serve inference through the same unchanged recognizer.
    labels = load_label_table(args.labels)
    records = load_index(args.index)
    probe = records[PROBE_INDEX_ROW]
    arrays = load_sequence_arrays(args.data_run_root / probe.virtual_glove_relative_path)
    compatibility: dict[str, object] = {
        "probe_selection": (
            f"row {PROBE_INDEX_ROW} of the frozen index, chosen before any prediction "
            "was seen"),
        "probe_sample_id": probe.sample_id,
        "probe_true_label_index": probe.label_index,
        "probe_true_label_ar": probe.label_ar,
        "probe_true_sign_id": probe.sign_id,
        "probe_signer_id": probe.signer_id,
        "probe_sequence_length": probe.sequence_length,
        # The probe proves the plumbing -- load, tensorize, 28 logits, label
        # resolution -- and nothing about quality. It is IN-SAMPLE for the
        # deployment model, which was fitted on every sequence including this one.
        # For the TASK-009B fold-01 checkpoint the same sequence is HELD OUT,
        # since signer 01 was that fold's test signer. The two predictions are
        # therefore not comparable, and neither is a performance estimate.
        "probe_relationship": {
            "deployment": "IN-SAMPLE (this sequence was in the deployment training set)",
            "task009b_research": (
                "HELD OUT (signer 01 was the fold-01 test signer) -- but a single "
                "sequence, not a performance estimate"),
        },
    }
    for name, path in (("deployment", args.run_root / DEPLOYMENT_CHECKPOINT),
                       ("task009b_research", args.research_checkpoint)):
        if not Path(path).is_file():
            compatibility[name] = {"loaded": False, "reason": f"{path} not present"}
            continue
        recognizer = SequenceRecognizer.from_checkpoint(path, label_table=args.labels)
        prediction = recognizer.predict_sequence(arrays)
        compatibility[name] = {
            "loaded": True,
            "checkpoint": str(path),
            "feature_set": recognizer.config.feature_set,
            "quaternion_policy": recognizer.config.quaternion_policy,
            "predicted_label_index": prediction.label_index,
            "predicted_label_ar": prediction.label_ar,
            "predicted_sign_id": prediction.sign_id,
            "confidence": prediction.confidence,
            "logits_length": len(prediction.logits),
            "probabilities_sum": float(sum(prediction.probabilities)),
            "label_mapping_consistent": (
                prediction.label_ar == labels[prediction.label_index].label_ar
                and prediction.sign_id == labels[prediction.label_index].sign_id),
            "matches_probe_ground_truth": prediction.label_index == probe.label_index,
        }

    history = json.loads((args.run_root / "history.json").read_text(encoding="utf-8"))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plot_training(history, args.figure)

    summary = json.loads((args.run_root / "training_summary.json").read_text(encoding="utf-8"))
    checks = {"files": files["passed"], "status": status["passed"], "plan": plan["passed"],
              "history": history_summary["passed"], "checkpoint_metadata": metadata["passed"]}
    all_problems = (files["problems"] + status["problems"] + plan["problems"]
                    + history_summary["problems"] + metadata["problems"])

    artifact = {
        "schema_version": "task009c_deployment_artifact_v1",
        "task": "TASK-009C",
        "checkpoint_filename": DEPLOYMENT_CHECKPOINT,
        "checkpoint_path": str(args.run_root / DEPLOYMENT_CHECKPOINT),
        "checkpoint_sha256": hashes.get(DEPLOYMENT_CHECKPOINT),
        "checkpoint_size_bytes": sizes.get(DEPLOYMENT_CHECKPOINT),
        "artifact_sha256": hashes,
        "epoch": metadata["epoch"],
        "epochs_planned": epochs,
        "epoch_policy": plan["epoch_policy"],
        "source_primary_best_epochs": plan["source_best_epochs"],
        "seed": metadata["experiment"].get("seed"),
        "samples": plan["training_samples"],
        "signers": sorted(plan["signers"]),
        "classes": plan["classes"],
        "configuration": plan["configuration"],
        "training_config": plan["training_config"],
        "parameter_count": summary.get("parameter_count"),
        "input_dim": metadata["input_dim"],
        "contract_version": CONTRACT_VERSION,
        "source_task009b_commit": metadata["extra"].get("source_task009b_analysis_commit"),
        "source_task009c_commit": summary.get("environment", {}).get("git_commit"),
        "verification_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
            check=False).stdout.strip(),
        "in_sample_training": {
            "final_train_loss": history_summary.get("final_train_loss"),
            "final_train_accuracy": history_summary.get("final_train_accuracy"),
            "initial_train_loss": history_summary.get("initial_train_loss"),
            "initial_train_accuracy": history_summary.get("initial_train_accuracy"),
            "warning": (
                "IN-SAMPLE values on the data the model was fitted to. NOT a performance "
                "estimate and never to be quoted as deployment accuracy."
            ),
        },
        "scientific_loso_reference": metadata["loso_reference"],
        "scientific_status": payload["extra"].get("scientific_status"),
        "runtime": {
            "wall_seconds": summary.get("wall_seconds"),
            "seconds_per_epoch": summary.get("seconds_per_epoch"),
            "peak_gpu_mib": summary.get("peak_gpu_mib"),
            "environment": summary.get("environment"),
        },
        "verification": {
            "checks": checks, "all_passed": not all_problems, "problems": all_problems,
            "files": files["files"], "status": {k: v for k, v in status.items()
                                                if k != "problems"},
            "history": {k: v for k, v in history_summary.items() if k != "problems"},
            "plan_matches_committed_copy": plan["matches_committed_plan"],
            "compatibility": compatibility,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True)
                           + "\n", encoding="utf-8")

    print()
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    print(f"  deployment.pt SHA-256 : {artifact['checkpoint_sha256']}")
    print(f"  size                  : {artifact['checkpoint_size_bytes']:,} bytes")
    print(f"  epochs                : {metadata['epoch']}/{epochs}")
    print(f"  final in-sample       : loss {history_summary['final_train_loss']:.4f}, "
          f"accuracy {history_summary['final_train_accuracy']*100:.2f}% (NOT a performance estimate)")
    print(f"  LOSO reference        : "
          f"{metadata['loso_reference']['mean_test_accuracy']*100:.2f}% accuracy / "
          f"{metadata['loso_reference']['mean_test_macro_f1']:.4f} macro F1")
    print()
    print(f"ARTIFACT MANIFEST: {args.output}")
    print(f"FIGURE: {args.figure}")
    print("STATUS:")
    print("VERIFIED" if not all_problems else "PROBLEMS FOUND", flush=True)
    return 0 if not all_problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
