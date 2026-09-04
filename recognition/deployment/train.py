"""TASK-009C all-signers deployment training.

A fixed-length fit on every Core-28 sequence. There is deliberately no validation
loader, no early stopping and no checkpoint selection: the epoch budget was frozen
in the deployment plan from TASK-009B evidence, so the model at the final epoch IS
the deployment model. That is why the artifact is named ``deployment.pt`` and not
``best.pt`` -- nothing here selected a "best".

Training loss and accuracy are printed for runtime and debugging visibility. They
are in-sample numbers on data the model is fitting and are not a performance
estimate of any kind.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch

from ..models.lstm_baseline import LSTMBaseline
from ..training.checkpoint import ExperimentSpec, load_checkpoint, save_checkpoint
from ..training.determinism import random_state, restore_random_state
from ..training.trainer import TrainingConfig, build_optimizer, train_one_epoch
from .plan import DEPLOYMENT_FOLD_TAG, DEPLOYMENT_TRAINING_ROLE, EPOCH_POLICY

# No validation set exists, so no metric selected this checkpoint. Recorded
# explicitly instead of leaving the research default in place, which would imply
# a selection that never happened.
DEPLOYMENT_METRIC_NAME = "not_applicable_fixed_epoch_budget"
DEPLOYMENT_CHECKPOINT = "deployment.pt"
RESUME_CHECKPOINT = "last.pt"


def deployment_spec(plan: Mapping[str, Any]) -> ExperimentSpec:
    """The checkpoint identity of a deployment model.

    Reuses the TASK-009B schema unchanged so the existing runtime loads it with
    no code change. ``fold`` carries the tag ``"all"`` rather than a fold number,
    which is honest -- there is no held-out signer -- and is accepted because the
    loader treats fold as run identity, never as a compatibility constraint.
    """

    configuration = plan["configuration"]
    return ExperimentSpec(
        feature_set=configuration["feature_set"],
        quaternion_policy=configuration["quaternion_policy"],
        pooling=configuration["pooling"],
        fold=DEPLOYMENT_FOLD_TAG,
        seed=int(plan["training_config"]["seed"]),
        input_policy=configuration["input_policy"],
        contract_version=plan["contract_version"],
    )


def deployment_metadata(plan: Mapping[str, Any], *, epochs_completed: int) -> dict[str, Any]:
    """Deployment provenance carried in the checkpoint's ``extra`` block."""

    audit = plan["dataset_audit"]
    return {
        "training_role": DEPLOYMENT_TRAINING_ROLE,
        "training_scope": plan["training_scope"],
        "training_samples": audit["indexed_samples"],
        "signers": sorted(audit["signers"]),
        "classes": audit["classes"],
        "epoch_policy": EPOCH_POLICY,
        "deployment_epochs": plan["epoch_budget"]["deployment_epochs"],
        "source_primary_best_epochs": plan["epoch_budget"]["source_best_epochs"],
        "epochs_completed": epochs_completed,
        "source_task009b_analysis_commit": plan.get("source_task009b_analysis_commit", ""),
        "loso_reference": plan["loso_reference"],
        "scientific_status": plan["scientific_status"],
        "held_out_data": "none -- fitted on every available sequence",
        "selection_metric": DEPLOYMENT_METRIC_NAME,
    }


def _format_clock(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN-safe
        return "--:--:--"
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def train_deployment_model(
    *,
    model: LSTMBaseline,
    plan: Mapping[str, Any],
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    output_dir: str | Path,
    resume: bool = False,
    seed_report: Mapping[str, Any] | None = None,
    training_samples: int | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fit for exactly the frozen number of epochs, then save ``deployment.pt``."""

    emit = log or (lambda message: print(message, flush=True))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    last_path = directory / RESUME_CHECKPOINT
    deployment_path = directory / DEPLOYMENT_CHECKPOINT

    total_epochs = int(plan["epoch_budget"]["deployment_epochs"])
    training = plan["training_config"]
    config = TrainingConfig(
        epochs=total_epochs,
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        grad_clip_norm=float(training["grad_clip_norm"]),
        # No validation set exists, so early stopping cannot and must not fire.
        early_stopping_patience=total_epochs + 1,
        optimizer=training["optimizer"],
        selection_metric=DEPLOYMENT_METRIC_NAME,
    )
    spec = deployment_spec(plan)

    model.to(device)
    optimizer = build_optimizer(model, config)
    start_epoch = 1
    history: list[dict[str, Any]] = []

    if resume and last_path.is_file():
        payload = load_checkpoint(last_path, expect=spec, map_location=device)
        stored_total = payload.get("extra", {}).get("deployment_epochs")
        if stored_total is not None and int(stored_total) != total_epochs:
            raise ValueError(
                f"cannot resume: checkpoint was training to {stored_total} epochs, the plan "
                f"now says {total_epochs}. The frozen plan must not change mid-run."
            )
        model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        history = list(payload.get("history", []))
        restored = restore_random_state(payload.get("rng_state"))
        emit(f"[resume] continuing at epoch {start_epoch}/{total_epochs} "
             f"(rng_restored={restored})")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    emit(f"[deployment] all-signers fit | {plan['configuration']['feature_set']} | "
         f"q={plan['configuration']['quaternion_policy']} | "
         f"pool={plan['configuration']['pooling']} | seed {spec.seed}")
    # Report the sequences actually being fitted, which differs from the plan's
    # audited total only in smoke mode.
    samples = (training_samples if training_samples is not None
               else plan["dataset_audit"]["indexed_samples"])
    emit(f"[deployment] device={device} params={model.parameter_count():,} "
         f"input_dim={model.config.input_dim} "
         f"samples={samples} epochs={total_epochs} (frozen by {EPOCH_POLICY})")

    started = time.perf_counter()
    completed = start_epoch - 1
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_started = time.perf_counter()
        stats = train_one_epoch(model, loader, optimizer, device, config)
        completed = epoch
        elapsed = time.perf_counter() - started
        per_epoch = elapsed / max(1, epoch - start_epoch + 1)
        remaining = per_epoch * (total_epochs - epoch)
        memory = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                  if device.type == "cuda" else None)

        entry = {
            "epoch": epoch,
            "train_loss": stats["loss"],
            "train_accuracy": stats["accuracy"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(entry)
        emit(
            f"[deployment epoch {epoch:3d}/{total_epochs}] "
            f"loss {stats['loss']:.4f} | acc {stats['accuracy']*100:6.2f}% | "
            f"{entry['epoch_seconds']:.1f}s | elapsed {_format_clock(elapsed)} | "
            f"ETA {_format_clock(remaining)}"
            + (f" | GPU {memory:.0f} MiB" if memory is not None else "")
        )

        save_checkpoint(
            last_path, model=model, spec=spec, optimizer_state=optimizer.state_dict(),
            epoch=epoch, best_epoch=epoch, best_metric=0.0,
            best_metric_name=DEPLOYMENT_METRIC_NAME, history=history,
            training_config={**config.to_dict(), **dict(seed_report or {})},
            rng_state=random_state(),
            extra=deployment_metadata(plan, epochs_completed=epoch),
        )

    # The final-epoch model IS the deployment model: nothing selected it, the
    # budget did.
    save_checkpoint(
        deployment_path, model=model, spec=spec, optimizer_state=optimizer.state_dict(),
        epoch=completed, best_epoch=completed, best_metric=0.0,
        best_metric_name=DEPLOYMENT_METRIC_NAME, history=history,
        training_config={**config.to_dict(), **dict(seed_report or {})},
        extra=deployment_metadata(plan, epochs_completed=completed),
    )

    total = time.perf_counter() - started
    summary = {
        "schema_version": "task009c_training_summary_v1",
        "training_role": DEPLOYMENT_TRAINING_ROLE,
        "epochs_planned": total_epochs,
        "training_samples": samples,
        "epochs_completed": completed,
        "epoch_policy": EPOCH_POLICY,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_train_accuracy": history[-1]["train_accuracy"] if history else None,
        "in_sample_warning": (
            "final_train_* are IN-SAMPLE values on data the model was fitted to. They are "
            "not a performance estimate. The generalization evidence is the TASK-009B LOSO "
            "result recorded in loso_reference."
        ),
        "history": history,
        "wall_seconds": total,
        "seconds_per_epoch": total / max(1, completed - start_epoch + 1),
        "peak_gpu_mib": (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                         if device.type == "cuda" else None),
        "parameter_count": model.parameter_count(),
        "deployment_checkpoint": str(deployment_path),
        "resume_checkpoint": str(last_path),
    }
    (directory / "history.json").write_text(json.dumps(history, indent=2) + "\n",
                                            encoding="utf-8")
    return summary


__all__ = [
    "DEPLOYMENT_METRIC_NAME", "DEPLOYMENT_CHECKPOINT", "RESUME_CHECKPOINT",
    "deployment_spec", "deployment_metadata", "train_deployment_model",
]
