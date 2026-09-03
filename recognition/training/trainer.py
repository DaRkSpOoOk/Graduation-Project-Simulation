"""TASK-009B training loop: conservative, resumable, and honest about selection.

Checkpoint selection uses **validation macro F1 only**. The held-out test signer
never influences early stopping, checkpoint choice, normalization or any
architecture decision, and the test split is not even loaded until an explicit
final evaluation.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ..data.contract import NUM_CLASSES
from ..models.lstm_baseline import LSTMBaseline
from ..models.metrics import classification_metrics, top_confusions
from .checkpoint import ExperimentSpec, load_checkpoint, save_checkpoint
from .determinism import random_state, restore_random_state


@dataclass(frozen=True)
class TrainingConfig:
    """Conservative defaults, frozen before any test evaluation is run."""

    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    early_stopping_patience: int = 12
    optimizer: str = "adamw"
    # Class counts range 149-160 (ratio 1.07), so ordinary cross-entropy is used.
    # A weighted loss would be an unjustified extra degree of freedom here.
    loss: str = "cross_entropy"
    selection_metric: str = "validation_macro_f1"
    num_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                weight_decay=config.weight_decay)
    raise ValueError(f"unknown optimizer {config.optimizer!r}")


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model: LSTMBaseline,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    collect_predictions: bool = False,
    labels_ar: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Deterministic evaluation: eval mode, no grad, no shuffling assumed."""

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    labels: list[int] = []
    predictions: list[int] = []
    records: list[dict[str, Any]] = []
    loss_sum = 0.0
    seen = 0
    for batch in loader:
        moved = move_batch(batch, device)
        logits = model(moved)
        target = moved["labels"]
        loss_sum += float(criterion(logits, target))
        seen += int(target.shape[0])
        predicted = logits.argmax(dim=-1)
        labels.extend(target.detach().cpu().tolist())
        predictions.extend(predicted.detach().cpu().tolist())
        if collect_predictions:
            probabilities = torch.softmax(logits, dim=-1).detach().cpu()
            for row in range(int(target.shape[0])):
                index = int(predicted[row])
                records.append({
                    "sample_id": batch["sample_ids"][row],
                    "signer_id": batch["signer_ids"][row],
                    "true_label_index": int(target[row]),
                    "predicted_label_index": index,
                    "confidence": float(probabilities[row, index]),
                    "length": int(batch["lengths"][row]),
                    "correct": int(target[row]) == index,
                })
    metrics = classification_metrics(labels, predictions, num_classes=NUM_CLASSES)
    metrics["cross_entropy"] = loss_sum / seen if seen else 0.0
    metrics["top_confusions"] = top_confusions(metrics["confusion_matrix"], labels_ar=labels_ar)
    if collect_predictions:
        metrics["predictions"] = records
    return metrics


def train_one_epoch(
    model: LSTMBaseline,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    seen = 0
    correct = 0
    for batch in loader:
        moved = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(moved)
        loss = criterion(logits, moved["labels"])
        if not torch.isfinite(loss):
            raise RuntimeError("training loss became non-finite")
        loss.backward()
        if config.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        size = int(moved["labels"].shape[0])
        total_loss += float(loss.detach()) * size
        seen += size
        correct += int((logits.detach().argmax(dim=-1) == moved["labels"]).sum())
    return {"loss": total_loss / seen if seen else 0.0,
            "accuracy": correct / seen if seen else 0.0}


def _format_eta(seconds: float) -> str:
    if seconds < 0 or not np.isfinite(seconds):
        return "--:--:--"
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _gpu_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def train(
    *,
    model: LSTMBaseline,
    spec: ExperimentSpec,
    train_loader: Iterable[Mapping[str, Any]],
    validation_loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    config: TrainingConfig,
    output_dir: str | Path,
    labels_ar: Mapping[int, str] | None = None,
    resume: bool = False,
    seed_report: Mapping[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train with early stopping on validation macro F1. Test is never consulted."""

    emit = log or (lambda message: print(message, flush=True))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    last_path = directory / "last.pt"
    best_path = directory / "best.pt"

    model.to(device)
    optimizer = build_optimizer(model, config)
    start_epoch = 1
    best_metric = -1.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, Any]] = []

    if resume and last_path.is_file():
        payload = load_checkpoint(last_path, expect=spec, map_location=device)
        model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_metric = float(payload["best_metric"])
        best_epoch = int(payload["best_epoch"])
        patience_counter = int(payload.get("early_stopping_counter", 0))
        history = list(payload.get("history", []))
        restored = restore_random_state(payload.get("rng_state"))
        emit(f"[resume] continuing from epoch {start_epoch} "
             f"(best {config.selection_metric} {best_metric:.4f} at epoch {best_epoch}, "
             f"rng_restored={restored})")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    header = (f"{spec.feature_set} | q={spec.quaternion_policy} | pool={spec.pooling} | "
              f"fold S{spec.fold} | seed {spec.seed}")
    emit(f"[train] {header}")
    emit(f"[train] device={device} params={model.parameter_count():,} "
         f"input_dim={model.config.input_dim} epochs={config.epochs}")

    started = time.perf_counter()
    completed_epochs = 0
    stopped_early = False
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, optimizer, device, config)
        validation = evaluate(model, validation_loader, device, labels_ar=labels_ar)
        completed_epochs = epoch
        elapsed = time.perf_counter() - started
        per_epoch = elapsed / max(1, epoch - start_epoch + 1)
        remaining = per_epoch * (config.epochs - epoch)

        improved = validation["macro_f1"] > best_metric
        if improved:
            best_metric = float(validation["macro_f1"])
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        entry = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "validation_loss": validation["cross_entropy"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "best_validation_macro_f1": best_metric,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(entry)

        memory = _gpu_memory_mib(device)
        emit(
            f"[epoch {epoch:3d}/{config.epochs}] "
            f"train_loss {train_stats['loss']:.4f} | "
            f"val_loss {validation['cross_entropy']:.4f} | "
            f"val_acc {validation['accuracy']*100:6.2f}% | "
            f"val_macroF1 {validation['macro_f1']:.4f} | "
            f"best {best_metric:.4f}@{best_epoch}{' *' if improved else '  '} | "
            f"{entry['epoch_seconds']:.1f}s | elapsed {_format_eta(elapsed)} | "
            f"ETA {_format_eta(remaining)}"
            + (f" | GPU {memory:.0f} MiB" if memory is not None else "")
        )

        common = dict(
            model=model, spec=spec, optimizer_state=optimizer.state_dict(), epoch=epoch,
            best_epoch=best_epoch, best_metric=best_metric,
            best_metric_name=config.selection_metric,
            early_stopping_counter=patience_counter, history=history,
            training_config={**config.to_dict(), **dict(seed_report or {})},
        )
        save_checkpoint(last_path, rng_state=random_state(), **common)
        if improved:
            save_checkpoint(best_path, extra={"validation_metrics": {
                k: v for k, v in validation.items() if k != "predictions"}}, **common)

        if patience_counter >= config.early_stopping_patience:
            emit(f"[train] early stopping: no improvement for "
                 f"{config.early_stopping_patience} epochs")
            stopped_early = True
            break

    total = time.perf_counter() - started
    result = {
        "experiment": spec.to_dict(),
        "training_config": config.to_dict(),
        "seeding": dict(seed_report or {}),
        "epochs_completed": completed_epochs,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_metric_name": config.selection_metric,
        "best_metric": best_metric,
        "history": history,
        "wall_seconds": total,
        "seconds_per_epoch": total / max(1, completed_epochs - start_epoch + 1),
        "peak_gpu_mib": _gpu_memory_mib(device),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "parameter_count": model.parameter_count(),
    }
    (directory / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = ["TrainingConfig", "build_optimizer", "move_batch", "evaluate",
           "train_one_epoch", "train"]
