"""Standalone inference API for the TASK-009B recognizer.

This exists for the parallel visualizer work. It depends on nothing from the
training machinery: no optimizer, no training loop, no training DataLoader, no
RGB frames. The recognizer consumes TASK-009A sensor tensors only, and returns
the authoritative Arabic label rather than a bare integer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..data.collate import collate_sequences
from ..data.contract import SequenceInputConfig
from ..data.labels import Core28Label, load_label_table
from ..data.sequence_dataset import build_feature_tensor
from .lstm_baseline import LSTMBaseline


@dataclass(frozen=True)
class Prediction:
    """One sequence's classification, with the label resolved authoritatively."""

    label_index: int
    label_ar: str
    sign_id: str
    label_en: str
    confidence: float
    logits: list[float]
    probabilities: list[float]
    sequence_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_index": self.label_index,
            "label_ar": self.label_ar,
            "sign_id": self.sign_id,
            "label_en": self.label_en,
            "confidence": self.confidence,
            "logits": self.logits,
            "probabilities": self.probabilities,
            "sequence_length": self.sequence_length,
        }

    def top_k(self, k: int = 5) -> list[dict[str, Any]]:
        order = sorted(range(len(self.probabilities)),
                       key=lambda i: self.probabilities[i], reverse=True)[:k]
        return [{"label_index": i, "probability": self.probabilities[i]} for i in order]


class SequenceRecognizer:
    """A loaded checkpoint that classifies one TASK-009A sequence at a time."""

    def __init__(
        self,
        model: LSTMBaseline,
        config: SequenceInputConfig,
        *,
        labels: Mapping[int, Core28Label] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.config = config
        self.labels = dict(labels) if labels is not None else None

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        label_table: str | Path | None = None,
    ) -> "SequenceRecognizer":
        """Load a checkpoint and the input configuration it was trained under.

        The tensorization config is taken from the checkpoint, never guessed, so
        a caller cannot accidentally feed ``full`` features to a ``bend_only``
        model.
        """

        from ..training.checkpoint import load_checkpoint, rebuild_model

        payload = load_checkpoint(path, map_location=device)
        model = rebuild_model(payload)
        experiment = payload["experiment"]
        config = SequenceInputConfig(
            feature_set=experiment["feature_set"],
            quaternion_policy=(experiment["quaternion_policy"]
                               if experiment["feature_set"] == "full" else "absolute"),
        )
        labels = None
        if label_table is not None:
            labels = load_label_table(label_table)
        return cls(model, config, labels=labels, device=device)

    def _resolve(self, index: int) -> Core28Label:
        if self.labels is None:
            self.labels = load_label_table()
        label = self.labels.get(index)
        if label is None:
            raise KeyError(f"label_index {index} is not in the Core-28 label table")
        return label

    @torch.no_grad()
    def predict_sequence(self, arrays: Mapping[str, Any]) -> Prediction:
        """Classify one sequence given raw TASK-009A arrays.

        ``arrays`` is what a ``virtual_glove.npz`` holds (or an equivalent live
        capture); tensorization goes through the frozen TASK-009A code path, so
        the visualizer and the training pipeline cannot diverge.
        """

        item = build_feature_tensor(arrays, self.config)
        item.update(sample_id="live", sign_id="", label_ar="", label_index=0,
                    signer_id="", official_partition="")
        return self.predict_item(item)

    @torch.no_grad()
    def predict_item(self, item: Mapping[str, Any]) -> Prediction:
        """Classify one already-tensorized TASK-009A item."""

        batch = collate_sequences([item], self.config)
        return self.predict_batch(batch)[0]

    @torch.no_grad()
    def predict_batch(self, batch: Mapping[str, Any]) -> list[Prediction]:
        """Classify a collated TASK-009A batch."""

        moved = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        logits = self.model(moved)
        probabilities = torch.softmax(logits, dim=-1)
        lengths = batch["lengths"].tolist()
        out: list[Prediction] = []
        for row in range(logits.shape[0]):
            index = int(logits[row].argmax())
            label = self._resolve(index)
            out.append(Prediction(
                label_index=index,
                label_ar=label.label_ar,
                sign_id=label.sign_id,
                label_en=label.label_en,
                confidence=float(probabilities[row, index]),
                logits=logits[row].detach().cpu().tolist(),
                probabilities=probabilities[row].detach().cpu().tolist(),
                sequence_length=int(lengths[row]),
            ))
        return out


__all__ = ["Prediction", "SequenceRecognizer"]
