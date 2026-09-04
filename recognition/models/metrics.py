"""Classification metrics for TASK-009B.

The metric definitions come from scikit-learn -- accuracy, precision, recall,
macro/weighted F1, confusion matrix and classification report are standard
problems with a battle-tested implementation, and there is no reason to write
them again. This module is a thin wrapper whose only jobs are project-specific:

* pin the label space to the frozen 28 Core-28 classes, so a class the model
  never predicts still appears (with zero support) instead of silently vanishing
  from a macro average;
* keep the experiment result schema stable regardless of scikit-learn's own
  return shapes, so result JSON stays comparable across runs;
* resolve confusion pairs to authoritative Arabic labels.

Everything here is JSON-serializable, because it lands in ``result.json``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix as sklearn_confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

from ..data.contract import NUM_CLASSES

ALL_LABELS = list(range(NUM_CLASSES))


def confusion_matrix(
    labels: Sequence[int], predictions: Sequence[int], num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Rows are true classes, columns predicted, always ``num_classes`` square.

    Wraps ``sklearn.metrics.confusion_matrix`` with an explicit ``labels`` list;
    without it scikit-learn sizes the matrix to the classes that happen to occur,
    which would make matrices from different folds non-comparable.
    """

    true = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if true.shape != predicted.shape:
        raise ValueError("labels and predictions must have the same length")
    for name, array in (("label", true), ("prediction", predicted)):
        if array.size and (array.min() < 0 or array.max() >= num_classes):
            raise ValueError(f"{name} outside [0, {num_classes})")
    return sklearn_confusion_matrix(true, predicted, labels=list(range(num_classes)))


def classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    *,
    probabilities: Sequence[Sequence[float]] | None = None,
    num_classes: int = NUM_CLASSES,
) -> dict[str, Any]:
    """Accuracy plus macro/weighted precision, recall and F1, over all classes."""

    true = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    matrix = confusion_matrix(true, predicted, num_classes)
    all_labels = list(range(num_classes))

    per_class_precision, per_class_recall, per_class_f1, support = (
        precision_recall_fscore_support(
            true, predicted, labels=all_labels, average=None, zero_division=0
        )
    )
    # Macro averages are taken over classes present in the reference labels, so a
    # class the model never predicts is penalised rather than excused, and an
    # absent class does not drag the mean toward zero.
    present = [label for label in all_labels if (true == label).any()]
    macro = precision_recall_fscore_support(
        true, predicted, labels=present, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        true, predicted, labels=all_labels, average="weighted", zero_division=0
    )

    result: dict[str, Any] = {
        "samples": int(true.size),
        "accuracy": float(accuracy_score(true, predicted)) if true.size else 0.0,
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "classes_present": len(present),
        "per_class": {
            "precision": [float(v) for v in per_class_precision],
            "recall": [float(v) for v in per_class_recall],
            "f1": [float(v) for v in per_class_f1],
            "support": [int(v) for v in support],
        },
        "confusion_matrix": matrix.tolist(),
    }
    if probabilities is not None and true.size:
        result["cross_entropy"] = float(
            log_loss(true, np.asarray(probabilities, dtype=np.float64), labels=all_labels)
        )
    return result


def classification_text_report(
    labels: Sequence[int],
    predictions: Sequence[int],
    *,
    labels_ar: Mapping[int, str] | None = None,
    num_classes: int = NUM_CLASSES,
) -> str:
    """scikit-learn's per-class report, with Arabic class names when available."""

    all_labels = list(range(num_classes))
    names = [f"{index:02d} {labels_ar.get(index, '')}".strip() if labels_ar else str(index)
             for index in all_labels]
    return classification_report(
        np.asarray(labels, dtype=np.int64), np.asarray(predictions, dtype=np.int64),
        labels=all_labels, target_names=names, zero_division=0, digits=4,
    )


def top_confusions(
    matrix: Sequence[Sequence[int]] | np.ndarray,
    *,
    limit: int = 10,
    labels_ar: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    """The most frequent off-diagonal (true, predicted) pairs.

    Project-specific: scikit-learn has no ranked-confusion-pair helper, and the
    Arabic label resolution is ours.
    """

    array = np.asarray(matrix, dtype=np.int64)
    off_diagonal = array.copy()
    np.fill_diagonal(off_diagonal, 0)
    order = np.argsort(off_diagonal, axis=None)[::-1][:limit]
    out: list[dict[str, Any]] = []
    for flat in order:
        true_index, predicted_index = np.unravel_index(int(flat), array.shape)
        count = int(off_diagonal[true_index, predicted_index])
        if count <= 0:
            break
        entry: dict[str, Any] = {
            "true_label_index": int(true_index),
            "predicted_label_index": int(predicted_index),
            "count": count,
        }
        if labels_ar:
            entry["true_label_ar"] = labels_ar.get(int(true_index), "")
            entry["predicted_label_ar"] = labels_ar.get(int(predicted_index), "")
        out.append(entry)
    return out


__all__ = [
    "ALL_LABELS", "confusion_matrix", "classification_metrics",
    "classification_text_report", "top_confusions",
]
