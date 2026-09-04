"""Duration Control A: how much of the label is predictable from length alone?

The classifier is ``sklearn.linear_model.LogisticRegression`` on a single
standardized scalar feature -- the sequence length -- inside a scikit-learn
``Pipeline`` so the scaler is fitted on train and merely applied to
validation/test. Using the library rather than a hand-rolled multinomial fit
keeps the control uncontroversial: a reviewer can check the reference number
without auditing our optimizer.

The control is fitted on TRAIN only. Test labels never touch it, in any role.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..data.contract import NUM_CLASSES
from ..models.metrics import classification_metrics


@dataclass
class LengthOnlyClassifier:
    """Multinomial logistic regression on the single feature ``sequence_length``."""

    num_classes: int = NUM_CLASSES
    max_iter: int = 1000
    C: float = 1.0
    seed: int = 0
    pipeline: Pipeline | None = field(default=None, repr=False)

    @staticmethod
    def _design(lengths: Sequence[int]) -> np.ndarray:
        return np.asarray(lengths, dtype=np.float64).reshape(-1, 1)

    def fit(self, lengths: Sequence[int], labels: Sequence[int]) -> "LengthOnlyClassifier":
        design = self._design(lengths)
        if design.size == 0:
            raise ValueError("cannot fit a length-only classifier on no samples")
        # The scaler lives inside the pipeline, so its mean/scale are fitted here
        # on TRAIN and only applied afterwards -- the same leakage rule the main
        # training pipeline follows.
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(
                max_iter=self.max_iter, C=self.C, random_state=self.seed)),
        ]).fit(design, np.asarray(labels, dtype=np.int64))
        return self

    def _require_fit(self) -> Pipeline:
        if self.pipeline is None:
            raise RuntimeError("classifier has not been fitted")
        return self.pipeline

    def predict(self, lengths: Sequence[int]) -> np.ndarray:
        return self._require_fit().predict(self._design(lengths))

    def predict_proba(self, lengths: Sequence[int]) -> np.ndarray:
        """Class probabilities over the full 28-class space.

        ``LogisticRegression`` only emits columns for the classes it saw during
        fitting, so the result is expanded back to the frozen label space before
        it is handed to anything that assumes 28 columns.
        """

        pipeline = self._require_fit()
        partial = pipeline.predict_proba(self._design(lengths))
        full = np.zeros((partial.shape[0], self.num_classes), dtype=np.float64)
        full[:, pipeline.named_steps["logistic"].classes_.astype(int)] = partial
        return full

    def evaluate(self, lengths: Sequence[int], labels: Sequence[int]) -> dict[str, Any]:
        predictions = self.predict(lengths)
        metrics = classification_metrics(
            labels, predictions.tolist(),
            probabilities=self.predict_proba(lengths), num_classes=self.num_classes,
        )
        return {k: v for k, v in metrics.items() if k != "per_class"}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": "sklearn.linear_model.LogisticRegression on sequence_length",
            "pipeline": "StandardScaler -> LogisticRegression (multinomial)",
            "feature": "sequence_length (scaler fitted on TRAIN only)",
            "num_classes": self.num_classes,
            "max_iter": self.max_iter,
            "C": self.C,
            "seed": self.seed,
            "fitted": self.pipeline is not None,
        }
        if self.pipeline is not None:
            scaler = self.pipeline.named_steps["scale"]
            logistic = self.pipeline.named_steps["logistic"]
            payload.update({
                "length_mean": float(scaler.mean_[0]),
                "length_scale": float(scaler.scale_[0]),
                "classes_seen": [int(c) for c in logistic.classes_],
                "coefficients": logistic.coef_.reshape(-1).tolist(),
                "intercepts": logistic.intercept_.tolist(),
            })
        return payload


def oracle_accuracy_from_length(lengths: Sequence[int], targets: Sequence[Any]) -> float:
    """In-sample upper bound achievable using length as the only feature.

    Kept as custom code deliberately: this is not a fitted model but a
    combinatorial ceiling -- assign every distinct length its own majority target
    -- and scikit-learn has no equivalent. It is the same statistic TASK-009A
    reported, so the control and the audit stay directly comparable.
    """

    materialized = list(lengths)
    if not materialized:
        return 0.0
    best: dict[int, Counter] = defaultdict(Counter)
    for length, target in zip(materialized, targets):
        best[int(length)][target] += 1
    return sum(c.most_common(1)[0][1] for c in best.values()) / len(materialized)


__all__ = ["LengthOnlyClassifier", "oracle_accuracy_from_length"]
