"""A small, demo-only bridge from queued TASK-008 sequences to TASK-009B.

The adapter deliberately owns no model or tensorization mathematics.  It loads
the exact stored ``virtual_glove.npz`` named by a queue item's descriptor and
delegates both feature construction and prediction to the frozen TASK-009A/B
APIs.  A checkpoint is an explicit caller choice and its recorded experiment
contract is checked before the model is made available to the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from recognition.data import CONTRACT_VERSION, NUM_CLASSES, load_sequence_arrays
from recognition.data.labels import Core28Label, load_label_table
from recognition.models.inference import Prediction, SequenceRecognizer
from recognition.models.lstm_baseline import model_input_dimension
from recognition.training.checkpoint import CheckpointError, ExperimentSpec, load_checkpoint


EXPECTED_FEATURE_SET = "full"
EXPECTED_QUATERNION_POLICY = "absolute"
EXPECTED_POOLING = "masked_mean"
EXPECTED_INPUT_POLICY = "values_and_feature_valid"


class CheckpointCompatibilityError(ValueError):
    """Raised when a selected checkpoint cannot serve the visualizer contract."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Safe-to-display metadata copied from a validated TASK-009B checkpoint."""

    path: str
    schema_version: str
    contract_version: str
    feature_set: str
    quaternion_policy: str
    pooling: str
    input_policy: str
    fold: str
    seed: int
    best_epoch: int | None
    best_metric: float | None
    best_metric_name: str | None
    input_dim: int
    num_classes: int
    device: str
    demo_only: bool = True

    @property
    def held_out_signer(self) -> str | None:
        if self.fold.isdigit() and int(self.fold) in (1, 2, 3):
            return f"S{int(self.fold):02d}"
        return None

    @property
    def display_name(self) -> str:
        fold = f"fold{self.fold}" if self.fold else "fold?"
        return f"{self.feature_set} / {self.quaternion_policy} / {self.pooling} / {fold} / seed{self.seed}"

    @property
    def warning(self) -> str:
        held_out = f" — held-out signer {self.held_out_signer}" if self.held_out_signer else ""
        return f"DEMO / RESEARCH CHECKPOINT ({self.display_name}){held_out}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "feature_set": self.feature_set,
            "quaternion_policy": self.quaternion_policy,
            "pooling": self.pooling,
            "input_policy": self.input_policy,
            "fold": self.fold,
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "best_metric_name": self.best_metric_name,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "device": self.device,
            "demo_only": self.demo_only,
            "held_out_signer": self.held_out_signer,
            "warning": self.warning,
        }


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """One sequence-level prediction attached to one queue sign item.

    ``available=False`` is an explicit engineering failure state.  It never
    substitutes the expected queue label as a prediction.
    """

    sample_id: str
    expected_label_index: int | None
    expected_character: str | None
    expected_sign_id: str | None
    predicted_label_index: int | None
    predicted_character: str | None
    predicted_sign_id: str | None
    confidence: float | None
    probabilities: tuple[float, ...]
    top_k: tuple[dict[str, Any], ...]
    sequence_length: int | None
    checkpoint_metadata: CheckpointMetadata | None
    available: bool
    error: str | None = None

    @classmethod
    def unavailable(
        cls,
        *,
        sample_id: str,
        expected_label_index: int | None,
        expected_character: str | None,
        expected_sign_id: str | None,
        checkpoint_metadata: CheckpointMetadata | None,
        error: str,
        sequence_length: int | None = None,
    ) -> "RecognitionResult":
        return cls(
            sample_id=sample_id,
            expected_label_index=expected_label_index,
            expected_character=expected_character,
            expected_sign_id=expected_sign_id,
            predicted_label_index=None,
            predicted_character=None,
            predicted_sign_id=None,
            confidence=None,
            probabilities=(),
            top_k=(),
            sequence_length=sequence_length,
            checkpoint_metadata=checkpoint_metadata,
            available=False,
            error=str(error),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "expected_label_index": self.expected_label_index,
            "expected_character": self.expected_character,
            "expected_sign_id": self.expected_sign_id,
            "predicted_label_index": self.predicted_label_index,
            "predicted_character": self.predicted_character,
            "predicted_sign_id": self.predicted_sign_id,
            "confidence": self.confidence,
            "probabilities": list(self.probabilities),
            "top_k": [dict(entry) for entry in self.top_k],
            "sequence_length": self.sequence_length,
            "checkpoint_metadata": (
                self.checkpoint_metadata.to_dict() if self.checkpoint_metadata is not None else None
            ),
            "available": self.available,
            "error": self.error,
        }


def _as_optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _as_optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


class RecognizerAdapter:
    """Validated TASK-009B inference for exact stored queue sequences."""

    def __init__(
        self,
        recognizer: SequenceRecognizer,
        metadata: CheckpointMetadata,
        labels: Mapping[int, Core28Label],
        *,
        run_root: str | Path,
    ) -> None:
        self.recognizer = recognizer
        self.metadata = metadata
        self.labels = dict(labels)
        self.run_root = Path(run_root)
        self._cache: dict[str, RecognitionResult] = {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        run_root: str | Path,
        labels_path: str | Path,
        device: str = "cpu",
    ) -> "RecognizerAdapter":
        """Load one explicit compatible checkpoint without changing it."""

        path = Path(checkpoint).expanduser().resolve()
        expected = ExperimentSpec(
            feature_set=EXPECTED_FEATURE_SET,
            quaternion_policy=EXPECTED_QUATERNION_POLICY,
            pooling=EXPECTED_POOLING,
            fold="00",
            seed=0,
            input_policy=EXPECTED_INPUT_POLICY,
        )
        try:
            payload = load_checkpoint(path, expect=expected, map_location=device)
        except (CheckpointError, OSError, RuntimeError, ValueError) as exc:
            raise CheckpointCompatibilityError(str(exc)) from exc

        experiment = payload.get("experiment")
        model_config = payload.get("model_config")
        if not isinstance(experiment, Mapping) or not isinstance(model_config, Mapping):
            raise CheckpointCompatibilityError("checkpoint lacks experiment/model_config metadata")

        required_experiment = {
            "feature_set": EXPECTED_FEATURE_SET,
            "quaternion_policy": EXPECTED_QUATERNION_POLICY,
            "pooling": EXPECTED_POOLING,
            "input_policy": EXPECTED_INPUT_POLICY,
            "contract_version": CONTRACT_VERSION,
        }
        for field, expected_value in required_experiment.items():
            if experiment.get(field) != expected_value:
                raise CheckpointCompatibilityError(
                    f"checkpoint experiment {field}={experiment.get(field)!r}; "
                    f"expected {expected_value!r}"
                )

        labels = load_label_table(labels_path)
        if len(labels) != NUM_CLASSES or sorted(labels) != list(range(NUM_CLASSES)):
            raise CheckpointCompatibilityError("selected label table is not the 28-class contiguous Core-28 table")

        input_dim = int(payload.get("input_dim", -1))
        expected_dim = model_input_dimension(EXPECTED_FEATURE_SET, EXPECTED_INPUT_POLICY)
        if input_dim != expected_dim:
            raise CheckpointCompatibilityError(
                f"checkpoint input_dim={input_dim}; expected {expected_dim} for the primary TASK-009A path"
            )
        if int(payload.get("num_classes", -1)) != NUM_CLASSES:
            raise CheckpointCompatibilityError("checkpoint does not expose exactly 28 output classes")
        for field, expected_value in (
            ("feature_set", EXPECTED_FEATURE_SET),
            ("input_policy", EXPECTED_INPUT_POLICY),
            ("pooling", EXPECTED_POOLING),
            ("num_classes", NUM_CLASSES),
            ("contract_version", CONTRACT_VERSION),
        ):
            if model_config.get(field) != expected_value:
                raise CheckpointCompatibilityError(
                    f"checkpoint model_config {field}={model_config.get(field)!r}; expected {expected_value!r}"
                )
        if not isinstance(payload.get("model_state"), Mapping):
            raise CheckpointCompatibilityError("checkpoint has no model state")

        try:
            recognizer = SequenceRecognizer.from_checkpoint(
                path,
                device=device,
                label_table=labels_path,
            )
        except Exception as exc:  # noqa: BLE001 - surface all load failures to visualization-only mode
            raise CheckpointCompatibilityError(
                f"TASK-009B recognizer could not be constructed: {type(exc).__name__}: {exc}"
            ) from exc

        metadata = CheckpointMetadata(
            path=str(path),
            schema_version=str(payload.get("schema_version", "")),
            contract_version=str(payload.get("contract_version", "")),
            feature_set=str(experiment["feature_set"]),
            quaternion_policy=str(experiment["quaternion_policy"]),
            pooling=str(experiment["pooling"]),
            input_policy=str(experiment["input_policy"]),
            fold=str(experiment.get("fold", "")),
            seed=int(experiment.get("seed", 0)),
            best_epoch=_as_optional_int(payload.get("best_epoch")),
            best_metric=_as_optional_float(payload.get("best_metric")),
            best_metric_name=(
                str(payload["best_metric_name"]) if payload.get("best_metric_name") is not None else None
            ),
            input_dim=input_dim,
            num_classes=int(payload["num_classes"]),
            device=str(device),
        )
        return cls(recognizer, metadata, labels, run_root=run_root)

    def clear_cache(self) -> None:
        self._cache.clear()

    def _label_for_item(self, item: Any) -> Core28Label | None:
        if item.label_index is None:
            return None
        label = self.labels.get(int(item.label_index))
        if label is None:
            raise ValueError(f"queue label_index {item.label_index} is absent from the selected Core-28 table")
        if item.character is not None and label.label_ar != item.character:
            raise ValueError(
                f"queue character {item.character!r} disagrees with label table {label.label_ar!r} "
                f"at index {item.label_index}"
            )
        if item.sign_id is not None and str(label.sign_id).zfill(4) != str(item.sign_id).zfill(4):
            raise ValueError(
                f"queue SignID {item.sign_id!r} disagrees with label table {label.sign_id!r}"
            )
        return label

    def predict_queue_item(self, item: Any) -> RecognitionResult | None:
        """Predict one sign from its stored virtual-glove artifact.

        Gaps return ``None`` and are never passed to TASK-009B.  All sequence
        failures become an explicit unavailable result so the queue can keep
        playing the remaining items.
        """

        if item.item_type == "gap":
            return None
        if item.item_type != "sign":
            raise ValueError(f"unsupported queue item type: {item.item_type!r}")
        sample_id = str(item.sample_id or "")
        if not sample_id:
            raise ValueError("sign queue item has no sample_id")
        if sample_id in self._cache:
            return self._cache[sample_id]

        try:
            self._label_for_item(item)
            descriptor = item.sequence_descriptor
            if descriptor is None or descriptor.sample_id != sample_id:
                raise ValueError("sign queue item has no matching sequence descriptor")
            path = self.run_root / descriptor.virtual_glove_relative_path
            arrays = load_sequence_arrays(path)
            length = int(arrays["frame_index"].shape[0])
            prediction: Prediction = self.recognizer.predict_sequence(arrays)
            if prediction.sequence_length != length:
                raise ValueError(
                    f"TASK-009B returned sequence length {prediction.sequence_length}, expected {length}"
                )
            if len(prediction.probabilities) != NUM_CLASSES:
                raise ValueError(
                    f"TASK-009B returned {len(prediction.probabilities)} probabilities, expected {NUM_CLASSES}"
                )
            top: list[dict[str, Any]] = []
            for candidate in prediction.top_k(3):
                index = int(candidate["label_index"])
                label = self.labels[index]
                top.append({
                    "label_index": index,
                    "character": label.label_ar,
                    "sign_id": label.sign_id,
                    "probability": float(candidate["probability"]),
                })
            result = RecognitionResult(
                sample_id=sample_id,
                expected_label_index=item.label_index,
                expected_character=item.character,
                expected_sign_id=item.sign_id,
                predicted_label_index=prediction.label_index,
                predicted_character=prediction.label_ar,
                predicted_sign_id=prediction.sign_id,
                confidence=float(prediction.confidence),
                probabilities=tuple(float(value) for value in prediction.probabilities),
                top_k=tuple(top),
                sequence_length=length,
                checkpoint_metadata=self.metadata,
                available=True,
            )
        except Exception as exc:  # noqa: BLE001 - one bad demo item must not stop the queue
            result = RecognitionResult.unavailable(
                sample_id=sample_id,
                expected_label_index=item.label_index,
                expected_character=item.character,
                expected_sign_id=item.sign_id,
                checkpoint_metadata=self.metadata,
                error=f"{type(exc).__name__}: {exc}",
            )
        self._cache[sample_id] = result
        return result


__all__ = [
    "EXPECTED_FEATURE_SET",
    "EXPECTED_INPUT_POLICY",
    "EXPECTED_POOLING",
    "EXPECTED_QUATERNION_POLICY",
    "CheckpointCompatibilityError",
    "CheckpointMetadata",
    "RecognitionResult",
    "RecognizerAdapter",
]
