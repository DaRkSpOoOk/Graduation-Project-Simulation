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
    # TASK-009C stores deployment provenance in the checkpoint's ``extra``
    # mapping. These fields are presentation metadata only; they do not
    # affect tensorization, model construction, or prediction.
    training_role: str = "unknown"
    training_scope: str | None = None
    training_samples: int | None = None
    signers: tuple[str, ...] = ()
    held_out_data: str | None = None
    scientific_reference_accuracy: float | None = None
    scientific_reference_macro_f1: float | None = None
    scientific_reference_note: str | None = None

    @property
    def held_out_signer(self) -> str | None:
        if self.fold.isdigit() and int(self.fold) in (1, 2, 3):
            return f"S{int(self.fold):02d}"
        return None

    @property
    def provenance_kind(self) -> str:
        """Classify provenance from recorded metadata, never from a filename."""

        if self.training_role == "deployment_all_signers":
            return "deployment"
        if self.fold in {"01", "02", "03"}:
            return "research_loso"
        return "unknown"

    @property
    def is_deployment(self) -> bool:
        return self.provenance_kind == "deployment"

    @property
    def is_research_loso(self) -> bool:
        return self.provenance_kind == "research_loso"

    @property
    def role_display(self) -> str:
        if self.is_deployment:
            return "DEPLOYMENT MODEL"
        if self.is_research_loso:
            return "DEMO / RESEARCH CHECKPOINT"
        return "CHECKPOINT PROVENANCE UNKNOWN"

    @property
    def training_scope_display(self) -> str:
        if self.is_deployment:
            sample_text = (
                f"{self.training_samples:,} training sequences"
                if self.training_samples is not None
                else "all available training sequences"
            )
            return f"All Core-28 signers / {sample_text}"
        if self.is_research_loso and self.held_out_signer is not None:
            return f"LOSO fold {self.fold} / held-out signer {self.held_out_signer}"
        return self.training_scope or "Training scope unavailable"

    @property
    def scientific_reference_display(self) -> str:
        if (
            self.scientific_reference_accuracy is not None
            and self.scientific_reference_macro_f1 is not None
        ):
            return (
                "LOSO reference only (not deployment accuracy): "
                f"{self.scientific_reference_accuracy:.2%} accuracy / "
                f"{self.scientific_reference_macro_f1:.4f} macro F1"
            )
        return self.scientific_reference_note or "Not embedded in checkpoint"

    @property
    def display_name(self) -> str:
        fold = f"fold{self.fold}" if self.fold else "fold?"
        return f"{self.feature_set} / {self.quaternion_policy} / {self.pooling} / {fold} / seed{self.seed}"

    @property
    def warning(self) -> str:
        if self.is_deployment:
            return f"{self.role_display} ({self.display_name}) — {self.training_scope_display}"
        if not self.is_research_loso:
            return f"{self.role_display} ({self.display_name})"
        held_out = f" — held-out signer {self.held_out_signer}" if self.held_out_signer else ""
        return f"{self.role_display} ({self.display_name}){held_out}"

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
            "training_role": self.training_role,
            "training_scope": self.training_scope,
            "training_samples": self.training_samples,
            "signers": list(self.signers),
            "provenance_kind": self.provenance_kind,
            "held_out_data": self.held_out_data,
            "scientific_reference_accuracy": self.scientific_reference_accuracy,
            "scientific_reference_macro_f1": self.scientific_reference_macro_f1,
            "scientific_reference_note": self.scientific_reference_note,
            "held_out_signer": self.held_out_signer,
            "role_display": self.role_display,
            "training_scope_display": self.training_scope_display,
            "scientific_reference_display": self.scientific_reference_display,
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


def _provenance_metadata(
    payload: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize TASK-009C provenance metadata.

    Deployment provenance is intentionally strict. A checkpoint with
    ``fold=all`` but without the explicit deployment role is classified as
    unknown rather than being guessed to be a deployment model. This keeps an
    accidentally copied or renamed research checkpoint from receiving the
    all-signers label in the UI.
    """

    extra = payload.get("extra", {})
    if extra is None:
        extra = {}
    if not isinstance(extra, Mapping):
        raise CheckpointCompatibilityError("checkpoint extra metadata must be a mapping")

    role_value = extra.get("training_role", "unknown")
    if role_value is None:
        role_value = "unknown"
    if not isinstance(role_value, str) or not role_value.strip():
        raise CheckpointCompatibilityError("checkpoint training_role must be a non-empty string")
    training_role = role_value.strip()

    training_scope = extra.get("training_scope")
    if training_scope is not None and not isinstance(training_scope, str):
        raise CheckpointCompatibilityError("checkpoint training_scope must be a string when present")

    training_samples = extra.get("training_samples")
    if training_samples is not None:
        if isinstance(training_samples, bool) or not isinstance(training_samples, int):
            raise CheckpointCompatibilityError("checkpoint training_samples must be an integer when present")
        training_samples = int(training_samples)

    signers_value = extra.get("signers", ())
    if signers_value is None:
        signers_value = ()
    if not isinstance(signers_value, (list, tuple)) or not all(
        isinstance(value, str) and value for value in signers_value
    ):
        raise CheckpointCompatibilityError("checkpoint signers must be a sequence of non-empty strings")
    signers = tuple(sorted(signers_value))
    if len(set(signers)) != len(signers):
        raise CheckpointCompatibilityError("checkpoint signers contain duplicates")

    if training_role == "deployment_all_signers":
        if str(experiment.get("fold", "")) != "all":
            raise CheckpointCompatibilityError(
                "deployment checkpoint must record experiment fold='all'"
            )
        if training_scope != "all_core28_sequences":
            raise CheckpointCompatibilityError(
                "deployment checkpoint must record training_scope='all_core28_sequences'"
            )
        if training_samples != 4222:
            raise CheckpointCompatibilityError(
                "deployment checkpoint must record training_samples=4222"
            )
        if signers != ("01", "02", "03"):
            raise CheckpointCompatibilityError(
                "deployment checkpoint must record signers ['01', '02', '03']"
            )
        if extra.get("classes") != NUM_CLASSES:
            raise CheckpointCompatibilityError(
                f"deployment checkpoint must record classes={NUM_CLASSES}"
            )

    held_out_data = extra.get("held_out_data")
    if held_out_data is not None and not isinstance(held_out_data, str):
        raise CheckpointCompatibilityError("checkpoint held_out_data must be a string when present")

    scientific_reference = extra.get("loso_reference")
    if scientific_reference is not None and not isinstance(scientific_reference, Mapping):
        raise CheckpointCompatibilityError("checkpoint loso_reference must be a mapping when present")
    scientific_reference = scientific_reference or {}
    accuracy = _as_optional_float(scientific_reference.get("mean_test_accuracy"))
    macro_f1 = _as_optional_float(scientific_reference.get("mean_test_macro_f1"))
    note = scientific_reference.get("note")
    if note is not None and not isinstance(note, str):
        raise CheckpointCompatibilityError("checkpoint LOSO reference note must be a string when present")

    return {
        "training_role": training_role,
        "training_scope": training_scope,
        "training_samples": training_samples,
        "signers": signers,
        "held_out_data": held_out_data,
        "scientific_reference_accuracy": accuracy,
        "scientific_reference_macro_f1": macro_f1,
        "scientific_reference_note": note,
    }


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
        provenance = _provenance_metadata(payload, experiment)

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
            **provenance,
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
