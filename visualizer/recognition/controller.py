"""Queue-facing recognition state, kept separate from renderer mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapter import CheckpointMetadata, RecognitionResult, RecognizerAdapter


@dataclass(frozen=True, slots=True)
class RecognitionControllerState:
    """Current display state for one queue item."""

    result: RecognitionResult | None
    enabled: bool
    disabled_reason: str | None = None


class RecognitionController:
    """Attach at most one cached sequence-level prediction to each sign."""

    def __init__(
        self,
        adapter: RecognizerAdapter | None = None,
        *,
        disabled_reason: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.disabled_reason = disabled_reason
        self.current = RecognitionControllerState(None, adapter is not None, disabled_reason)
        self._cache: dict[str, RecognitionResult] = {}

    @property
    def enabled(self) -> bool:
        return self.adapter is not None

    @property
    def metadata(self) -> CheckpointMetadata | None:
        return self.adapter.metadata if self.adapter is not None else None

    def clear(self) -> None:
        self.current = RecognitionControllerState(None, self.enabled, self.disabled_reason)
        self._cache.clear()
        if self.adapter is not None:
            self.adapter.clear_cache()

    def result_for(self, item: Any) -> RecognitionResult | None:
        """Return a cached result, or compute one sequence-level prediction."""

        if item.item_type == "gap":
            self.current = RecognitionControllerState(None, self.enabled, self.disabled_reason)
            return None
        if self.adapter is None:
            result = RecognitionResult.unavailable(
                sample_id=str(item.sample_id or ""),
                expected_label_index=item.label_index,
                expected_character=item.character,
                expected_sign_id=item.sign_id,
                checkpoint_metadata=None,
                error=self.disabled_reason or "recognition disabled; no checkpoint selected",
            )
            self.current = RecognitionControllerState(result, False, self.disabled_reason)
            return result
        sample_id = str(item.sample_id or "")
        if sample_id in self._cache:
            result = self._cache[sample_id]
        else:
            try:
                result = self.adapter.predict_queue_item(item)
            except Exception as exc:  # noqa: BLE001 - a model failure is a display state, not a queue failure
                result = RecognitionResult.unavailable(
                    sample_id=sample_id,
                    expected_label_index=item.label_index,
                    expected_character=item.character,
                    expected_sign_id=item.sign_id,
                    checkpoint_metadata=self.metadata,
                    error=f"{type(exc).__name__}: {exc}",
                )
            if result is None:
                self.current = RecognitionControllerState(None, True, None)
                return None
            self._cache[sample_id] = result
        self.current = RecognitionControllerState(result, True, None)
        return result


__all__ = ["RecognitionController", "RecognitionControllerState"]
