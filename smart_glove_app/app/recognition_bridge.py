"""Asynchronous application boundary around the frozen recognition adapter."""

from __future__ import annotations

from typing import Any


class RecognitionBridge:
    """Keep recognizer inputs separate from presentation/interpolated frames.

    ``predict`` accepts only a queue item.  The bridge never accepts a
    ``PresentationFrame`` or any render buffer, which makes it impossible for
    display-time interpolation to enter TASK-009A/B arrays through this path.
    Imports of the PyTorch-backed visualizer adapter remain lazy so the
    no-checkpoint application stays lightweight.
    """

    def __init__(self, adapter: Any | None = None, *, disabled_reason: str | None = None) -> None:
        self.adapter = adapter
        self.disabled_reason = disabled_reason
        self._controller: Any | None = None
        if adapter is not None:
            self._make_controller()

    def _make_controller(self) -> None:
        if self.adapter is None:
            self._controller = None
            return
        from visualizer.recognition.controller import RecognitionController

        self._controller = RecognitionController(self.adapter)

    @property
    def enabled(self) -> bool:
        return self.adapter is not None and self._controller is not None

    @property
    def metadata(self) -> Any | None:
        return self.adapter.metadata if self.adapter is not None else None

    def set_adapter(self, adapter: Any, *, disabled_reason: str | None = None) -> None:
        self.adapter = adapter
        self.disabled_reason = disabled_reason
        self._make_controller()

    def disable(self, reason: str | None = None) -> None:
        self.adapter = None
        self._controller = None
        if reason is not None:
            self.disabled_reason = str(reason)

    def clear(self) -> None:
        if self._controller is not None:
            self._controller.clear()

    def predict(self, queue_item: Any) -> Any | None:
        """Predict one exact queued sign; gaps are never sent to recognition."""

        if self._controller is None:
            return None
        return self._controller.result_for(queue_item)


__all__ = ["RecognitionBridge"]
