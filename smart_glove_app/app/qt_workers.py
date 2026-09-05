"""Small QRunnable jobs used to keep disk/model work off the GUI thread."""

from __future__ import annotations

from typing import Any


try:
    from PySide6.QtCore import QObject, QRunnable, Signal

    QT_WORKERS_AVAILABLE = True
except ImportError:  # pragma: no cover - minimal CI fallback
    QObject = object  # type: ignore[assignment,misc]
    QRunnable = object  # type: ignore[assignment,misc]
    Signal = None  # type: ignore[assignment]
    QT_WORKERS_AVAILABLE = False


if QT_WORKERS_AVAILABLE:

    class WorkerSignals(QObject):
        loaded = Signal(int, object)
        failed = Signal(int, str)


    class SequenceLoadTask(QRunnable):
        """Read one TASK-008 sequence through the existing loader."""

        def __init__(
            self,
            token: int,
            item: Any,
            *,
            run_root: Any,
            manifest_path: Any | None,
        ) -> None:
            super().__init__()
            self.token = int(token)
            self.item = item
            self.run_root = run_root
            self.manifest_path = manifest_path
            self.signals = WorkerSignals()

        def run(self) -> None:
            try:
                from visualizer.app.integration import load_sequence_for_item

                sequence = load_sequence_for_item(
                    self.item,
                    run_root=self.run_root,
                    manifest_path=self.manifest_path,
                )
                self.signals.loaded.emit(self.token, sequence)
            except Exception as exc:  # noqa: BLE001 - converted to UI state
                self.signals.failed.emit(self.token, f"{type(exc).__name__}: {exc}")


    class CheckpointLoadTask(QRunnable):
        """Load and validate one explicit TASK-009C/TASK-009B checkpoint."""

        def __init__(
            self,
            token: int,
            checkpoint: Any,
            *,
            run_root: Any,
            labels_path: Any,
            device: str,
        ) -> None:
            super().__init__()
            self.token = int(token)
            self.checkpoint = checkpoint
            self.run_root = run_root
            self.labels_path = labels_path
            self.device = device
            self.signals = WorkerSignals()

        @staticmethod
        def _device(value: str) -> str:
            if value != "auto":
                return value
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"

        def run(self) -> None:
            try:
                from visualizer.recognition import RecognizerAdapter

                adapter = RecognizerAdapter.from_checkpoint(
                    self.checkpoint,
                    run_root=self.run_root,
                    labels_path=self.labels_path,
                    device=self._device(self.device),
                )
                self.signals.loaded.emit(self.token, adapter)
            except Exception as exc:  # noqa: BLE001 - checkpoint failure is a UI state
                self.signals.failed.emit(self.token, f"{type(exc).__name__}: {exc}")


    class RecognitionTask(QRunnable):
        """Run one sequence-level prediction without touching render buffers."""

        def __init__(self, token: int, queue_item: Any, bridge: Any) -> None:
            super().__init__()
            self.token = int(token)
            self.queue_item = queue_item
            self.bridge = bridge
            self.signals = WorkerSignals()

        def run(self) -> None:
            try:
                result = self.bridge.predict(self.queue_item)
                self.signals.loaded.emit(self.token, result)
            except Exception as exc:  # noqa: BLE001 - converted to explicit UI state
                self.signals.failed.emit(self.token, f"{type(exc).__name__}: {exc}")


else:

    class WorkerSignals:  # type: ignore[no-redef]
        pass


    class SequenceLoadTask:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PySide6 is required for background application workers")


    class CheckpointLoadTask:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PySide6 is required for background application workers")


    class RecognitionTask:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PySide6 is required for background application workers")


__all__ = [
    "QT_WORKERS_AVAILABLE",
    "CheckpointLoadTask",
    "RecognitionTask",
    "SequenceLoadTask",
]
