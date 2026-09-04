"""Qt-facing application state for the persistent Core-28 scene."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from visualizer.contract import validate_sensor_layout
from visualizer.keyboard import Core28Keyboard
from visualizer.mapping import Core28Resolver
from visualizer.queue import PlaybackQueue, QueueState, UnsupportedTextError

from smart_glove_app.rendering.hand_mesh_state import PersistentRenderScene, PresentationFrame
from smart_glove_app.rendering.mano_topology import ManoTopology
from smart_glove_app.rendering.qt_geometry import QtHandGeometry
from smart_glove_app.rendering.sensor_markers import SensorMarkerModel

from .playback_controller import PersistentPlaybackController, PlaybackDisplayState
from .recognition_bridge import RecognitionBridge
from .qt_workers import CheckpointLoadTask, RecognitionTask, SequenceLoadTask


try:
    from PySide6.QtCore import QObject, Property, QThreadPool, QTimer, Signal, Slot

    QT_CONTROLLER_AVAILABLE = True
except ImportError:  # pragma: no cover - the entry point reports this clearly
    QObject = object  # type: ignore[assignment,misc]
    Property = None  # type: ignore[assignment]
    QThreadPool = object  # type: ignore[assignment,misc]
    QTimer = object  # type: ignore[assignment,misc]
    Signal = None  # type: ignore[assignment]
    Slot = lambda *args, **kwargs: (lambda function: function)  # type: ignore[assignment]
    QT_CONTROLLER_AVAILABLE = False


def _default_sensor_layout() -> tuple[Any, ...]:
    """Use the frozen TASK-006 layout for idle marker rows."""

    from virtual_glove.layout import layout_document

    return validate_sensor_layout(layout_document())


DEFAULT_SENSOR_LAYOUT = _default_sensor_layout()
EXEMPLAR_MODES = ("canonical", "signer01", "signer02", "signer03", "random")
SPEEDS = (0.5, 1.0, 2.0)
EM_DASH = "—"


if QT_CONTROLLER_AVAILABLE:

    class Core28ApplicationController(QObject):
        """Own application state while keeping scientific and render state separate."""

        queuedTextChanged = Signal()
        statusMessageChanged = Signal()
        activeLabelChanged = Signal()
        currentSampleIdChanged = Signal()
        frameStateChanged = Signal()
        queueStateChanged = Signal()
        playbackPlayingChanged = Signal()
        smoothRenderingChanged = Signal()
        renderMetricsChanged = Signal()
        recognitionStateChanged = Signal()
        topologyStateChanged = Signal()
        diagnosticsVisibleChanged = Signal()
        modeChanged = Signal()
        resetViewRequested = Signal()

        def __init__(
            self,
            *,
            run_root: str | Path,
            resolver: Core28Resolver,
            manifest_path: str | Path | None,
            topology: ManoTopology | None,
            checkpoint: str | Path | None = None,
            labels_path: str | Path | None = None,
            device: str = "auto",
            initial_text: str = "",
            mode: str = "canonical",
            rng_seed: int | None = None,
            speed: float = 1.0,
            smooth_rendering: bool = True,
        ) -> None:
            super().__init__()
            if mode not in EXEMPLAR_MODES:
                raise ValueError(f"unsupported exemplar mode: {mode!r}")
            if float(speed) not in SPEEDS:
                raise ValueError(f"speed must be one of {SPEEDS}")
            self._run_root = Path(run_root).expanduser().resolve()
            self._manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path else None
            self._labels_path = Path(labels_path).expanduser().resolve() if labels_path else None
            self._resolver = resolver
            self._keyboard = Core28Keyboard(resolver.mapping)
            self._queue = PlaybackQueue(resolver)
            self._device = str(device)
            self._mode = mode
            self._rng_seed = rng_seed
            self._speed = float(speed)
            self._requested_text = ""
            self._active_item: Any | None = None
            self._sequence: Any | None = None
            self._playback: PersistentPlaybackController | None = None
            self._gap_deadline: float | None = None
            self._active_token = 0
            self._closed = False
            self._smooth_rendering = bool(smooth_rendering)
            self._diagnostics_visible = False

            # These are the two persistent GPU geometry providers.  They are
            # intentionally constructed before any queue item is loaded.
            self.scene = PersistentRenderScene(topology)
            self.left_geometry = QtHandGeometry("LEFT", topology)
            self.right_geometry = QtHandGeometry("RIGHT", topology)
            self.left_markers = SensorMarkerModel(DEFAULT_SENSOR_LAYOUT, self)
            self.right_markers = SensorMarkerModel(DEFAULT_SENSOR_LAYOUT, self)
            self._upload_presentation(self.scene.clear_sequence())

            self._recognition = RecognitionBridge()
            self._checkpoint_path = str(Path(checkpoint).expanduser().resolve()) if checkpoint else ""
            self._recognition_status = "Recognition disabled"
            self._recognition_role = "Visualization-only mode"
            self._recognition_scope = EM_DASH
            self._recognition_reference = EM_DASH
            self._recognition_checkpoint = EM_DASH
            self._expected_character = EM_DASH
            self._predicted_character = EM_DASH
            self._confidence_text = EM_DASH
            self._recognition_error = ""

            self._status_message = "Ready. Type Arabic text or select a Core-28 key."
            self._active_label = "No active sign"
            self._current_sample_id = EM_DASH
            self._frame_index = -1
            self._frame_position = -1
            self._frame_count = 0
            self._source_fps = 0.0
            self._render_fps = 0.0
            self._render_frame_count = 0
            self._render_window_start = time.monotonic()
            self._graphics_api = "Qt Quick 3D / RHI"

            self._load_pool = QThreadPool(self)
            self._load_pool.setMaxThreadCount(2)
            self._recognition_pool = QThreadPool(self)
            self._recognition_pool.setMaxThreadCount(1)
            self._timer = QTimer(self)
            self._timer.setInterval(16)
            self._timer.timeout.connect(self._tick)
            self._timer.start()

            if self._checkpoint_path:
                self._recognition_status = "Loading recognition checkpoint…"
                self._start_checkpoint_load()
            if initial_text:
                self.enqueue_text(initial_text)

        # ---- QML properties -------------------------------------------------

        @Property(str, notify=queuedTextChanged)
        def queuedText(self) -> str:  # noqa: N802 - QML property name
            return self._requested_text

        @Property(str, notify=statusMessageChanged)
        def statusMessage(self) -> str:  # noqa: N802
            return self._status_message

        @Property(str, notify=activeLabelChanged)
        def activeLabel(self) -> str:  # noqa: N802
            return self._active_label

        @Property(str, notify=currentSampleIdChanged)
        def currentSampleId(self) -> str:  # noqa: N802
            return self._current_sample_id

        @Property(int, notify=frameStateChanged)
        def frameIndex(self) -> int:  # noqa: N802
            return self._frame_index

        @Property(int, notify=frameStateChanged)
        def framePosition(self) -> int:  # noqa: N802
            return self._frame_position

        @Property(int, notify=frameStateChanged)
        def frameCount(self) -> int:  # noqa: N802
            return self._frame_count

        @Property(float, notify=frameStateChanged)
        def activeSequenceFps(self) -> float:  # noqa: N802
            return self._source_fps

        @Property(str, notify=frameStateChanged)
        def leftState(self) -> str:  # noqa: N802
            return self.scene.left.payload.state

        @Property(str, notify=frameStateChanged)
        def rightState(self) -> str:  # noqa: N802
            return self.scene.right.payload.state

        @Property(bool, notify=frameStateChanged)
        def leftDimmed(self) -> bool:  # noqa: N802
            return self.scene.left.payload.dimmed

        @Property(bool, notify=frameStateChanged)
        def rightDimmed(self) -> bool:  # noqa: N802
            return self.scene.right.payload.dimmed

        @Property(int, notify=queueStateChanged)
        def queueCount(self) -> int:  # noqa: N802
            return len(self._queue.items)

        @Property(int, notify=queueStateChanged)
        def queueCompleted(self) -> int:  # noqa: N802
            return self._queue.completed

        @Property(float, notify=queueStateChanged)
        def queueProgress(self) -> float:  # noqa: N802
            count = len(self._queue.items)
            return 0.0 if count == 0 else min(1.0, self._queue.completed / count)

        @Property(bool, notify=playbackPlayingChanged)
        def playbackPlaying(self) -> bool:  # noqa: N802
            return bool(self._playback is not None and self._playback.playing)

        @Property(bool, notify=smoothRenderingChanged)
        def smoothRendering(self) -> bool:  # noqa: N802
            return self._smooth_rendering

        @Property(float, notify=renderMetricsChanged)
        def renderFps(self) -> float:  # noqa: N802
            return self._render_fps

        @Property(str, notify=renderMetricsChanged)
        def graphicsApi(self) -> str:  # noqa: N802
            return self._graphics_api

        @Property(bool, notify=diagnosticsVisibleChanged)
        def diagnosticsVisible(self) -> bool:  # noqa: N802
            return self._diagnostics_visible

        @Property(str, notify=topologyStateChanged)
        def topologyStatus(self) -> str:  # noqa: N802
            return self.scene.topology_status

        @Property(bool, notify=topologyStateChanged)
        def surfaceMode(self) -> bool:  # noqa: N802
            return self.scene.topology_available

        @Property(str, notify=recognitionStateChanged)
        def recognitionStatus(self) -> str:  # noqa: N802
            return self._recognition_status

        @Property(str, notify=recognitionStateChanged)
        def recognitionRole(self) -> str:  # noqa: N802
            return self._recognition_role

        @Property(str, notify=recognitionStateChanged)
        def recognitionScope(self) -> str:  # noqa: N802
            return self._recognition_scope

        @Property(str, notify=recognitionStateChanged)
        def recognitionReference(self) -> str:  # noqa: N802
            return self._recognition_reference

        @Property(str, notify=recognitionStateChanged)
        def recognitionCheckpoint(self) -> str:  # noqa: N802
            return self._recognition_checkpoint

        @Property(str, notify=recognitionStateChanged)
        def expectedCharacter(self) -> str:  # noqa: N802
            return self._expected_character

        @Property(str, notify=recognitionStateChanged)
        def predictedCharacter(self) -> str:  # noqa: N802
            return self._predicted_character

        @Property(str, notify=recognitionStateChanged)
        def confidenceText(self) -> str:  # noqa: N802
            return self._confidence_text

        @Property("QVariantList", constant=True)
        def keyboardRows(self) -> list[list[str]]:  # noqa: N802
            # Keep this boundary QVariant-friendly: QML receives nested lists
            # of authoritative characters rather than Python dict objects.
            # The keys stay in mapping order; QML applies RightToLeft layout
            # mirroring for presentation only.
            rows: list[list[str]] = []
            start = 0
            for size in (10, 9, 9):
                rows.append([key.character for key in self._keyboard.keys[start : start + size]])
                start += size
            return rows

        @Property(list, constant=True)
        def exemplarModes(self) -> list[str]:  # noqa: N802
            return list(EXEMPLAR_MODES)

        @Property(str, notify=modeChanged)
        def exemplarMode(self) -> str:  # noqa: N802
            return self._mode

        # ---- queue/application actions --------------------------------------

        def _emit_all_state(self) -> None:
            self.queuedTextChanged.emit()
            self.statusMessageChanged.emit()
            self.activeLabelChanged.emit()
            self.currentSampleIdChanged.emit()
            self.frameStateChanged.emit()
            self.queueStateChanged.emit()
            self.playbackPlayingChanged.emit()
            self.recognitionStateChanged.emit()

        def _set_status(self, message: str) -> None:
            self._status_message = str(message)
            self.statusMessageChanged.emit()

        def _set_active_item(self, item: Any | None) -> None:
            self._active_item = item
            if item is None:
                self._active_label = "No active sign"
                self._expected_character = EM_DASH
            else:
                self._active_label = str(item.character) if item.item_type == "sign" else "Neutral gap"
                self._expected_character = str(item.character) if item.item_type == "sign" else EM_DASH
            self.activeLabelChanged.emit()
            self.recognitionStateChanged.emit()

        def _reset_render_state(self) -> None:
            self._sequence = None
            self._playback = None
            self._gap_deadline = None
            self._frame_index = -1
            self._frame_position = -1
            self._frame_count = 0
            self._source_fps = 0.0
            self._current_sample_id = EM_DASH
            self._upload_presentation(self.scene.clear_sequence())
            self.frameStateChanged.emit()
            self.currentSampleIdChanged.emit()
            self.playbackPlayingChanged.emit()

        def _upload_presentation(self, frame: PresentationFrame) -> None:
            self.left_geometry.set_payload(frame.left)
            self.right_geometry.set_payload(frame.right)
            self.left_markers.update_markers(frame.left.marker_positions, frame.left.marker_valid)
            self.right_markers.update_markers(frame.right.marker_positions, frame.right.marker_valid)

        def _set_requested_text(self, text: str) -> None:
            self._requested_text = str(text)
            self.queuedTextChanged.emit()

        def _selected_seed(self) -> int | None:
            if self._mode == "random" and self._rng_seed is None:
                raise ValueError("random exemplar mode requires an explicit seed")
            return self._rng_seed

        def _begin_current(self) -> None:
            if self._closed:
                return
            item = self._queue.current
            if item is None:
                self._set_active_item(None)
                self._reset_render_state()
                self._set_status("Queue complete. Both neutral hands remain visible.")
                self._emit_queue_state()
                return
            if item.state == QueueState.PENDING:
                item = self._queue.start()
            self._set_active_item(item)
            self._active_token += 1
            token = self._active_token
            if item.item_type == "gap":
                self._reset_render_state()
                self._gap_deadline = time.monotonic() + max(0, int(item.gap_after_ms or 0)) / 1000.0
                self._set_status("Neutral gap — hands remain visible")
                self._emit_queue_state()
                return

            self._reset_render_state()
            self._current_sample_id = str(item.sample_id or EM_DASH)
            self.currentSampleIdChanged.emit()
            self._set_status(f"Loading stored sequence for {item.character}…")
            task = SequenceLoadTask(
                token,
                item,
                run_root=self._run_root,
                manifest_path=self._manifest_path,
            )
            task.signals.loaded.connect(self._sequence_loaded)
            task.signals.failed.connect(self._sequence_failed)
            self._load_pool.start(task)
            if self._recognition.enabled:
                self._start_recognition(item, token)
            self._emit_queue_state()

        def _emit_queue_state(self) -> None:
            self.queueStateChanged.emit()
            self.activeLabelChanged.emit()

        @Slot(str)
        def enqueueCharacter(self, character: str) -> None:  # noqa: N802
            value = str(character)
            if len(value) != 1:
                self._set_status("Select one Core-28 character at a time.")
                return
            was_idle = self._queue.current is None
            try:
                if value.isspace():
                    self._queue.enqueue_gap(value)
                else:
                    self._queue.enqueue_character(value, mode=self._mode, rng_seed=self._selected_seed())
            except (ValueError, OSError) as exc:
                self._set_status(f"Could not queue {value}: {exc}")
                return
            self._set_requested_text(self._requested_text + value)
            self._set_status(f"Queued {value}. Repeated characters remain separate events.")
            if was_idle:
                self._begin_current()
            else:
                self._emit_queue_state()

        def _enqueue_text(self, text: str) -> None:
            value = str(text)
            try:
                validation = self._keyboard.validate_text(value)
                if not validation.is_valid:
                    issue = validation.unsupported[0]
                    raise ValueError(f"unsupported character at position {issue.position}: {issue.character}")
                self._selected_seed()
                self._queue.clear()
                self._active_token += 1
                self._queue.enqueue_text(
                    value,
                    mode=self._mode,
                    rng_seed=self._rng_seed,
                    unsupported_policy="reject",
                )
            except (UnsupportedTextError, ValueError, OSError) as exc:
                self._set_status(f"Text was not queued: {exc}")
                return
            self._set_requested_text(value)
            if value:
                self._begin_current()
            else:
                self._reset_render_state()
                self._set_active_item(None)
                self._set_status("Queue cleared. Both neutral hands remain visible.")
                self._emit_queue_state()

        @Slot(str)
        def enqueueText(self, text: str) -> None:  # noqa: N802
            self._enqueue_text(text)

        def enqueue_text(self, text: str) -> None:
            """Python-friendly alias used by startup wiring and tests."""

            self._enqueue_text(text)

        @Slot()
        def appendSpace(self) -> None:  # noqa: N802
            self.enqueueCharacter(" ")

        def _rebuild_from_requested_text(self, text: str) -> None:
            self._active_token += 1
            self._queue.clear()
            self._reset_render_state()
            self._set_active_item(None)
            if text:
                try:
                    self._queue.enqueue_text(
                        text,
                        mode=self._mode,
                        rng_seed=self._rng_seed,
                        unsupported_policy="reject",
                    )
                except (UnsupportedTextError, ValueError) as exc:
                    self._set_status(f"Could not rebuild queue: {exc}")
                    return
                self._begin_current()
            else:
                self._set_status("Queue is empty. Both neutral hands remain visible.")
                self._emit_queue_state()

        @Slot()
        def backspace(self) -> None:
            if not self._requested_text:
                return
            value = self._requested_text[:-1]
            self._set_requested_text(value)
            self._rebuild_from_requested_text(value)

        @Slot()
        def clearQueue(self) -> None:  # noqa: N802
            self._requested_text = ""
            self.queuedTextChanged.emit()
            self._active_token += 1
            self._queue.clear()
            self._set_active_item(None)
            self._reset_render_state()
            self._set_status("Queue cleared. Both neutral hands remain visible.")
            self._emit_queue_state()

        @Slot()
        def restart(self) -> None:
            if self._queue.current is None and self._queue.items:
                self._queue.reset()
                self._begin_current()
                return
            if self._playback is not None:
                self._playback.restart()
                self._update_from_playback(self._playback.tick())
                self._set_status("Current sequence restarted.")

        @Slot()
        def playPause(self) -> None:  # noqa: N802
            if self._playback is None:
                if self._queue.current is not None:
                    self._set_status("Waiting for the stored sequence to load…")
                return
            if self._playback.playing:
                self._playback.pause()
                self._set_status("Playback paused.")
            else:
                self._playback.play()
                self._set_status("Playback playing.")
            self.playbackPlayingChanged.emit()

        @Slot(float)
        def setSpeed(self, speed: float) -> None:  # noqa: N802
            value = float(speed)
            if value not in SPEEDS:
                return
            if self._playback is not None:
                self._playback.set_speed(value)
            self._speed = value
            self._set_status(f"Playback speed {value:g}×.")

        @Slot(bool)
        def setSmoothRendering(self, enabled: bool) -> None:  # noqa: N802
            self._smooth_rendering = bool(enabled)
            self.smoothRenderingChanged.emit()
            self._set_status("Smooth rendering on." if self._smooth_rendering else "Exact stored frames on.")

        @Slot(str)
        def setExemplarMode(self, mode: str) -> None:  # noqa: N802
            value = str(mode)
            if value not in EXEMPLAR_MODES:
                return
            self._mode = value
            self.modeChanged.emit()
            self._set_status(f"Exemplar mode: {value}.")

        @Slot()
        def resetView(self) -> None:  # noqa: N802
            self.resetViewRequested.emit()

        @Slot()
        def toggleDiagnostics(self) -> None:  # noqa: N802
            self._diagnostics_visible = not self._diagnostics_visible
            self.diagnosticsVisibleChanged.emit()

        @Slot()
        def recordRenderFrame(self) -> None:  # noqa: N802
            """Record one Qt Quick FrameAnimation tick for diagnostics."""

            now = time.monotonic()
            self._render_frame_count += 1
            elapsed = now - self._render_window_start
            if elapsed >= 1.0:
                self._render_fps = self._render_frame_count / elapsed
                self._render_frame_count = 0
                self._render_window_start = now
                self.renderMetricsChanged.emit()

        @Slot()
        def shutdown(self) -> None:
            """Stop timers/workers without destroying or recreating the scene."""

            if self._closed:
                return
            self._closed = True
            self._active_token += 1
            self._timer.stop()
            self._load_pool.clear()
            self._recognition_pool.clear()
            # Closing is the one place where waiting is preferable to leaving
            # a checkpoint/inference worker attached to a dying Qt process.
            self._load_pool.waitForDone(10000)
            self._recognition_pool.waitForDone(10000)

        @Slot(str)
        def setGraphicsApi(self, value: str) -> None:  # noqa: N802
            if value:
                self._graphics_api = str(value)
                self.renderMetricsChanged.emit()

        # ---- background tasks -----------------------------------------------

        def _start_checkpoint_load(self) -> None:
            if not self._checkpoint_path or self._labels_path is None:
                self._recognition_status = "Recognition disabled — labels path unavailable"
                self.recognitionStateChanged.emit()
                return
            token = self._active_token
            task = CheckpointLoadTask(
                token,
                self._checkpoint_path,
                run_root=self._run_root,
                labels_path=self._labels_path,
                device=self._device,
            )
            task.signals.loaded.connect(self._checkpoint_loaded)
            task.signals.failed.connect(self._checkpoint_failed)
            self._load_pool.start(task)

        @Slot(int, object)
        def _checkpoint_loaded(self, token: int, adapter: Any) -> None:
            if self._closed:
                return
            self._recognition.set_adapter(adapter)
            metadata = adapter.metadata
            self._recognition_status = "Recognition ready"
            self._recognition_role = str(metadata.role_display)
            self._recognition_scope = str(metadata.training_scope_display)
            reference = str(metadata.scientific_reference_display)
            if metadata.is_deployment and ("67.63%" not in reference or "0.6607" not in reference):
                reference = "LOSO reference only (not deployment accuracy): 67.63% accuracy / 0.6607 macro F1"
            self._recognition_reference = reference
            self._recognition_checkpoint = Path(metadata.path).name
            self._recognition_error = ""
            self.recognitionStateChanged.emit()
            if self._active_item is not None and self._active_item.item_type == "sign":
                self._start_recognition(self._active_item, self._active_token)

        @Slot(int, str)
        def _checkpoint_failed(self, token: int, error: str) -> None:
            if self._closed:
                return
            self._recognition.disable(error)
            self._recognition_status = "Recognition disabled"
            self._recognition_role = "Visualization-only mode"
            self._recognition_scope = EM_DASH
            self._recognition_reference = EM_DASH
            self._recognition_checkpoint = EM_DASH
            self._recognition_error = str(error)
            self._set_status(f"Checkpoint unavailable; visualizer-only mode continues. {error}")
            self.recognitionStateChanged.emit()

        def _start_recognition(self, item: Any, token: int) -> None:
            task = RecognitionTask(token, item, self._recognition)
            task.signals.loaded.connect(self._recognition_loaded)
            task.signals.failed.connect(self._recognition_failed)
            self._recognition_pool.start(task)

        @Slot(int, object)
        def _recognition_loaded(self, token: int, result: Any) -> None:
            if self._closed or token != self._active_token or self._active_item is None:
                return
            if result is None:
                return
            if bool(getattr(result, "available", False)):
                self._recognition_status = "Prediction ready"
                self._predicted_character = str(getattr(result, "predicted_character", None) or EM_DASH)
                confidence = getattr(result, "confidence", None)
                self._confidence_text = EM_DASH if confidence is None else f"{float(confidence):.1%}"
            else:
                self._recognition_status = "Prediction unavailable"
                self._predicted_character = EM_DASH
                self._confidence_text = EM_DASH
            self.recognitionStateChanged.emit()

        @Slot(int, str)
        def _recognition_failed(self, token: int, error: str) -> None:
            if self._closed or token != self._active_token:
                return
            self._recognition_status = "Prediction unavailable"
            self._predicted_character = EM_DASH
            self._confidence_text = EM_DASH
            self._recognition_error = str(error)
            self.recognitionStateChanged.emit()

        @Slot(int, object)
        def _sequence_loaded(self, token: int, sequence: Any) -> None:
            if self._closed or token != self._active_token or self._active_item is None:
                return
            if sequence is None:
                self._sequence_failed(token, "sequence loader returned no sign sequence")
                return
            try:
                self.scene.attach_sequence(sequence)
                self._sequence = sequence
                self._playback = PersistentPlaybackController(
                    sequence.timestamps,
                    sequence.frame_indices,
                    speed=self._speed,
                )
                self._frame_count = len(sequence)
                duration = float(sequence.timestamps[-1] - sequence.timestamps[0])
                self._source_fps = (len(sequence) - 1) / duration if duration > 0 else 0.0
                self._set_status(f"Playing {self._active_item.character} from stored TASK-008 frames.")
                self._update_from_playback(self._playback.play())
                self._emit_queue_state()
            except Exception as exc:  # noqa: BLE001 - explicit unavailable item state
                self._sequence_failed(token, f"{type(exc).__name__}: {exc}")

        @Slot(int, str)
        def _sequence_failed(self, token: int, error: str) -> None:
            if self._closed or token != self._active_token:
                return
            item = self._queue.current
            if item is not None:
                self._queue.fail(error, unavailable=True)
            self._set_status(f"Stored sequence unavailable; continuing queue. {error}")
            self._reset_render_state()
            self._set_active_item(None)
            self._emit_queue_state()
            self._begin_current()

        # ---- presentation clock --------------------------------------------

        def _update_from_playback(self, state: PlaybackDisplayState) -> None:
            if self._sequence is None:
                return
            frame = self.scene.update_sequence_frame(
                state.position,
                interpolation_alpha=state.interpolation_alpha,
                smooth=self._smooth_rendering,
            )
            self._upload_presentation(frame)
            self._frame_index = int(state.frame_index)
            self._frame_position = int(state.position)
            self._frame_count = len(self._sequence)
            self.frameStateChanged.emit()
            self.playbackPlayingChanged.emit()

        def _finish_current_if_needed(self) -> None:
            if self._playback is None or self._playback.playing or self._queue.current is None:
                return
            if not self._playback.at_end:
                return
            previous = self._queue.current
            self._queue.advance()
            self._set_status(f"Completed {previous.character}; moving to the next queue event.")
            self._begin_current()

        def _tick(self) -> None:
            if self._closed:
                return
            now = time.monotonic()
            if self._gap_deadline is not None and now >= self._gap_deadline:
                self._gap_deadline = None
                if self._queue.current is not None:
                    self._queue.advance()
                self._begin_current()
            elif self._playback is not None:
                state = self._playback.tick(now)
                self._update_from_playback(state)
                self._finish_current_if_needed()

else:

    class Core28ApplicationController:  # type: ignore[no-redef]
        """Import-safe placeholder when PySide6 is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PySide6 is required for the Core-28 application")


__all__ = ["Core28ApplicationController", "DEFAULT_SENSOR_LAYOUT", "QT_CONTROLLER_AVAILABLE"]
