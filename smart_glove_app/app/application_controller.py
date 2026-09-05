"""Qt-facing application state for the persistent Core-28 scene."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from visualizer.contract import validate_sensor_layout
from visualizer.keyboard import Core28Keyboard
from visualizer.mapping import Core28Resolver
from visualizer.queue import PlaybackQueue, QueueState, UnsupportedTextError

from smart_glove_app.rendering.hand_mesh_state import (
    PersistentRenderScene,
    PresentationFrame,
)
from smart_glove_app.rendering.mano_topology import ManoTopology
from smart_glove_app.rendering.qt_geometry import QtHandGeometry
from smart_glove_app.rendering.glb_index import GlbIndexError, build_scene_index
from smart_glove_app.rendering.presentation_rig import (
    VIEW_MODES,
    PresentationRig,
    load_presentation_rig,
)
from smart_glove_app.rendering.sensor_markers import SensorMarkerModel

from .hand_pose_solver import HandPose, HandPoseSolver
from .motion_quality import (
    PresentationTransition,
    TransitionConfig,
    copy_pose_map,
)
from .playback_controller import (
    PersistentPlaybackController,
    PlaybackBoundaryTrace,
    PlaybackDisplayState,
)
from .recognition_bridge import RecognitionBridge
from .qt_workers import CheckpointLoadTask, RecognitionTask, SequenceLoadTask


try:
    from PySide6.QtCore import (
        QUrl,
        QObject,
        Property,
        QThreadPool,
        QTimer,
        Signal,
        Slot,
    )

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
MATERIAL_MODES = ("SKIN", "GLOVE", "WIREFRAME")
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
        handPoseChanged = Signal()
        rigAssetStateChanged = Signal()
        viewModeChanged = Signal()
        materialModeChanged = Signal()
        speedChanged = Signal()

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
            boundary_hold_ms: float = 80.0,
            transition_min_ms: float = 150.0,
            transition_max_ms: float = 350.0,
            rig_profile: PresentationRig | None = None,
            rig_asset_path: str | Path | None = None,
            debug_mano_points: bool = False,
            diagnostics_visible: bool = False,
            view_mode: str = "PALM",
            material_mode: str = "SKIN",
        ) -> None:
            super().__init__()
            if mode not in EXEMPLAR_MODES:
                raise ValueError(f"unsupported exemplar mode: {mode!r}")
            if float(speed) not in SPEEDS:
                raise ValueError(f"speed must be one of {SPEEDS}")
            self._run_root = Path(run_root).expanduser().resolve()
            self._manifest_path = (
                Path(manifest_path).expanduser().resolve() if manifest_path else None
            )
            self._labels_path = (
                Path(labels_path).expanduser().resolve() if labels_path else None
            )
            self._resolver = resolver
            self._keyboard = Core28Keyboard(resolver.mapping)
            self._queue = PlaybackQueue(resolver)
            self._device = str(device)
            self._mode = mode
            self._rng_seed = rng_seed
            self._speed = float(speed)
            self._transition_config = TransitionConfig(
                boundary_hold_ms=float(boundary_hold_ms),
                minimum_duration_ms=float(transition_min_ms),
                maximum_duration_ms=float(transition_max_ms),
            )
            self._requested_text = ""
            self._active_item: Any | None = None
            self._sequence: Any | None = None
            self._playback: PersistentPlaybackController | None = None
            self._gap_deadline: float | None = None
            self._transition: PresentationTransition | None = None
            self._transition_source: dict[str, HandPose] | None = None
            self._transition_anchor_time: float | None = None
            self._transition_phase = "IDLE"
            self._transition_distance_deg = 0.0
            self._transition_duration_ms = 0.0
            self._active_token = 0
            self._closed = False
            self._smooth_rendering = bool(smooth_rendering)
            self._diagnostics_visible = bool(diagnostics_visible)
            self._debug_mano_points = bool(debug_mano_points)
            self._rig_profile = rig_profile or load_presentation_rig()
            self._solver = HandPoseSolver(self._rig_profile)
            self._hand_pose = self._solver.neutral_qml_pose()
            self._last_hand_poses: dict[str, HandPose] = {
                side: self._solver.neutral_pose(side) for side in ("LEFT", "RIGHT")
            }
            self._active_trace: PlaybackBoundaryTrace | None = None
            self._motion_traces: list[dict[str, Any]] = []
            # Presentation state. Deliberately separate from anything recorded:
            # playing a sign can change bones, never the view or the layout.
            self._view_mode = str(view_mode).upper()
            if self._view_mode not in VIEW_MODES:
                raise ValueError(f"unsupported view mode: {view_mode!r}")
            self._material_mode = str(material_mode).upper()
            if self._material_mode not in MATERIAL_MODES:
                raise ValueError(f"unsupported appearance: {material_mode!r}")
            # One GLB per hand: see the rig profile's one_file_per_hand note.
            self._rig_asset_dir = (
                Path(rig_asset_path).expanduser().resolve()
                if rig_asset_path is not None
                else None
            )
            self._hand_assets: dict[str, Path] = {}
            self._scene_index: dict[str, Any] = {}
            self._scene_index_error = ""
            if self._rig_asset_dir is not None and self._rig_asset_dir.is_dir():
                sides: dict[str, Any] = {}
                for side, filename in self._rig_profile.glb_filenames.items():
                    candidate = self._rig_asset_dir / filename
                    if not candidate.is_file():
                        self._scene_index_error = (
                            f"missing hand asset: {candidate.name}"
                        )
                        break
                    try:
                        index = build_scene_index(
                            candidate,
                            roots={side: self._rig_profile.roots[side]},
                            bones=self._rig_profile.required_bones,
                        )
                    except GlbIndexError as exc:
                        self._scene_index_error = str(exc)
                        break
                    self._hand_assets[side] = candidate
                    sides[side] = index["sides"][side]
                else:
                    self._scene_index = {"sides": sides}
            elif self._rig_asset_dir is None:
                self._scene_index_error = "no hand asset directory was supplied"
            else:
                self._scene_index_error = (
                    f"hand asset directory not found: {self._rig_asset_dir}"
                )

            self._rig_asset_available = (
                bool(self._scene_index) and not self._scene_index_error
            )
            self._rig_asset_status = (
                "Hand assets ready"
                if self._rig_asset_available
                else (
                    self._scene_index_error
                    or "Hand assets unavailable — use --rig-asset"
                )
            )

            # These are the two persistent GPU geometry providers.  They are
            # intentionally constructed before any queue item is loaded.
            self.scene = PersistentRenderScene(topology)
            self.left_geometry = QtHandGeometry("LEFT", topology)
            self.right_geometry = QtHandGeometry("RIGHT", topology)
            self.left_markers = SensorMarkerModel(DEFAULT_SENSOR_LAYOUT, self)
            self.right_markers = SensorMarkerModel(DEFAULT_SENSOR_LAYOUT, self)
            self._upload_presentation(self.scene.clear_sequence())

            self._recognition = RecognitionBridge()
            self._checkpoint_path = (
                str(Path(checkpoint).expanduser().resolve()) if checkpoint else ""
            )
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
            self._hand_pose_update_count = 0

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
            return bool(
                self._transition is not None
                or (self._playback is not None and self._playback.playing)
            )

        @Property(bool, notify=smoothRenderingChanged)
        def smoothRendering(self) -> bool:  # noqa: N802
            return self._smooth_rendering

        @Property(float, notify=renderMetricsChanged)
        def renderFps(self) -> float:  # noqa: N802
            return self._render_fps

        @Property(str, notify=renderMetricsChanged)
        def graphicsApi(self) -> str:  # noqa: N802
            return self._graphics_api

        @Property(int, notify=renderMetricsChanged)
        def rigPoseUpdateCount(self) -> int:  # noqa: N802
            return self._hand_pose_update_count

        @Property(str, notify=renderMetricsChanged)
        def transitionPhase(self) -> str:  # noqa: N802
            return self._transition_phase

        @Property(float, notify=renderMetricsChanged)
        def transitionDistanceDeg(self) -> float:  # noqa: N802
            return self._transition_distance_deg

        @Property(float, notify=renderMetricsChanged)
        def transitionDurationMs(self) -> float:  # noqa: N802
            return self._transition_duration_ms

        @Property("QVariantList", notify=renderMetricsChanged)
        def motionTrace(self) -> list[dict[str, Any]]:  # noqa: N802
            """Completed source/queue boundary traces for diagnostics and reports."""

            return list(self._motion_traces)

        @Property(bool, notify=diagnosticsVisibleChanged)
        def diagnosticsVisible(self) -> bool:  # noqa: N802
            return self._diagnostics_visible

        @Property(str, notify=topologyStateChanged)
        def topologyStatus(self) -> str:  # noqa: N802
            return self.scene.topology_status

        @Property(bool, notify=topologyStateChanged)
        def surfaceMode(self) -> bool:  # noqa: N802
            return self.scene.topology_available

        @Property(float, notify=speedChanged)
        def speed(self) -> float:  # noqa: N802
            return self._speed

        @Property(str, notify=viewModeChanged)
        def viewMode(self) -> str:  # noqa: N802
            return self._view_mode

        @Property(str, notify=materialModeChanged)
        def materialMode(self) -> str:  # noqa: N802
            return self._material_mode

        @Property("QVariantMap", notify=handPoseChanged)
        def handPose(self) -> dict[str, Any]:  # noqa: N802
            """Render-only local bone rotations for both persistent hands."""

            return self._hand_pose

        @Property("QVariantMap", constant=True)
        def rigProfile(self) -> dict[str, Any]:  # noqa: N802
            payload = self._rig_profile.as_qml()
            payload["sceneIndex"] = self._scene_index
            payload["assetUrls"] = {
                side: QUrl.fromLocalFile(str(path)).toString()
                for side, path in self._hand_assets.items()
            }
            return payload

        @Property(bool, notify=rigAssetStateChanged)
        def rigAssetAvailable(self) -> bool:  # noqa: N802
            return self._rig_asset_available

        @Property(str, notify=rigAssetStateChanged)
        def rigAssetPath(self) -> str:  # noqa: N802
            return str(self._rig_asset_dir) if self._rig_asset_dir is not None else ""

        @Property(str, notify=rigAssetStateChanged)
        def leftAssetUrl(self) -> str:  # noqa: N802
            """QML-friendly file URL for the LEFT hand's RuntimeLoader."""

            path = self._hand_assets.get("LEFT")
            return QUrl.fromLocalFile(str(path)).toString() if path is not None else ""

        @Property(str, notify=rigAssetStateChanged)
        def rightAssetUrl(self) -> str:  # noqa: N802
            """QML-friendly file URL for the RIGHT hand's RuntimeLoader."""

            path = self._hand_assets.get("RIGHT")
            return QUrl.fromLocalFile(str(path)).toString() if path is not None else ""

        @Property(str, notify=rigAssetStateChanged)
        def rigAssetStatus(self) -> str:  # noqa: N802
            return self._rig_asset_status

        @Property(bool, constant=True)
        def debugManoPoints(self) -> bool:  # noqa: N802
            return self._debug_mano_points

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
                rows.append(
                    [key.character for key in self._keyboard.keys[start : start + size]]
                )
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
                self._active_label = (
                    str(item.character) if item.item_type == "sign" else "Neutral gap"
                )
                self._expected_character = (
                    str(item.character) if item.item_type == "sign" else EM_DASH
                )
            self.activeLabelChanged.emit()
            self.recognitionStateChanged.emit()

        def _set_transition_state(
            self,
            phase: str,
            *,
            distance_deg: float | None = None,
            duration_ms: float | None = None,
        ) -> None:
            self._transition_phase = str(phase)
            if distance_deg is not None:
                self._transition_distance_deg = float(distance_deg)
            if duration_ms is not None:
                self._transition_duration_ms = float(duration_ms)
            self.renderMetricsChanged.emit()
            self.playbackPlayingChanged.emit()

        def _reset_render_state(self, *, keep_pose: bool = False) -> None:
            self._sequence = None
            self._playback = None
            self._gap_deadline = None
            self._transition = None
            self._transition_source = None
            self._transition_anchor_time = None
            self._set_transition_state("IDLE", distance_deg=0.0, duration_ms=0.0)
            self._frame_index = -1
            self._frame_position = -1
            self._frame_count = 0
            self._source_fps = 0.0
            self._current_sample_id = EM_DASH
            self._solver.reset()
            if not keep_pose:
                self._last_hand_poses = {
                    side: self._solver.neutral_pose(side) for side in ("LEFT", "RIGHT")
                }
                self._hand_pose = self._solver.neutral_qml_pose()
                self.handPoseChanged.emit()
            self._upload_presentation(self.scene.clear_sequence())
            self.frameStateChanged.emit()
            self.currentSampleIdChanged.emit()
            self.playbackPlayingChanged.emit()

        def _upload_presentation(self, frame: PresentationFrame) -> None:
            self.left_geometry.set_payload(frame.left)
            self.right_geometry.set_payload(frame.right)
            self.left_markers.update_markers(
                frame.left.marker_positions, frame.left.marker_valid
            )
            self.right_markers.update_markers(
                frame.right.marker_positions, frame.right.marker_valid
            )

        def _set_requested_text(self, text: str) -> None:
            self._requested_text = str(text)
            self.queuedTextChanged.emit()

        def _selected_seed(self) -> int | None:
            if self._mode == "random" and self._rng_seed is None:
                raise ValueError("random exemplar mode requires an explicit seed")
            return self._rng_seed

        def _begin_current(
            self,
            *,
            transition_source: dict[str, HandPose] | None = None,
            transition_anchor_time: float | None = None,
        ) -> None:
            if self._closed:
                return
            item = self._queue.current
            if item is None:
                self._set_active_item(None)
                # Hold the last rendered pose rather than snapping the hands
                # open: the final shape of a sign is the part worth reading.
                self._reset_render_state(keep_pose=True)
                self._set_status("Queue complete. The final sign pose is held.")
                self._emit_queue_state()
                return
            if item.state == QueueState.PENDING:
                item = self._queue.start()
            self._set_active_item(item)
            self._active_token += 1
            token = self._active_token
            if item.item_type == "gap":
                self._reset_render_state()
                self._gap_deadline = (
                    time.monotonic() + max(0, int(item.gap_after_ms or 0)) / 1000.0
                )
                self._set_status("Neutral gap — hands remain visible")
                self._emit_queue_state()
                return

            # Preserve the completed sign while the next stored sequence loads;
            # the presentation-only transition is built once its first source
            # frame has been solved below.
            self._reset_render_state(keep_pose=transition_source is not None)
            if transition_source is not None:
                self._transition_source = copy_pose_map(transition_source)
                self._transition_anchor_time = (
                    time.monotonic()
                    if transition_anchor_time is None
                    else float(transition_anchor_time)
                )
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
                    self._queue.enqueue_character(
                        value, mode=self._mode, rng_seed=self._selected_seed()
                    )
            except (ValueError, OSError) as exc:
                self._set_status(f"Could not queue {value}: {exc}")
                return
            self._set_requested_text(self._requested_text + value)
            self._set_status(
                f"Queued {value}. Repeated characters remain separate events."
            )
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
                    raise ValueError(
                        f"unsupported character at position {issue.position}: {issue.character}"
                    )
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
            if self._transition is not None and self._playback is not None:
                # An explicit restart is a user action, so it cancels the
                # presentation-only boundary and starts the current stored
                # sequence at its exact first source frame.
                self._transition = None
                self._transition_source = None
                self._transition_anchor_time = None
                self._solver.reset()
                self._set_transition_state("IDLE", distance_deg=0.0, duration_ms=0.0)
                self._playback.restart()
                self._update_from_playback(self._playback.play())
                self._set_status("Current sequence restarted.")
                return
            if self._playback is not None:
                self._solver.reset()
                self._playback.restart()
                self._update_from_playback(self._playback.play(time.monotonic()))
                self._set_status("Current sequence restarted.")

        @Slot()
        def playPause(self) -> None:  # noqa: N802
            if self._transition is not None:
                self._set_status("Transition in progress; playback remains continuous.")
                return
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
            self.speedChanged.emit()
            self._set_status(f"Playback speed {value:g}×.")

        @Slot(bool)
        def setSmoothRendering(self, enabled: bool) -> None:  # noqa: N802
            self._smooth_rendering = bool(enabled)
            self.smoothRenderingChanged.emit()
            self._set_status(
                "Smooth rendering on."
                if self._smooth_rendering
                else "Exact stored frames on."
            )

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
            self._view_mode = "PALM"
            self.viewModeChanged.emit()
            self.resetViewRequested.emit()

        @Slot(str)
        def setViewMode(self, mode: str) -> None:  # noqa: N802
            value = str(mode).upper()
            if value not in VIEW_MODES:
                self._set_status(f"Unknown view: {mode}")
                return
            if value == self._view_mode:
                return
            self._view_mode = value
            self.viewModeChanged.emit()
            self._set_status("Palm view" if value == "PALM" else "Back-of-hand view")

        @Slot()
        def toggleViewMode(self) -> None:  # noqa: N802
            self.setViewMode("BACK" if self._view_mode == "PALM" else "PALM")

        @Slot(str)
        def setMaterialMode(self, mode: str) -> None:  # noqa: N802
            value = str(mode).upper()
            if value not in MATERIAL_MODES:
                self._set_status(f"Unknown appearance: {mode}")
                return
            self._material_mode = value
            self.materialModeChanged.emit()

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

        @Slot(str)
        def setRigAssetStatus(self, value: str) -> None:  # noqa: N802
            """Receive the RuntimeLoader result without changing asset state."""

            status = str(value).strip()
            if not status or status == self._rig_asset_status:
                return
            self._rig_asset_status = status
            self.rigAssetStateChanged.emit()

        # ---- background tasks -----------------------------------------------

        def _start_checkpoint_load(self) -> None:
            if not self._checkpoint_path or self._labels_path is None:
                self._recognition_status = (
                    "Recognition disabled — labels path unavailable"
                )
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
            if metadata.is_deployment and (
                "67.63%" not in reference or "0.6607" not in reference
            ):
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
            self._set_status(
                f"Checkpoint unavailable; visualizer-only mode continues. {error}"
            )
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
                self._predicted_character = str(
                    getattr(result, "predicted_character", None) or EM_DASH
                )
                confidence = getattr(result, "confidence", None)
                self._confidence_text = (
                    EM_DASH if confidence is None else f"{float(confidence):.1%}"
                )
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

        def _preview_first_pose(self, sequence: Any) -> dict[str, HandPose]:
            """Solve a target endpoint without carrying state into playback."""

            self._solver.reset()
            target = self._solver.frame_pose(sequence.frame_at(0))
            snapshot = copy_pose_map(target)
            self._solver.reset()
            return snapshot

        @Slot(int, object)
        def _sequence_loaded(self, token: int, sequence: Any) -> None:
            if self._closed or token != self._active_token or self._active_item is None:
                return
            if sequence is None:
                self._sequence_failed(
                    token, "sequence loader returned no sign sequence"
                )
                return
            try:
                self.scene.attach_sequence(sequence)
                self._solver.reset()
                self._sequence = sequence
                self._playback = PersistentPlaybackController(
                    sequence.timestamps,
                    sequence.frame_indices,
                    speed=self._speed,
                )
                self._frame_count = len(sequence)
                duration = float(sequence.timestamps[-1] - sequence.timestamps[0])
                self._source_fps = (
                    (len(sequence) - 1) / duration if duration > 0 else 0.0
                )
                self._active_trace = PlaybackBoundaryTrace.for_sequence(
                    str(sequence.sample_id),
                    str(self._active_item.character),
                    sequence.frame_indices,
                )
                if self._transition_source is not None:
                    source = copy_pose_map(self._transition_source)
                    anchor = self._transition_anchor_time
                    if anchor is None:
                        anchor = time.monotonic()
                    loaded_at = time.monotonic()
                    # The target endpoint is only known after the worker has
                    # loaded its stored sequence.  If that takes longer than
                    # the configured hold, start the blend at load time
                    # rather than silently consuming the whole transition
                    # while the previous pose is being held.  A fast load
                    # retains the original final-pose hold exactly.
                    hold_seconds = self._transition_config.boundary_hold_ms / 1000.0
                    transition_started_at = max(
                        float(anchor), loaded_at - hold_seconds
                    )
                    target = self._preview_first_pose(sequence)
                    self._transition = PresentationTransition(
                        source,
                        target,
                        started_at=transition_started_at,
                        rig=self._rig_profile,
                        config=self._transition_config,
                    )
                    self._active_trace.set_transition_plan(
                        distance_degrees=self._transition.plan.distance_degrees,
                        duration_ms=self._transition.plan.duration_ms,
                        hold_ms=self._transition.plan.hold_ms,
                    )
                    self._transition_source = None
                    self._transition_anchor_time = None
                    self._set_transition_state(
                        "BOUNDARY_HOLD",
                        distance_deg=self._transition.plan.distance_degrees,
                        duration_ms=self._transition.plan.duration_ms,
                    )
                    self._set_status(
                        f"Transitioning to {self._active_item.character}; stored frames start after the boundary."
                    )
                else:
                    self._set_status(
                        f"Playing {self._active_item.character} from stored TASK-008 frames."
                    )
                    self._update_from_playback(self._playback.play(time.monotonic()))
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

        def _publish_hand_poses(self, poses: dict[str, HandPose]) -> None:
            """Publish a new absolute pose map to the persistent QML rig."""

            self._last_hand_poses = copy_pose_map(poses)
            self._hand_pose = HandPoseSolver.qml_pose(poses)
            self._hand_pose_update_count += 1
            self.handPoseChanged.emit()

        def _update_from_playback(self, state: PlaybackDisplayState) -> None:
            if self._sequence is None:
                return
            source_frame = self._sequence.frame_at(state.position)
            next_frame = (
                self._sequence.frames[state.position + 1]
                if state.position + 1 < len(self._sequence.frames)
                else None
            )
            # This is a presentation-only skeleton update.  Recognition still
            # receives the queue item through RecognitionTask and never sees
            # these render-time interpolated quaternions.
            poses = self._solver.frame_pose(
                source_frame,
                next_frame=next_frame,
                interpolation_alpha=state.interpolation_alpha,
                smooth=self._smooth_rendering,
            )
            self._publish_hand_poses(poses)
            if self._active_trace is not None:
                self._active_trace.record(state.position)
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

        def _finish_current_if_needed(self, now: float) -> None:
            if (
                self._playback is None
                or self._playback.playing
                or self._queue.current is None
            ):
                return
            if not self._playback.at_end:
                return
            if self._active_trace is not None:
                # The queue must not advance until the terminal source anchor
                # has been published.  The normal path reaches this through
                # the final PlaybackDisplayState; the guard also makes a
                # dropped-event/race visible in the trace instead of hiding it.
                if not self._active_trace.last_frame_presented:
                    return
                self._active_trace.mark_queue_advance()
                self._motion_traces.append(self._active_trace.to_dict())
                self._active_trace = None
            previous = self._queue.current
            final_pose = copy_pose_map(self._last_hand_poses)
            self._queue.advance()
            self._set_status(
                f"Completed {previous.character}; moving to the next queue event."
            )
            self._begin_current(
                transition_source=(
                    final_pose if self._queue.current is not None else None
                ),
                transition_anchor_time=now,
            )

        def _tick_transition(self, now: float) -> None:
            transition = self._transition
            if transition is None:
                return
            sample = transition.sample(now)
            self._set_transition_state(
                sample.phase,
                distance_deg=transition.plan.distance_degrees,
                duration_ms=transition.plan.duration_ms,
            )
            self._publish_hand_poses(dict(sample.poses))
            if not sample.done:
                return
            self._transition = None
            self._set_transition_state("PLAYING", distance_deg=0.0, duration_ms=0.0)
            if self._playback is not None and self._sequence is not None:
                self._set_status(
                    f"Playing {self._active_item.character} from stored TASK-008 frames."
                )
                self._update_from_playback(self._playback.play(now))
                self._emit_queue_state()

        def _exact_source_state(
            self, position: int, *, playing: bool
        ) -> PlaybackDisplayState:
            """Build an exact anchor state for a source position catch-up."""

            frame = self._sequence.frame_at(position)
            return PlaybackDisplayState(
                position=int(position),
                frame_index=int(frame.frame_index),
                timestamp_seconds=float(frame.timestamp_seconds),
                interpolation_alpha=0.0,
                playing=bool(playing),
            )

        def _tick_source_playback(self, now: float) -> None:
            if self._playback is None or self._sequence is None:
                return
            state = self._playback.tick(now)
            previous_position = self._frame_position
            # If a GUI callback arrives late, publish each crossed source
            # anchor in order before the clock's current interpolated state.
            # This preserves source-frame visitation and solver state without
            # changing timestamps, arrays, masks, or recognizer input.
            if 0 <= previous_position < state.position - 1:
                for position in range(previous_position + 1, state.position):
                    self._update_from_playback(
                        self._exact_source_state(position, playing=True)
                    )
            self._update_from_playback(state)
            self._finish_current_if_needed(now)

        def _tick(self) -> None:
            if self._closed:
                return
            now = time.monotonic()
            if self._gap_deadline is not None and now >= self._gap_deadline:
                self._gap_deadline = None
                if self._queue.current is not None:
                    self._queue.advance()
                self._begin_current()
            elif self._transition is not None:
                self._tick_transition(now)
            elif self._playback is not None:
                self._tick_source_playback(now)

else:

    class Core28ApplicationController:  # type: ignore[no-redef]
        """Import-safe placeholder when PySide6 is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PySide6 is required for the Core-28 application")


__all__ = [
    "Core28ApplicationController",
    "DEFAULT_SENSOR_LAYOUT",
    "QT_CONTROLLER_AVAILABLE",
]
