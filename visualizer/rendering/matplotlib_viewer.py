"""Matplotlib/Tk interactive 3D virtual-glove playback."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..contract import TRACK_ORDER, PlaybackSequence, SensorReading
from ..geometry import SKELETON_EDGES, sensor_marker_positions, sequence_bounds
from ..playback import PlaybackController


class MatplotlibGloveViewer:
    """Render one sequence with a 3D geometry view and synchronized dashboard.

    Matplotlib is imported lazily so artifact loading and unit tests do not
    require a display server.  Stored MANO vertices are drawn as a dense point
    cloud because TASK-008 serializes vertices but not MANO triangle topology;
    tracked 21-joint chains are drawn over that cloud.
    """

    def __init__(
        self,
        sequence: PlaybackSequence,
        *,
        initial_position: int = 0,
        speed: float = 1.0,
        figure: Any | None = None,
        on_sequence_finished: Callable[[], None] | None = None,
        defer_initial_draw: bool = False,
        use_idle_draw: bool = True,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            from matplotlib.widgets import Button, RadioButtons, Slider
        except ImportError as exc:  # pragma: no cover - depends on local GUI environment
            raise RuntimeError(
                "TASK-007A GUI requires matplotlib; install the optional visualizer dependency"
            ) from exc

        self._plt = plt
        self.sequence = sequence
        self._on_sequence_finished = on_sequence_finished
        self._defer_initial_draw = bool(defer_initial_draw)
        self._use_idle_draw = bool(use_idle_draw)
        self.controller = PlaybackController(sequence.timestamps, sequence.frame_indices, speed=speed)
        self.controller.seek(initial_position)
        self._ignore_slider = False
        self._bounds = sequence_bounds(sequence)

        self.fig = figure if figure is not None else plt.figure(figsize=(16, 9), num=f"TASK-007A — {sequence.sample_id}")
        grid = GridSpec(1, 2, figure=self.fig, width_ratios=(1.08, 0.92))
        self.ax_3d = self.fig.add_subplot(grid[0, 0], projection="3d")
        self.ax_dashboard = self.fig.add_subplot(grid[0, 1])
        self.fig.subplots_adjust(left=0.035, right=0.985, top=0.93, bottom=0.19, wspace=0.08)

        self.ax_slider = self.fig.add_axes((0.08, 0.095, 0.58, 0.035))
        self.slider = Slider(
            self.ax_slider,
            "stored frame",
            0,
            max(0, len(sequence) - 1),
            valinit=initial_position,
            valstep=1,
            valfmt="%0.0f",
        )
        if not self._use_idle_draw:
            # Slider.set_val otherwise queues a second idle draw every time
            # the playback timer moves the exact-frame position.
            self.slider.drawon = False
        self.slider.on_changed(self._on_slider)

        self._buttons: list[Any] = []
        for label, x, callback in (
            ("Play", 0.685, self.play),
            ("Pause", 0.745, self.pause),
            ("Restart", 0.805, self.restart),
        ):
            axis = self.fig.add_axes((x, 0.09, 0.055, 0.045))
            button = Button(axis, label)
            button.on_clicked(lambda _event, fn=callback: fn())
            self._buttons.append(button)

        speed_axis = self.fig.add_axes((0.88, 0.045, 0.09, 0.11))
        self._speed_radio = RadioButtons(speed_axis, ("0.5×", "1×", "2×"), active=1)
        self._speed_radio.on_clicked(self._on_speed)
        self.fig.text(0.78, 0.145, "Playback speed", fontsize=8)

        self._timer = self.fig.canvas.new_timer(interval=30)
        self._timer.add_callback(self._on_timer)
        self.update(initial_position)

    def _on_slider(self, value: float) -> None:
        if self._ignore_slider:
            return
        self.controller.seek(int(round(value)))
        self.update(self.controller.position)

    def _on_speed(self, label: str) -> None:
        self.controller.set_speed(float(label.rstrip("×")))

    def _on_timer(self) -> None:
        position = self.controller.tick()
        self.update(position)
        if not self.controller.playing:
            self._timer.stop()
            if self.controller.at_end and self._on_sequence_finished is not None:
                self._on_sequence_finished()

    def _set_slider(self, position: int) -> None:
        if int(round(self.slider.val)) == position:
            return
        self._ignore_slider = True
        try:
            self.slider.set_val(position)
        finally:
            self._ignore_slider = False

    def play(self) -> None:
        self.controller.play()
        self._timer.start()

    def pause(self) -> None:
        self.controller.pause()
        self._timer.stop()
        self.update(self.controller.position)

    def restart(self) -> None:
        self.controller.restart()
        self._timer.stop()
        self._set_slider(0)
        self.update(0)

    @staticmethod
    def _reading_text(reading: SensorReading) -> str:
        if not reading.valid:
            return f"{reading.sensor.sensor_id:<27} INVALID (mask=0)"
        if isinstance(reading.value, tuple):
            value = "[" + ", ".join(f"{item:+.3f}" for item in reading.value) + "]"
        else:
            value = f"{float(reading.value):.4f}"
        return f"{reading.sensor.sensor_id:<27} {value}"

    def _draw_dashboard(self, position: int) -> None:
        frame = self.sequence.frame_at(position)
        self.ax_dashboard.clear()
        self.ax_dashboard.set_axis_off()
        label = self.sequence.label_ar or "(not supplied)"
        if self.sequence.metadata.get("mesh", {}).get("embedded_mano_vertices_available"):
            geometry_label = "embedded MANO vertex cloud + tracked 21-joint landmarks"
        else:
            geometry_label = "tracked 21-joint landmarks"
        header = (
            f"sample: {self.sequence.sample_id}\n"
            f"label: {label}    signer: {self.sequence.signer_id or '(unknown)'}\n"
            f"stored frame: {frame.frame_index}   position: {position + 1}/{len(self.sequence)}\n"
            f"timestamp: {frame.timestamp_seconds:.6f} s\n"
            f"geometry: {geometry_label}\n"
            "surface topology: NOT STORED\n"
            "H = Hall/magnetic angular package   IMU = palm orientation package"
        )
        self.ax_dashboard.text(0.0, 0.995, header, va="top", ha="left", family="monospace", fontsize=7.5)
        for column, track in enumerate(TRACK_ORDER):
            x = 0.005 + column * 0.515
            hand = frame.hand(track)
            state = f"{hand.state} / geometry present" if hand.present else f"{hand.state} / NO GEOMETRY"
            text = [track, state]
            text.extend(self._reading_text(reading) for reading in self.sequence.sensor_readings(position, track))
            self.ax_dashboard.text(
                x,
                0.78,
                "\n".join(text),
                va="top",
                ha="left",
                family="monospace",
                fontsize=5.8,
                color="#1f2937" if hand.present else "#b91c1c",
            )
        self.ax_dashboard.set_xlim(0, 1)
        self.ax_dashboard.set_ylim(0, 1)

    def _draw_hand(self, position: int, track: str) -> None:
        frame = self.sequence.frame_at(position)
        hand = frame.hand(track)
        color = "#2563eb" if track == "LEFT" else "#ea580c"
        if not hand.present:
            self.ax_3d.text2D(
                0.03,
                0.94 if track == "LEFT" else 0.89,
                f"{track}: {hand.state} (not rendered)",
                transform=self.ax_3d.transAxes,
                color="#b91c1c",
                fontsize=9,
            )
            return

        if hand.mesh_vertices is not None:
            mesh = np.asarray(hand.mesh_vertices)
            self.ax_3d.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], s=1.8, alpha=0.18, color=color, depthshade=False)
        if hand.landmarks_3d is not None:
            points = np.asarray(hand.landmarks_3d)
            self.ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], s=16, color=color, depthshade=True)
            for start, end in SKELETON_EDGES:
                self.ax_3d.plot(
                    points[[start, end], 0],
                    points[[start, end], 1],
                    points[[start, end], 2],
                    color=color,
                    linewidth=1.3,
                    alpha=0.82,
                )

            readings = {reading.sensor.sensor_id: reading for reading in self.sequence.sensor_readings(position, track)}
            marker_positions = sensor_marker_positions(points, self.sequence.sensor_layout)
            for sensor in self.sequence.sensor_layout:
                marker = marker_positions[sensor.sensor_id]
                if marker is None:
                    continue
                reading = readings[sensor.sensor_id]
                marker_color = color if reading.valid else "#6b7280"
                marker_style = "o" if reading.valid else "x"
                self.ax_3d.scatter(
                    [marker[0]],
                    [marker[1]],
                    [marker[2]],
                    s=32 if sensor.display_marker == "H" else 58,
                    marker=marker_style,
                    color=marker_color,
                    depthshade=False,
                    linewidths=1.2,
                )
                display = sensor.display_marker if reading.valid else f"{sensor.display_marker}?"
                self.ax_3d.text(marker[0], marker[1], marker[2], display, color=marker_color, fontsize=6)

        self.ax_3d.text2D(
            0.03,
            0.94 if track == "LEFT" else 0.89,
            f"{track}: {hand.state}  raw={hand.raw_detection_index}",
            transform=self.ax_3d.transAxes,
            color=color,
            fontsize=9,
        )

    def _draw_3d(self, position: int) -> None:
        self.ax_3d.clear()
        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")
        lower, upper = self._bounds
        self.ax_3d.set_xlim(float(lower[0]), float(upper[0]))
        self.ax_3d.set_ylim(float(lower[1]), float(upper[1]))
        self.ax_3d.set_zlim(float(lower[2]), float(upper[2]))
        try:
            self.ax_3d.set_box_aspect((1, 1, 1))
        except AttributeError:  # pragma: no cover - old Matplotlib compatibility
            pass
        frame = self.sequence.frame_at(position)
        self.ax_3d.set_title(
            f"TASK-007A stored 3D geometry — frame {frame.frame_index} / {frame.timestamp_seconds:.4f}s"
        )
        for track in TRACK_ORDER:
            self._draw_hand(position, track)

    def update(self, position: int) -> None:
        position = int(position)
        if self.controller.position != position:
            self.controller.seek(position)
        self._set_slider(position)
        self._draw_3d(position)
        self._draw_dashboard(position)
        if self._defer_initial_draw:
            # Embedded Tk canvases need to be allocated before their first
            # synchronous draw. The host draws once after construction;
            # subsequent updates follow the configured draw mode.
            self._defer_initial_draw = False
            return
        if self._use_idle_draw:
            self.fig.canvas.draw_idle()
        else:
            self.fig.canvas.draw()

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(destination, dpi=120)
        return destination

    def show(self) -> None:
        self._plt.show()
