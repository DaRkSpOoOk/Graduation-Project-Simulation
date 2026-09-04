"""Tk/Matplotlib desktop application for the TASK-007C/007D integration.

The window owns presentation state only.  Character mapping, exemplar
selection, queue state, artifact loading, frame playback, and sequence-level
recognition remain in their respective facing objects.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

from visualizer.keyboard import Core28Keyboard
from visualizer.mapping import Core28Resolver
from visualizer.queue import QueueState, UnsupportedTextError

from .integration import QueuePlaybackSession, VisualizerIntegrationError


MODE_LABELS = {
    "Canonical": "canonical",
    "Signer 01": "signer01",
    "Signer 02": "signer02",
    "Signer 03": "signer03",
    "Seeded random": "random",
}


class Core28VisualizerApplication:
    """Interactive Core-28 text/keyboard to stored-sequence playback app."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        manifest_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        catalog_path: str | Path | None = None,
        initial_text: str = "",
        mode: str = "canonical",
        rng_seed: int | None = None,
        speed: float = 1.0,
        recognition_adapter: Any | None = None,
        recognition_error: str | None = None,
        show_recognition: bool = False,
        root: tk.Tk | None = None,
    ) -> None:
        if mode not in set(MODE_LABELS.values()):
            raise ValueError(f"unsupported exemplar mode: {mode!r}")
        if speed not in {0.5, 1.0, 2.0}:
            raise ValueError("speed must be one of 0.5, 1.0, or 2.0")

        self.root = root or tk.Tk()
        self.root.title("Core-28 Virtual Smart Glove Visualizer")
        self.root.geometry("1500x980")
        self.root.minsize(1050, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        resolver_kwargs: dict[str, Any] = {}
        if labels_path is not None:
            resolver_kwargs["labels_path"] = labels_path
        if catalog_path is not None:
            resolver_kwargs["catalog_path"] = catalog_path
        resolver = Core28Resolver(**resolver_kwargs)
        self.session = QueuePlaybackSession(
            run_root=run_root,
            resolver=resolver,
            manifest_path=manifest_path,
        )
        self.keyboard = Core28Keyboard(resolver.mapping)

        self._viewer: Any | None = None
        self._canvas: Any | None = None
        self._figure: Any | None = None
        self._gap_after_id: str | None = None
        self._poll_after_id: str | None = None
        self._completion_scheduled = False
        self._closed = False
        self._speed = float(speed)
        self._recognition_adapter = recognition_adapter
        self._recognition_error = recognition_error
        self._show_recognition = bool(show_recognition)
        self._recognition_cache: dict[str, Any] = {}
        self._active_recognition: Any | None = None

        self.text_var = tk.StringVar(value=initial_text)
        self.mode_var = tk.StringVar(value=self._mode_to_label(mode))
        self.seed_var = tk.StringVar(value="" if rng_seed is None else str(rng_seed))
        self.speed_var = tk.StringVar(value=f"{speed:g}×")
        self.status_var = tk.StringVar(value="Ready. Type text or press a Core-28 key.")
        self.current_var = tk.StringVar(value="No active sequence")
        self.recognition_status_var = tk.StringVar(value="Recognition disabled")
        self.recognition_expected_var = tk.StringVar(value="—")
        self.recognition_predicted_var = tk.StringVar(value="—")
        self.recognition_confidence_var = tk.StringVar(value="—")
        self.recognition_model_var = tk.StringVar(value="Visualization-only mode")
        self.recognition_top_var = tk.StringVar(value="—")
        self._build_ui()
        self._poll_playback()

        if initial_text:
            # Let Tk realize the complete window before the first 3D canvas
            # is created.  This is especially important for TkAgg, whose
            # initial widget allocation can otherwise be only one pixel.
            self.root.after(100, lambda: self._queue_typed_text(start=True))

    @staticmethod
    def _mode_to_label(mode: str) -> str:
        for label, value in MODE_LABELS.items():
            if value == mode:
                return label
        raise ValueError(f"unsupported exemplar mode: {mode!r}")

    def _selected_mode_and_seed(self) -> tuple[str, int | None]:
        mode = MODE_LABELS[self.mode_var.get()]
        if mode != "random":
            return mode, None
        raw_seed = self.seed_var.get().strip()
        if not raw_seed:
            raise ValueError("Seeded random mode requires an explicit integer seed")
        try:
            return mode, int(raw_seed)
        except ValueError as exc:
            raise ValueError("random exemplar seed must be an integer") from exc

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(
            header,
            text="Core-28 Virtual Smart Glove Visualizer",
            font=("TkDefaultFont", 15, "bold"),
        ).pack(side="left")
        ttk.Label(header, text="WiLoR/MANO vertex cloud + tracked LEFT/RIGHT playback").pack(
            side="right", padx=4
        )

        text_row = ttk.Frame(outer)
        text_row.pack(fill="x", pady=(0, 6))
        ttk.Label(text_row, text="Arabic text:").pack(side="left", padx=(0, 5))
        self.text_entry = ttk.Entry(text_row, textvariable=self.text_var, justify="right")
        self.text_entry.pack(side="left", fill="x", expand=True)
        self.text_entry.bind("<Return>", lambda _event: self._queue_typed_text(start=True))
        ttk.Button(text_row, text="Queue typed text", command=lambda: self._queue_typed_text(start=True)).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(text_row, text="Clear queue", command=self._clear_queue).pack(side="left", padx=(6, 0))

        options = ttk.Frame(outer)
        options.pack(fill="x", pady=(0, 6))
        ttk.Label(options, text="Exemplar:").pack(side="left")
        mode_box = ttk.Combobox(
            options,
            textvariable=self.mode_var,
            values=tuple(MODE_LABELS),
            state="readonly",
            width=16,
        )
        mode_box.pack(side="left", padx=(5, 12))
        mode_box.bind("<<ComboboxSelected>>", self._mode_changed)
        ttk.Label(options, text="Seed (random only):").pack(side="left")
        ttk.Entry(options, textvariable=self.seed_var, width=10).pack(side="left", padx=(5, 12))
        ttk.Label(options, text="App speed:").pack(side="left")
        speed_box = ttk.Combobox(
            options,
            textvariable=self.speed_var,
            values=("0.5×", "1×", "2×"),
            state="readonly",
            width=6,
        )
        speed_box.pack(side="left", padx=5)
        speed_box.bind("<<ComboboxSelected>>", self._speed_changed)
        ttk.Label(options, textvariable=self.current_var).pack(side="right", padx=5)

        keyboard_frame = ttk.LabelFrame(outer, text="Arabic Core-28 keyboard (logical order is preserved)")
        keyboard_frame.pack(fill="x", pady=(0, 6))
        for row_index, row in enumerate(self.keyboard.rtl_rows):
            row_frame = ttk.Frame(keyboard_frame)
            row_frame.pack(fill="x", anchor="e")
            for display_index, key in enumerate(row):
                # Core28Keyboard returns each visual row in RTL order.  Grid
                # columns are mirrored so the first item appears at the
                # right, while the callback still receives the exact Unicode
                # code point from the authoritative mapping.
                column = len(row) - display_index - 1
                button = ttk.Button(
                    row_frame,
                    text=key.character,
                    width=4,
                    command=lambda character=key.character: self._enqueue_key(character),
                )
                button.grid(row=0, column=column, padx=2, pady=2)
            row_frame.grid_columnconfigure(0, weight=1)
            row_frame.grid_columnconfigure(len(row) + 1, weight=1)

        split = ttk.PanedWindow(outer, orient="horizontal")
        split.pack(fill="both", expand=True, pady=(0, 6))
        self.viewer_host = ttk.Frame(split, relief="sunken", borderwidth=1)
        queue_panel = ttk.Frame(split, padding=(8, 0, 0, 0))
        split.add(self.viewer_host, weight=5)
        split.add(queue_panel, weight=2)

        ttk.Label(queue_panel, text="Current queue", font=("TkDefaultFont", 11, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        list_frame = ttk.Frame(queue_panel)
        list_frame.pack(fill="both", expand=True)
        self.queue_list = tk.Listbox(
            list_frame,
            activestyle="none",
            exportselection=False,
            font=("TkFixedFont", 10),
            height=20,
        )
        self.queue_list.pack(side="left", fill="both", expand=True)
        queue_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.queue_list.yview)
        queue_scroll.pack(side="right", fill="y")
        self.queue_list.configure(yscrollcommand=queue_scroll.set)

        if self._show_recognition:
            recognition_frame = ttk.LabelFrame(queue_panel, text="Recognition (demo only)")
            recognition_frame.pack(fill="x", pady=(8, 0))
            recognition_rows = (
                ("Status", self.recognition_status_var),
                ("Expected", self.recognition_expected_var),
                ("Predicted", self.recognition_predicted_var),
                ("Confidence", self.recognition_confidence_var),
                ("Model", self.recognition_model_var),
                ("Top 3", self.recognition_top_var),
            )
            for row_index, (label, variable) in enumerate(recognition_rows):
                ttk.Label(recognition_frame, text=f"{label}:").grid(
                    row=row_index, column=0, sticky="nw", padx=(5, 5), pady=2
                )
                value_label = ttk.Label(
                    recognition_frame,
                    textvariable=variable,
                    justify="left",
                    wraplength=300,
                )
                value_label.grid(row=row_index, column=1, sticky="nw", padx=(0, 5), pady=2)
                if label == "Predicted":
                    # The actual prediction is never altered to match the
                    # expectation; color only makes disagreement obvious.
                    self._recognition_prediction_label = value_label
        else:
            self._recognition_prediction_label = None

        controls = ttk.Frame(queue_panel)
        controls.pack(fill="x", pady=(8, 0))
        for label, callback in (
            ("Start", self._start),
            ("Pause", self._pause),
            ("Next", self._next),
            ("Restart current", self._restart_current),
            ("Restart queue", self._restart_queue),
        ):
            ttk.Button(controls, text=label, command=callback).pack(fill="x", pady=2)

        status = ttk.Label(outer, textvariable=self.status_var, anchor="w", relief="sunken", padding=4)
        status.pack(fill="x")

    def _mode_changed(self, _event: object = None) -> None:
        try:
            mode, _ = self._selected_mode_and_seed()
            self.status_var.set(f"Future queue entries use {mode}; existing entries were not re-resolved.")
        except ValueError as exc:
            self.status_var.set(str(exc))

    def _speed_changed(self, _event: object = None) -> None:
        self._speed = float(self.speed_var.get().rstrip("×"))
        if self._viewer is not None:
            self._viewer.controller.set_speed(self._speed)
        self.status_var.set(f"Playback speed set to {self._speed:g}×.")

    def _queue_typed_text(self, *, start: bool) -> None:
        text = self.text_var.get()
        if not text:
            self.status_var.set("Enter at least one Core-28 character.")
            return
        try:
            mode, seed = self._selected_mode_and_seed()
            self.session.enqueue_text(text, mode=mode, rng_seed=seed)
        except (UnsupportedTextError, ValueError) as exc:
            self.status_var.set(f"Nothing queued: {exc}")
            return
        self._refresh_queue()
        self.status_var.set(f"Queued {len(text)} logical input characters; repeated letters remain distinct.")
        if start and self.session.queue.current is not None and self._viewer is None:
            self._start()

    def _enqueue_key(self, character: str) -> None:
        try:
            mode, seed = self._selected_mode_and_seed()
            self.session.enqueue_character(character, mode=mode, rng_seed=seed)
        except ValueError as exc:
            self.status_var.set(f"Key not queued: {exc}")
            return
        self._refresh_queue()
        self.status_var.set(f"Queued {character}; queue order and repetitions are preserved.")
        if self._viewer is None:
            self._start()

    def _refresh_queue(self) -> None:
        self.queue_list.delete(0, tk.END)
        items = self.session.queue.items
        current = self.session.queue.current
        for index, item in enumerate(items):
            if item.item_type == "gap":
                detail = "NEUTRAL GAP (no sequence)"
            else:
                detail = f"SignID {item.sign_id} / {item.sample_id}"
            line = f"{index + 1:02d}  {item.character}  {item.state.value:<9}  {detail}"
            self.queue_list.insert(tk.END, line)
            if item is current:
                self.queue_list.itemconfig(index, background="#dbeafe", foreground="#111827")
            elif item.state == QueueState.COMPLETED:
                self.queue_list.itemconfig(index, foreground="#166534")
            elif item.state in {QueueState.FAILED, QueueState.UNAVAILABLE}:
                self.queue_list.itemconfig(index, foreground="#b91c1c")

    def _set_recognition_inactive(self, *, reason: str, expected: str = "—") -> None:
        """Set presentation state without fabricating a prediction."""

        self._active_recognition = None
        self.recognition_status_var.set(reason)
        self.recognition_expected_var.set(expected)
        self.recognition_predicted_var.set("—")
        self.recognition_confidence_var.set("—")
        self.recognition_top_var.set("—")
        if self._recognition_adapter is not None:
            metadata = self._recognition_adapter.metadata
            self.recognition_model_var.set(metadata.warning)
        elif self._recognition_error:
            self.recognition_model_var.set("Checkpoint unavailable")
        else:
            self.recognition_model_var.set("Visualization-only mode")
        if self._recognition_prediction_label is not None:
            self._recognition_prediction_label.configure(foreground="#111827")

    def _update_recognition(self, item: Any) -> None:
        """Compute/display one cached sequence prediction for the active sign."""

        if item.item_type == "gap":
            self._set_recognition_inactive(reason="Skipped for neutral gap", expected="(gap)")
            return
        expected = str(item.character or "—")
        if self._recognition_adapter is None:
            reason = self._recognition_error or "No checkpoint selected"
            self._set_recognition_inactive(reason="Recognition unavailable", expected=expected)
            self.recognition_top_var.set(reason)
            return

        sample_id = str(item.sample_id or "")
        try:
            if sample_id in self._recognition_cache:
                result = self._recognition_cache[sample_id]
            else:
                result = self._recognition_adapter.predict_queue_item(item)
                self._recognition_cache[sample_id] = result
            if result is None:
                self._set_recognition_inactive(reason="Skipped for neutral gap", expected="(gap)")
                return
            self._active_recognition = result
            self.recognition_expected_var.set(expected)
            metadata = result.checkpoint_metadata
            self.recognition_model_var.set(metadata.warning if metadata is not None else "Demo checkpoint")
            if not result.available:
                self.recognition_status_var.set("Recognition unavailable")
                self.recognition_predicted_var.set("—")
                self.recognition_confidence_var.set("—")
                self.recognition_top_var.set(result.error or "model error")
                if self._recognition_prediction_label is not None:
                    self._recognition_prediction_label.configure(foreground="#b91c1c")
                return
            self.recognition_status_var.set("Sequence prediction ready (demo only)")
            predicted = result.predicted_character or "—"
            if result.predicted_sign_id:
                predicted = f"{predicted} / SignID {result.predicted_sign_id}"
            self.recognition_predicted_var.set(predicted)
            self.recognition_confidence_var.set(
                f"{float(result.confidence):.1%} (max softmax probability)"
            )
            self.recognition_top_var.set(
                " | ".join(
                    f"{entry['character']} {float(entry['probability']):.1%}"
                    for entry in result.top_k
                )
            )
            if self._recognition_prediction_label is not None:
                color = "#166534" if result.predicted_character == item.character else "#b91c1c"
                self._recognition_prediction_label.configure(foreground=color)
        except Exception as exc:  # noqa: BLE001 - inference must not corrupt queue playback
            self._set_recognition_inactive(reason="Recognition unavailable", expected=expected)
            self.recognition_top_var.set(f"{type(exc).__name__}: {exc}")

    def _destroy_viewer(self) -> None:
        if self._viewer is not None:
            # Avoid ``MatplotlibGloveViewer.pause()`` here: it requests an
            # idle redraw, and closing the Tk canvas immediately afterwards
            # can leave that callback racing widget destruction.  Stopping
            # the timer and controller is sufficient for disposal.
            self._viewer.controller.pause()
            self._viewer._timer.stop()
        self._viewer = None
        self._figure = None
        if self._canvas is not None:
            idle_draw_id = getattr(self._canvas, "_idle_draw_id", None)
            if idle_draw_id is not None:
                try:
                    self.root.after_cancel(idle_draw_id)
                except tk.TclError:
                    pass
                self._canvas._idle_draw_id = None
            self._canvas.get_tk_widget().destroy()
            self._canvas = None

    def _show_sequence(self, sequence: Any, *, autoplay: bool) -> None:
        self._cancel_gap()
        self._destroy_viewer()
        for child in self.viewer_host.winfo_children():
            child.destroy()
        try:
            import matplotlib

            matplotlib.use("TkAgg", force=True)
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            from visualizer.rendering import MatplotlibGloveViewer

            # Keep the initial requested size below the usual queue-panel
            # viewport.  Tk/PanedWindow otherwise briefly lays out the
            # canvas at its 1300px figure request before shrinking it.
            figure = Figure(figsize=(9, 6), dpi=100)
            # Attach the Tk canvas before constructing the viewer so its
            # Matplotlib timer is a Tk timer and advances inside mainloop.
            canvas = FigureCanvasTkAgg(figure, master=self.viewer_host)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            # A newly created Tk canvas reports a 1x1 allocation until idle
            # geometry management runs.  Let it realize its real size before
            # the viewer schedules its first redraw; this avoids a FreeType
            # raster-overflow on long dashboard labels.
            self.root.update_idletasks()
            viewer = MatplotlibGloveViewer(
                sequence,
                figure=figure,
                speed=self._speed,
                on_sequence_finished=self._on_sequence_finished,
                defer_initial_draw=True,
                use_idle_draw=False,
            )
            # Slider construction can queue one idle draw before the viewer
            # has been configured for synchronous embedded-canvas rendering.
            # Cancel that stale callback before realizing the widget; otherwise
            # TkAgg may render the still-unallocated canvas concurrently with
            # the first explicit draw.
            idle_draw_id = getattr(canvas, "_idle_draw_id", None)
            if idle_draw_id is not None:
                try:
                    self.root.after_cancel(idle_draw_id)
                except tk.TclError:
                    pass
                canvas._idle_draw_id = None
            # Matplotlib widgets can schedule an idle redraw while they are
            # being created.  Drain that callback before the first explicit
            # draw so TkAgg never has two pending renders for this canvas.
            self.root.update_idletasks()
            canvas.draw()
        except (ImportError, RuntimeError, ValueError) as exc:
            raise VisualizerIntegrationError(f"cannot create the Matplotlib/Tk viewer: {exc}") from exc
        self._figure = figure
        self._canvas = canvas
        self._viewer = viewer
        if autoplay:
            viewer.play()

    def _show_gap(self, item: Any, *, autoplay: bool) -> None:
        self._destroy_viewer()
        self._cancel_gap()
        for child in self.viewer_host.winfo_children():
            child.destroy()
        message = ttk.Label(
            self.viewer_host,
            text=(
                "NEUTRAL GAP\n\n"
                "No geometry or sensor frame is generated for whitespace/punctuation.\n"
                f"Duration: {item.gap_after_ms} ms"
            ),
            anchor="center",
            justify="center",
        )
        message.pack(fill="both", expand=True)
        if autoplay:
            delay = max(1, int(item.gap_after_ms or 0))
            self._gap_after_id = self.root.after(delay, self._finish_gap)

    def _cancel_gap(self) -> None:
        if self._gap_after_id is not None:
            try:
                self.root.after_cancel(self._gap_after_id)
            except tk.TclError:
                pass
            self._gap_after_id = None

    def _activate_current(self, *, autoplay: bool = True) -> None:
        item = self.session.queue.current
        if item is None:
            self._refresh_queue()
            self.status_var.set("Queue complete.")
            self._set_recognition_inactive(reason="Queue complete")
            return
        try:
            if self.session.current_item is not item:
                self.session.start()
            sequence = self.session.current_sequence
            if item.item_type == "gap":
                self._show_gap(item, autoplay=autoplay)
                self._update_recognition(item)
                self.status_var.set("Neutral gap: presentation pause; no sample was loaded.")
            elif sequence is None:
                raise VisualizerIntegrationError("sign item loaded without a PlaybackSequence")
            else:
                self._show_sequence(sequence, autoplay=autoplay)
                self._update_recognition(item)
                self.status_var.set(f"Playing {item.character} / SignID {item.sign_id} / {item.sample_id}")
        except (VisualizerIntegrationError, OSError, ValueError) as exc:
            failed = self.session.queue.fail(str(exc), unavailable=True)
            if failed is not None:
                self.session.current_item = failed
                self.session.current_sequence = None
                self._set_recognition_inactive(
                    reason="Recognition unavailable",
                    expected=str(failed.character or "—"),
                )
                self.recognition_top_var.set(f"sequence unavailable: {exc}")
            self._refresh_queue()
            self.status_var.set(f"Unavailable: {exc}")
            if self.session.queue.current is not None:
                self.root.after(0, lambda: self._activate_current(autoplay=autoplay))
            return
        self._refresh_queue()

    def _start(self) -> None:
        self._cancel_gap()
        if self.session.queue.current is None:
            self.status_var.set("Queue is empty.")
            return
        try:
            self.session.start()
        except (VisualizerIntegrationError, OSError, ValueError) as exc:
            self.status_var.set(f"Cannot start queue: {exc}")
            return
        if self._viewer is not None and self.session.current_sequence is not None:
            self._viewer.play()
            self.status_var.set("Playback resumed.")
        else:
            self._activate_current(autoplay=True)

    def _pause(self) -> None:
        self._cancel_gap()
        if self._viewer is not None:
            self._viewer.pause()
            self.status_var.set("Playback paused.")
        elif self.session.queue.current is not None:
            self.status_var.set("Gap paused; press Start to continue the neutral transition.")
        self._refresh_queue()

    def _advance(self) -> None:
        self._cancel_gap()
        self._destroy_viewer()
        next_item = self.session.complete_current()
        self._refresh_queue()
        if next_item is None:
            self.status_var.set("Queue complete.")
            return
        self._activate_current(autoplay=True)

    def _next(self) -> None:
        if self.session.queue.current is None:
            self.status_var.set("Queue is complete or empty.")
            return
        self._advance()

    def _on_sequence_finished(self) -> None:
        if self._completion_scheduled or self._closed:
            return
        self._completion_scheduled = True
        self.root.after(0, self._advance_finished_sequence)

    def _advance_finished_sequence(self) -> None:
        self._completion_scheduled = False
        if self._closed:
            return
        self._advance()

    def _finish_gap(self) -> None:
        self._gap_after_id = None
        if self._closed:
            return
        self._advance()

    def _restart_current(self) -> None:
        item = self.session.queue.current
        if item is None:
            self.status_var.set("Queue is complete or empty.")
            return
        if item.item_type == "gap":
            self._show_gap(item, autoplay=True)
            return
        if self._viewer is None:
            self._activate_current(autoplay=True)
        else:
            self._viewer.restart()
            self._viewer.play()
            self.status_var.set("Current sequence restarted.")

    def _restart_queue(self) -> None:
        self._cancel_gap()
        self._destroy_viewer()
        self.session.reset()
        self._refresh_queue()
        if self.session.queue.current is not None:
            self._activate_current(autoplay=True)
        else:
            self.status_var.set("Queue reset; it is empty.")

    def _clear_queue(self) -> None:
        self._cancel_gap()
        self._destroy_viewer()
        self.session.clear()
        for child in self.viewer_host.winfo_children():
            child.destroy()
        self._refresh_queue()
        self.current_var.set("No active sequence")
        self._set_recognition_inactive(reason="Recognition idle")
        self.status_var.set("Queue cleared.")

    def _poll_playback(self) -> None:
        if self._closed:
            return
        sequence = self.session.current_sequence
        if self._viewer is not None and sequence is not None:
            frame = sequence.frame_at(self._viewer.controller.position)
            item = self.session.queue.current
            self.current_var.set(
                f"Current: {item.character if item else '-'} / {sequence.sample_id} / "
                f"frame {frame.frame_index} / {frame.timestamp_seconds:.4f}s"
            )
        elif self.session.queue.current is None:
            self.current_var.set("Queue complete")
        self._poll_after_id = self.root.after(100, self._poll_playback)

    def run(self) -> None:
        """Enter Tk's event loop."""

        self.root.mainloop()

    def close(self) -> None:
        self._closed = True
        self._cancel_gap()
        if self._poll_after_id is not None:
            try:
                self.root.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        self._destroy_viewer()
        # Quit first so no new Tk callbacks are dispatched while the
        # Matplotlib-owned child commands are being torn down.  Some Tk
        # builds can report an already-deleted widget command during the
        # recursive destroy; the process/window is still safely closed in
        # that case.
        self.root.quit()
        try:
            self.root.destroy()
        except tk.TclError:
            pass


__all__ = ["Core28VisualizerApplication", "MODE_LABELS"]
