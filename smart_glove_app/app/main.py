"""Primary PySide6/Qt Quick 3D entry point for TASK-007F."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

from visualizer.app.integration import (
    VisualizerIntegrationError,
    run_headless_queue,
    run_headless_recognizer_queue,
)
from visualizer.mapping import Core28Resolver

from smart_glove_app.rendering.mano_topology import ManoTopologyError, load_mano_topology


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets" / "manifests" / "karsl_core28.csv"
DEFAULT_LABELS = PROJECT_ROOT / "datasets" / "manifests" / "karsl_core28_labels.csv"
DEFAULT_CATALOG = PROJECT_ROOT / "visualizer" / "catalog" / "core28_exemplars.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
        help="external TASK-008 run root containing pose/tracking/kinematics/virtual_glove",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="optional explicit TASK-009C deployment.pt or compatible research checkpoint",
    )
    parser.add_argument(
        "--mano-model",
        type=Path,
        default=None,
        help="optional locally licensed MANO_RIGHT.pkl/MANO_LEFT.pkl for triangle topology",
    )
    parser.add_argument("--device", default="auto", help="recognition device: auto, cpu, or cuda")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--text", default="", help="optional Arabic text to queue on startup")
    parser.add_argument(
        "--mode",
        choices=("canonical", "signer01", "signer02", "signer03", "random"),
        default="canonical",
    )
    parser.add_argument("--seed", type=int, default=None, help="explicit seed required for random exemplars")
    parser.add_argument("--speed", type=float, choices=(0.5, 1.0, 2.0), default=1.0)
    parser.add_argument(
        "--no-smooth-rendering",
        action="store_true",
        help="display exact stored source frames without visual interpolation",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="traverse the queue without opening Qt and print JSON metadata",
    )
    parser.add_argument(
        "--smoke-seconds",
        type=float,
        default=None,
        help="close automatically after this many seconds (useful for native GUI smoke tests)",
    )
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="print presentation/runtime metrics when the GUI exits",
    )
    return parser


def _resolver(args: argparse.Namespace) -> Core28Resolver:
    return Core28Resolver(labels_path=args.labels, catalog_path=args.catalog)


def _load_headless_recognizer(args: argparse.Namespace) -> tuple[object | None, str | None]:
    if args.checkpoint is None:
        return None, None
    try:
        from visualizer.recognition import RecognizerAdapter

        device = args.device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        return (
            RecognizerAdapter.from_checkpoint(
                args.checkpoint,
                run_root=args.run_root,
                labels_path=args.labels,
                device=device,
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - mirror the legacy optional-checkpoint behavior
        return None, f"{type(exc).__name__}: {exc}"


def _run_headless(args: argparse.Namespace, resolver: Core28Resolver) -> int:
    adapter, recognition_error = _load_headless_recognizer(args)
    try:
        if args.checkpoint is None:
            result = run_headless_queue(
                args.text,
                run_root=args.run_root,
                resolver=resolver,
                manifest_path=args.manifest,
                mode=args.mode,
                rng_seed=args.seed,
            )
        else:
            result = run_headless_recognizer_queue(
                args.text,
                run_root=args.run_root,
                recognition_adapter=adapter,
                recognition_error=recognition_error,
                resolver=resolver,
                manifest_path=args.manifest,
                mode=args.mode,
                rng_seed=args.seed,
            )
    except (VisualizerIntegrationError, OSError, ValueError) as exc:
        print(f"TASK-007F headless error: {exc}", file=sys.stderr)
        return 2
    _write_utf8_json(asdict(result))
    return 0


def _write_utf8_json(payload: object) -> None:
    """Write Unicode queue/diagnostic output safely on native Windows consoles."""

    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=list) + "\n"
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        buffer.write(rendered.encode("utf-8"))
        buffer.flush()
    else:  # pragma: no cover - StringIO-style test streams
        sys.stdout.write(rendered)


def _qml_error_text(engine: object) -> str:
    warnings = getattr(engine, "warnings", None)
    if callable(warnings):
        return "\n".join(str(error) for error in warnings())
    return "QQmlApplicationEngine created no root object"


def _run_gui(args: argparse.Namespace, resolver: Core28Resolver) -> int:
    try:
        from PySide6.QtCore import QUrl, QTimer
        from PySide6.QtGui import QGuiApplication, QSurfaceFormat
        from PySide6.QtQml import QQmlApplicationEngine
        from PySide6.QtQuick import QQuickWindow
        try:
            from PySide6.QtQuickControls2 import QQuickStyle
        except ImportError:  # pragma: no cover - bundled in standard PySide6 wheels
            QQuickStyle = None
    except ImportError as exc:
        print(
            "PySide6 is required for the primary application. Install it with "
            "python -m pip install -e .[gui]",
            file=sys.stderr,
        )
        print(f"Import detail: {exc}", file=sys.stderr)
        return 2

    try:
        topology = load_mano_topology(args.mano_model)
    except ManoTopologyError as exc:
        print(f"TASK-007F MANO topology error: {exc}", file=sys.stderr)
        return 2

    surface_format = QSurfaceFormat()
    surface_format.setDepthBufferSize(24)
    surface_format.setStencilBufferSize(8)
    surface_format.setSamples(4)
    QSurfaceFormat.setDefaultFormat(surface_format)

    app = QGuiApplication(sys.argv[:1])
    app.setApplicationName("Arabic Smart Glove")
    app.setOrganizationName("Graduation Project Simulation")
    if QQuickStyle is not None:
        QQuickStyle.setStyle("Basic")

    from smart_glove_app.app.application_controller import Core28ApplicationController

    controller = Core28ApplicationController(
        run_root=args.run_root,
        resolver=resolver,
        manifest_path=args.manifest,
        topology=topology,
        checkpoint=args.checkpoint,
        labels_path=args.labels,
        device=args.device,
        initial_text=args.text,
        mode=args.mode,
        rng_seed=args.seed,
        speed=args.speed,
        smooth_rendering=not args.no_smooth_rendering,
    )
    engine = QQmlApplicationEngine()
    engine.setInitialProperties(
        {
            "appState": controller,
            "leftGeometryObject": controller.left_geometry,
            "rightGeometryObject": controller.right_geometry,
            "leftMarkerModel": controller.left_markers,
            "rightMarkerModel": controller.right_markers,
        }
    )
    # Child QML components receive these objects through the engine context;
    # the root ApplicationWindow also receives the same values as initial
    # properties so bindings are established before construction.
    context = engine.rootContext()
    context.setContextProperty("appState", controller)
    context.setContextProperty("leftGeometryObject", controller.left_geometry)
    context.setContextProperty("rightGeometryObject", controller.right_geometry)
    context.setContextProperty("leftMarkerModel", controller.left_markers)
    context.setContextProperty("rightMarkerModel", controller.right_markers)
    qml_path = Path(__file__).resolve().parents[1] / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    if not engine.rootObjects():
        print(f"TASK-007F QML load failed:\n{_qml_error_text(engine)}", file=sys.stderr)
        controller.shutdown()
        return 2

    api = QQuickWindow.graphicsApi()
    api_name = getattr(api, "name", str(api).split(".")[-1])
    controller.setGraphicsApi(f"Qt Quick RHI · {api_name}")
    if args.smoke_seconds is not None:
        if args.smoke_seconds < 0:
            print("--smoke-seconds must be non-negative", file=sys.stderr)
            controller.shutdown()
            return 2
        QTimer.singleShot(int(args.smoke_seconds * 1000), app.quit)
    exit_code = app.exec()
    controller.shutdown()
    if args.print_metrics:
        _write_utf8_json(
            {
                "render_fps": controller.renderFps,
                "active_sequence_fps": controller.activeSequenceFps,
                "graphics_api": controller.graphicsApi,
                "surface_mode": controller.surfaceMode,
                "topology_status": controller.topologyStatus,
                "recognition_status": controller.recognitionStatus,
                "recognition_role": controller.recognitionRole,
                "recognition_reference": controller.recognitionReference,
                "expected_character": controller.expectedCharacter,
                "predicted_character": controller.predictedCharacter,
                "confidence": controller.confidenceText,
                "left_geometry_creation_count": controller.left_geometry.geometry_creation_count,
                "right_geometry_creation_count": controller.right_geometry.geometry_creation_count,
                "left_geometry_update_count": controller.left_geometry.update_count,
                "right_geometry_update_count": controller.right_geometry.update_count,
            }
        )
    return int(exit_code)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        resolver = _resolver(args)
    except (OSError, ValueError) as exc:
        print(f"TASK-007F mapping/catalog error: {exc}", file=sys.stderr)
        return 2
    if args.headless:
        return _run_headless(args, resolver)
    return _run_gui(args, resolver)


if __name__ == "__main__":
    raise SystemExit(main())
