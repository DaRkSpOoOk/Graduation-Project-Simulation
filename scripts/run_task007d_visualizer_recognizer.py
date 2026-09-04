#!/usr/bin/env python3
"""Run TASK-007C playback with optional TASK-009B demo inference.

The checkpoint is intentionally optional.  Without one this entry point has
the same visualization-only behavior as TASK-007C.  With one, the selected
checkpoint is loaded once and receives the exact stored virtual-glove sequence
for each queued sign; it is never run frame by frame and no demo accuracy is
calculated.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from visualizer.app.integration import (
    VisualizerIntegrationError,
    run_headless_queue,
    run_headless_recognizer_queue,
)
from visualizer.mapping import Core28Resolver


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets" / "manifests" / "karsl_core28.csv"
DEFAULT_LABELS = PROJECT_ROOT / "datasets" / "manifests" / "karsl_core28_labels.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="optional explicit TASK-009B checkpoint for demo inference")
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device for the selected checkpoint (default: cpu)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--text", default="", help="optional Arabic text to queue on startup")
    parser.add_argument(
        "--mode",
        choices=("canonical", "signer01", "signer02", "signer03", "random"),
        default="canonical",
    )
    parser.add_argument("--seed", type=int, default=None, help="required for --mode random")
    parser.add_argument("--speed", type=float, choices=(0.5, 1.0, 2.0), default=1.0)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="traverse the queue and print sequence-level demo predictions without opening a window",
    )
    return parser


def _resolver(args: argparse.Namespace) -> Core28Resolver:
    kwargs = {"labels_path": args.labels}
    if args.catalog is not None:
        kwargs["catalog_path"] = args.catalog
    return Core28Resolver(**kwargs)


def _load_recognizer(args: argparse.Namespace):
    if args.checkpoint is None:
        return None, None
    # Import lazily so visualizer-only mode retains TASK-007C's lighter
    # dependency behavior and does not require PyTorch just to open the UI.
    try:
        from visualizer.recognition import RecognizerAdapter

        adapter = RecognizerAdapter.from_checkpoint(
            args.checkpoint,
            run_root=args.run_root,
            labels_path=args.labels,
            device=args.device,
        )
    except Exception as exc:  # noqa: BLE001 - selected-checkpoint failures degrade to visualization-only mode
        return None, f"{type(exc).__name__}: {exc}"
    return adapter, None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        resolver = _resolver(args)
    except (OSError, ValueError) as exc:
        print(f"TASK-007D mapping/catalog error: {exc}", file=sys.stderr)
        return 2

    adapter, recognition_error = _load_recognizer(args)
    if recognition_error:
        print(
            "TASK-007D: checkpoint rejected; continuing in visualization-only mode:\n"
            f"  {recognition_error}",
            file=sys.stderr,
        )

    if args.headless:
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
            print(f"TASK-007D headless error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=list))
        return 0

    from visualizer.app.main_window import Core28VisualizerApplication

    application = Core28VisualizerApplication(
        run_root=args.run_root,
        manifest_path=args.manifest,
        labels_path=args.labels,
        catalog_path=args.catalog,
        initial_text=args.text,
        mode=args.mode,
        rng_seed=args.seed,
        speed=args.speed,
        recognition_adapter=adapter,
        recognition_error=recognition_error,
        show_recognition=True,
    )
    application.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
