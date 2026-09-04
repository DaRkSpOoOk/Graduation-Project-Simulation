#!/usr/bin/env python3
"""Run the integrated Core-28 keyboard/queue/3D virtual-glove application."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from visualizer.app.integration import run_headless_queue
from visualizer.mapping import Core28Resolver


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets" / "manifests" / "karsl_core28.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=None)
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
        help="validate and traverse the queue without opening a desktop window",
    )
    return parser


def _resolver(args: argparse.Namespace) -> Core28Resolver | None:
    kwargs = {}
    if args.labels is not None:
        kwargs["labels_path"] = args.labels
    if args.catalog is not None:
        kwargs["catalog_path"] = args.catalog
    return Core28Resolver(**kwargs) if kwargs else None


def main() -> int:
    args = build_parser().parse_args()
    resolver = _resolver(args)
    if args.headless:
        result = run_headless_queue(
            args.text,
            run_root=args.run_root,
            resolver=resolver,
            manifest_path=args.manifest,
            mode=args.mode,
            rng_seed=args.seed,
        )
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
    )
    application.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
