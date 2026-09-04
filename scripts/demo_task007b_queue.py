#!/usr/bin/env python3
"""Demonstrate Core-28 text resolution and ordered playback queue semantics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualizer.mapping import Core28Resolver  # noqa: E402
from visualizer.queue import PlaybackQueue, UnsupportedTextError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="محمد", help="Core-28 Arabic text to enqueue")
    parser.add_argument("--mode", default="canonical", help="canonical, signerNN, or random")
    parser.add_argument("--seed", type=int, default=None, help="required for random mode")
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "datasets/manifests/karsl_core28_labels.csv",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "visualizer/catalog/core28_exemplars.json",
    )
    parser.add_argument(
        "--no-contract-demo",
        action="store_true",
        help="only print the requested queue, without the all-letters/unsupported demonstrations",
    )
    args = parser.parse_args()

    try:
        resolver = Core28Resolver(labels_path=args.labels, catalog_path=args.catalog)
        queue = PlaybackQueue(resolver)
        queue.enqueue_text(args.text, mode=args.mode, rng_seed=args.seed)
    except UnsupportedTextError as error:
        print(f"Unsupported text: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"Cannot build Core-28 queue: {error}", file=sys.stderr)
        return 2

    print(f"Input: {args.text}")
    print("Queue:")
    for index, item in enumerate(queue.items, start=1):
        if item.is_sign:
            print(f"{index}. {item.character} -> {item.sign_id} -> {item.sample_id}")
        else:
            print(f"{index}. {item.character!r} -> neutral gap ({item.gap_after_ms} ms)")

    if not args.no_contract_demo:
        print("\nAll 28 canonical exemplars:")
        for label in resolver.mapping.labels:
            result = resolver.resolve_character(label.character)
            print(f"{label.character} -> {result.sign_id} -> {result.sample_id}")
        print("\nUnsupported-character demonstration:")
        try:
            resolver.resolve_character("أ")
        except ValueError as error:
            print(error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
