#!/usr/bin/env python3
"""Build the deterministic TASK-007B Core-28 visualizer catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualizer.catalog.builder import CatalogBuildError, build_catalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "datasets/manifests/karsl_core28_virtual_glove.csv",
        help="authoritative TASK-008 virtual-glove manifest",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "datasets/manifests/karsl_core28_labels.csv",
        help="authoritative Core-28 labels manifest",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="external TASK-008 production run root; it is read only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "visualizer/catalog/core28_exemplars.json",
        help="deterministic catalog JSON output",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "visualizer/catalog/core28_exemplars.csv",
        help="compact canonical-only CSV companion",
    )
    args = parser.parse_args()
    try:
        payload = build_catalog(
            manifest_path=args.manifest,
            run_root=args.run_root,
            labels_path=args.labels,
            output_path=args.output,
            output_csv_path=args.output_csv,
        )
    except (CatalogBuildError, OSError, ValueError) as error:
        parser.error(str(error))
    source = payload["source"]
    print(f"catalog={args.output}")
    print(f"catalog_csv={args.output_csv}")
    print(f"classes={len(payload['entries'])}")
    print(f"rows_audited={source['rows_audited']}")
    print(f"candidates_accepted={source['candidates_accepted']}")
    print(f"candidates_rejected={source['candidates_rejected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
