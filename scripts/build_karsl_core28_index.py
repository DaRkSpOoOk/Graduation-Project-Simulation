#!/usr/bin/env python3
"""Build the compact portable index for external TASK-008A outputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset.manifest import load_manifest, manifest_sha256  # noqa: E402
from evaluation.dataset.orchestrator import write_index  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/karsl_core28.csv")
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("TASK008A_RUN_ROOT", "runs/task008a-karsl-core28")))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets/manifests/karsl_core28_virtual_glove.csv",
    )
    args = parser.parse_args(argv)
    rows = load_manifest(args.manifest)
    write_index(
        args.output,
        rows,
        args.run_root,
        manifest_hash_value=manifest_sha256(args.manifest),
    )
    print(f"Wrote {args.output} ({len(rows)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
