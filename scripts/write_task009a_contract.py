#!/usr/bin/env python3
"""Emit the frozen TASK-009A sequence-input contract as machine-readable JSON.

A test asserts the committed file equals this module's output, so the config and
the code cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.data.contract import contract_document  # noqa: E402

DEFAULT_OUTPUT = ROOT / "configs/recognition/task009a_sequence_input_v1.json"


def render() -> str:
    return json.dumps(contract_document(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
