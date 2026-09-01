#!/usr/bin/env python3
"""Download the official MediaPipe Hand Landmarker task bundle."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def download(url: str, destination: Path, overwrite: bool = False) -> str:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination exists; pass --overwrite to replace it: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
    temporary.replace(destination)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("datasets/raw/models/hand_landmarker.task"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        checksum = download(args.url, args.output, overwrite=args.overwrite)
    except Exception as error:  # CLI boundary.
        print(f"MediaPipe model download failed: {error}", file=sys.stderr)
        return 1
    print(f"downloaded={args.output}\nsha256={checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
