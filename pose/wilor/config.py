"""Configuration and asset-path resolution for the WiLoR extractor.

Values below (image size, focal length, MANO hand-joint count) come from the
official WiLoR release config `pretrained_models/model_config.yaml`
(commit fcb9113, 2026-04-07) at
https://github.com/rolpotamias/WiLoR/blob/main/pretrained_models/model_config.yaml
Documented in reports/pose/wilor/TASK-002-wilor-karsl-pilot.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# From pretrained_models/model_config.yaml: MODEL.IMAGE_SIZE
DEFAULT_IMAGE_SIZE = 256
# From pretrained_models/model_config.yaml: EXTRA.FOCAL_LENGTH
DEFAULT_FOCAL_LENGTH = 5000
# From pretrained_models/model_config.yaml: MANO.NUM_HAND_JOINTS (finger joints, excludes wrist)
DEFAULT_NUM_HAND_JOINTS = 15
# From demo.py default
DEFAULT_RESCALE_FACTOR = 2.0
# From demo.py: detector(img_cv2, conf=0.3, ...)
DEFAULT_DETECTOR_CONFIDENCE = 0.3

WILOR_UPSTREAM_COMMIT = "fcb911312a38fa8badd30d9656a167485d61b8f9"
WILOR_UPSTREAM_REPO = "https://github.com/rolpotamias/WiLoR"

# Official checkpoint sources (unrestricted, no account needed).
DETECTOR_CHECKPOINT_URL = (
    "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/"
    "pretrained_models/detector.pt"
)
WILOR_CHECKPOINT_URL = (
    "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/"
    "pretrained_models/wilor_final.ckpt"
)
# MANO_RIGHT.pkl is NOT auto-downloadable: requires an account at
# https://mano.is.tue.mpg.de and manual acceptance of the MANO license.
MANO_LICENSE_URL = "https://mano.is.tue.mpg.de/license.html"


@dataclass(slots=True)
class WilorAssetPaths:
    """Resolved filesystem locations for WiLoR's external assets.

    All assets live outside the repository (never committed - see
    AGENTS.md "Repository artifact policy"). Defaults to
    ``~/.cache/wilor_assets`` but can be overridden with the
    ``WILOR_ASSETS_DIR`` environment variable.
    """

    assets_dir: Path
    detector_checkpoint: Path
    wilor_checkpoint: Path
    model_config: Path
    mano_right_pkl: Path

    @classmethod
    def resolve(cls, assets_dir: str | Path | None = None) -> "WilorAssetPaths":
        base = Path(assets_dir or os.environ.get("WILOR_ASSETS_DIR", "~/.cache/wilor_assets"))
        base = base.expanduser()
        return cls(
            assets_dir=base,
            detector_checkpoint=base / "pretrained_models" / "detector.pt",
            wilor_checkpoint=base / "pretrained_models" / "wilor_final.ckpt",
            model_config=base / "pretrained_models" / "model_config.yaml",
            mano_right_pkl=base / "mano_data" / "MANO_RIGHT.pkl",
        )


@dataclass(slots=True)
class WilorRuntimeConfig:
    """Runtime knobs for extraction, independent of checkpoint identity."""

    device: str = "cuda"
    fast_mode: bool = False  # official --fast: FP16 + backbone layer-skip/compile
    detector_confidence: float = DEFAULT_DETECTOR_CONFIDENCE
    rescale_factor: float = DEFAULT_RESCALE_FACTOR
    batch_size: int = 1
