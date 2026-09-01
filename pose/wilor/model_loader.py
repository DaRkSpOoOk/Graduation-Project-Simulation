"""Loads the official WiLoR detector + reconstruction model.

The official repository (https://github.com/rolpotamias/WiLoR) is not a
published PyPI package: it is meant to be cloned and imported from its own
checkout. This loader expects a local clone at ``WILOR_SOURCE_DIR`` (falls
back to ``<assets_dir>/WiLoR``) and inserts it onto ``sys.path``.

Checkpoint/MANO asset resolution happens before any heavy import so that a
missing gated MANO file fails fast with a clear, specific error instead of a
deep stack trace from inside smplx/pickle. See
reports/pose/wilor/TASK-002-wilor-karsl-pilot.md for why MANO_RIGHT.pkl is
not obtainable automatically in this environment.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import WilorAssetPaths, WilorRuntimeConfig
from .errors import ManoAssetMissingError, WilorAssetMissingError, WilorDependencyError


@dataclass(slots=True)
class WilorPipeline:
    model: Any
    detector: Any
    model_cfg: Any
    device: Any
    fast_mode: bool
    checkpoint_paths: WilorAssetPaths


def _resolve_source_dir(assets: WilorAssetPaths) -> Path:
    override = os.environ.get("WILOR_SOURCE_DIR")
    return Path(override).expanduser() if override else assets.assets_dir / "WiLoR"


def _load_yolo_checkpoint(checkpoint_path: str) -> Any:
    """Load the official YOLO hand detector checkpoint.

    Works around a version-skew issue reproduced in this environment:
    ultralytics==8.1.34 (pinned by WiLoR's requirements.txt, released before
    PyTorch 2.6) calls ``torch.load(file, map_location="cpu")`` without
    ``weights_only=False``. Since PyTorch 2.6, ``weights_only`` defaults to
    True, so loading detector.pt raises
    ``UnpicklingError: ... GLOBAL ultralytics.nn.tasks.PoseModel was not an
    allowed global``. detector.pt comes from the official WiLoR HuggingFace
    Space (see config.py DETECTOR_CHECKPOINT_URL) and is trusted, so we
    temporarily restore the pre-2.6 default rather than pin an older torch.
    See reports/pose/wilor/TASK-002-wilor-karsl-pilot.md, Installation.
    """
    import torch  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    original_load = torch.load

    def _load_with_full_unpickling(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _load_with_full_unpickling
    try:
        return YOLO(checkpoint_path)
    finally:
        torch.load = original_load


def check_assets(assets: WilorAssetPaths) -> None:
    """Raise a specific, actionable error for the first missing asset."""
    if not assets.detector_checkpoint.exists():
        raise WilorAssetMissingError(
            assets.detector_checkpoint,
            "Download with:\n"
            f"  wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/"
            f"pretrained_models/detector.pt -P {assets.detector_checkpoint.parent}",
        )
    if not assets.wilor_checkpoint.exists():
        raise WilorAssetMissingError(
            assets.wilor_checkpoint,
            "Download with:\n"
            f"  wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/"
            f"pretrained_models/wilor_final.ckpt -P {assets.wilor_checkpoint.parent}",
        )
    if not assets.model_config.exists():
        raise WilorAssetMissingError(
            assets.model_config,
            "Copy pretrained_models/model_config.yaml from the WiLoR source checkout.",
        )
    if not assets.mano_right_pkl.exists():
        raise ManoAssetMissingError(assets.mano_right_pkl)


def load_pipeline(
    assets: WilorAssetPaths | None = None,
    runtime: WilorRuntimeConfig | None = None,
) -> WilorPipeline:
    """Load the WiLoR model + YOLO hand detector.

    Raises:
        ManoAssetMissingError: MANO_RIGHT.pkl not present (expected in
            environments without manual MANO license acceptance).
        WilorAssetMissingError: any other required checkpoint/config missing.
        WilorDependencyError: the official ``wilor`` source package (or a
            third-party dependency such as ``ultralytics``/``smplx``) is not
            importable.
    """
    assets = assets or WilorAssetPaths.resolve()
    runtime = runtime or WilorRuntimeConfig()

    check_assets(assets)

    source_dir = _resolve_source_dir(assets)
    if not source_dir.is_dir():
        raise WilorDependencyError(
            f"WiLoR source checkout not found at {source_dir}. Clone with:\n"
            f"  git clone --recursive https://github.com/rolpotamias/WiLoR.git {source_dir}"
        )
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    try:
        import torch  # noqa: PLC0415
        from wilor.configs import get_config  # noqa: PLC0415
        from wilor.models import WiLoR  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only with full env
        raise WilorDependencyError(
            "Failed to import WiLoR/torch/ultralytics. Install the pinned "
            "requirements documented in "
            "reports/pose/wilor/TASK-002-wilor-karsl-pilot.md."
        ) from exc

    device_name = runtime.device if torch.cuda.is_available() or runtime.device == "cpu" else "cpu"
    device = torch.device(device_name)

    # Re-implements wilor.models.load_wilor(), but (a) points MANO.DATA_DIR /
    # MODEL_PATH / MEAN_PARAMS at our absolute asset paths instead of the
    # upstream helper's hardcoded "./mano_data/" (cwd-relative, fragile), and
    # (b) passes init_renderer=False so pyrender/OpenGL is never required
    # for raw-output extraction (see pose/wilor/visualize.py docstring).
    model_cfg = get_config(str(assets.model_config), update_cachedir=True)
    if "vit" in model_cfg.MODEL.BACKBONE.TYPE and "BBOX_SHAPE" not in model_cfg.MODEL:
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()
    model_cfg.defrost()
    model_cfg.MANO.DATA_DIR = str(assets.mano_right_pkl.parent) + "/"
    model_cfg.MANO.MODEL_PATH = str(assets.mano_right_pkl.parent) + "/"
    model_cfg.MANO.MEAN_PARAMS = str(assets.mano_right_pkl.parent / "mano_mean_params.npz")
    model_cfg.freeze()

    model = WiLoR.load_from_checkpoint(
        str(assets.wilor_checkpoint), strict=False, cfg=model_cfg, init_renderer=False
    )
    if runtime.fast_mode:
        torch.set_float32_matmul_precision("high")
        model = model.half()
        model.backbone = torch.compile(model.backbone)
        model.backbone.skip_blocks = True

    detector = _load_yolo_checkpoint(str(assets.detector_checkpoint))
    model = model.to(device)
    detector = detector.to(device)
    model.eval()

    return WilorPipeline(
        model=model,
        detector=detector,
        model_cfg=model_cfg,
        device=device,
        fast_mode=runtime.fast_mode,
        checkpoint_paths=assets,
    )


def load_detector_only(
    assets: WilorAssetPaths | None = None,
    device: str = "cuda",
) -> Any:
    """Load only the YOLO hand detector -- no MANO/WiLoR checkpoint needed.

    Used for the detector_only extraction mode (see
    pose/wilor/frame_extraction.py) when MANO assets are unavailable.
    """
    import torch  # noqa: PLC0415

    assets = assets or WilorAssetPaths.resolve()
    if not assets.detector_checkpoint.exists():
        raise WilorAssetMissingError(
            assets.detector_checkpoint,
            "Download with:\n"
            f"  wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/"
            f"pretrained_models/detector.pt -P {assets.detector_checkpoint.parent}",
        )
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    detector = _load_yolo_checkpoint(str(assets.detector_checkpoint))
    return detector.to(device_obj)
