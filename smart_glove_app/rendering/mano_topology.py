"""MANO topology loading for the TASK-007F presentation renderer.

The TASK-008 artifacts intentionally contain MANO vertices but not triangle
indices.  This module accepts a user-supplied, locally licensed MANO asset and
extracts its authoritative topology without generating or fitting any pose.
The normal application path tries the established :mod:`smplx` loader first;
the narrow pickle-field fallback exists so the UI can still run in a clean
visualizer environment where smplx is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


MANO_VERTEX_COUNT = 778
MANO_FACE_COUNT = 1538


class ManoTopologyError(ValueError):
    """Raised when a supplied MANO topology cannot serve the renderer."""


class ManoAssetUnavailableError(ManoTopologyError):
    """Raised when an explicitly requested local MANO asset is unavailable."""


def _readonly(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array)
    value.setflags(write=False)
    return value


def validate_mano_faces(
    faces: Any,
    *,
    expected_vertex_count: int = MANO_VERTEX_COUNT,
    expected_face_count: int | None = MANO_FACE_COUNT,
    require_full_vertex_coverage: bool = True,
) -> np.ndarray:
    """Validate and return a read-only triangle index array.

    ``expected_face_count`` is configurable for small in-memory test meshes,
    while the file loader keeps the real MANO count strict by default.  Full
    vertex coverage catches accidental use of a partial topology with an
    otherwise plausible ``(N, 3)`` shape.
    """

    try:
        array = np.asarray(faces)
    except Exception as exc:  # noqa: BLE001 - normalize arbitrary asset errors
        raise ManoTopologyError("MANO faces could not be converted to an array") from exc
    if array.ndim != 2 or array.shape[1] != 3:
        raise ManoTopologyError(f"MANO faces must have shape (F, 3), got {array.shape}")
    if array.shape[0] <= 0:
        raise ManoTopologyError("MANO topology must contain at least one triangle")
    if expected_face_count is not None and array.shape[0] != int(expected_face_count):
        raise ManoTopologyError(
            f"MANO topology must contain {expected_face_count} faces, got {array.shape[0]}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ManoTopologyError(f"MANO face indices must be integral, got {array.dtype}")
    array = np.asarray(array, dtype=np.int64)
    if np.any(array < 0) or np.any(array >= int(expected_vertex_count)):
        lower = int(array.min())
        upper = int(array.max())
        raise ManoTopologyError(
            "MANO face index out of range: "
            f"min={lower}, max={upper}, expected [0, {expected_vertex_count - 1}]"
        )
    if np.any(array[:, 0] == array[:, 1]) or np.any(array[:, 0] == array[:, 2]) or np.any(
        array[:, 1] == array[:, 2]
    ):
        raise ManoTopologyError("MANO topology contains degenerate triangles")
    if require_full_vertex_coverage:
        used = np.unique(array)
        expected = np.arange(int(expected_vertex_count), dtype=np.int64)
        if not np.array_equal(used, expected):
            missing = np.setdiff1d(expected, used)
            raise ManoTopologyError(
                f"MANO topology does not cover all {expected_vertex_count} vertices; "
                f"missing {missing[:8].tolist()}"
            )
    return _readonly(array.astype(np.uint32, copy=False))


def _validate_neutral_vertices(vertices: Any, expected_vertex_count: int) -> np.ndarray:
    array = np.asarray(vertices, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (expected_vertex_count, 3):
        raise ManoTopologyError(
            "MANO neutral template must have shape "
            f"({expected_vertex_count}, 3), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ManoTopologyError("MANO neutral template contains non-finite values")
    return _readonly(array)


def _asset_hand(path: Path) -> str:
    token = path.name.upper()
    if re.search(r"(?:^|[_-])LEFT(?:[_-]|\.)", token):
        return "LEFT"
    return "RIGHT"


def _load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            # Official MANO files are Python pickles and some distributions
            # were produced under Python 2.  latin1 keeps those byte strings
            # readable without changing numeric arrays.
            return pickle.load(handle, encoding="latin1")
    except Exception as exc:  # noqa: BLE001 - normalize arbitrary local-asset failures
        raise ManoTopologyError(f"could not read MANO asset: {path}") from exc


def _field(payload: Any, names: tuple[str, ...]) -> Any | None:
    if isinstance(payload, Mapping):
        for name in names:
            if name in payload:
                return payload[name]
    for name in names:
        if hasattr(payload, name):
            return getattr(payload, name)
    return None


def _smplx_model(path: Path) -> Any | None:
    """Return an smplx MANO object when the optional loader is usable."""

    try:
        import smplx  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        source_hand = _asset_hand(path)
        return smplx.create(
            model_path=str(path.parent),
            model_type="mano",
            is_rhand=source_hand != "LEFT",
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
        )
    except Exception:  # noqa: BLE001 - dependency fallback is intentional
        return None


def _faces_from_smplx(path: Path) -> tuple[Any | None, Any | None]:
    model = _smplx_model(path)
    if model is None:
        return None, None
    return getattr(model, "faces", None), getattr(model, "v_template", None)


def _canonicalize_right(
    faces: np.ndarray,
    neutral_vertices: np.ndarray | None,
    *,
    source_hand: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    if source_hand != "LEFT":
        return faces, neutral_vertices
    # Official MANO_LEFT is the x-reflected right template and has reversed
    # face winding.  Canonicalize to right-hand data once, then apply the
    # display-track winding in ManoTopology.faces_for_track().
    canonical_faces = np.ascontiguousarray(faces[:, [0, 2, 1]])
    if neutral_vertices is None:
        return _readonly(canonical_faces), None
    canonical_vertices = np.array(neutral_vertices, dtype=np.float32, copy=True)
    canonical_vertices[:, 0] *= -1.0
    return _readonly(canonical_faces), _readonly(canonical_vertices)


@dataclass(frozen=True, slots=True)
class ManoTopology:
    """Validated right-hand MANO topology plus optional neutral vertices."""

    faces: np.ndarray
    source_path: Path
    source_sha256: str
    source_hand: str
    neutral_vertices: np.ndarray | None
    topology_sha256: str

    def __post_init__(self) -> None:
        if self.faces.ndim != 2 or self.faces.shape[1] != 3 or self.faces.shape[0] <= 0:
            raise ManoTopologyError(f"MANO topology has unexpected shape: {self.faces.shape}")
        if self.faces.dtype != np.uint32:
            raise ManoTopologyError(f"MANO topology must use uint32 indices, got {self.faces.dtype}")
        if self.source_hand not in {"LEFT", "RIGHT"}:
            raise ManoTopologyError(f"unknown MANO source hand: {self.source_hand!r}")
        if self.neutral_vertices is not None:
            _validate_neutral_vertices(self.neutral_vertices, MANO_VERTEX_COUNT)

    @property
    def vertex_count(self) -> int:
        return MANO_VERTEX_COUNT

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def has_neutral_vertices(self) -> bool:
        return self.neutral_vertices is not None

    def faces_for_track(self, track: str) -> np.ndarray:
        """Return indices for a displayed RIGHT or mirrored LEFT hand."""

        normalized = str(track).upper()
        if normalized not in {"LEFT", "RIGHT"}:
            raise KeyError(f"unknown hand track: {track!r}")
        if normalized == "RIGHT":
            return self.faces
        # Reflection changes handedness, so reverse each triangle for LEFT.
        return _readonly(self.faces[:, [0, 2, 1]])

    def neutral_for_track(self, track: str) -> np.ndarray | None:
        if self.neutral_vertices is None:
            return None
        normalized = str(track).upper()
        if normalized == "RIGHT":
            return self.neutral_vertices
        if normalized != "LEFT":
            raise KeyError(f"unknown hand track: {track!r}")
        mirrored = np.array(self.neutral_vertices, dtype=np.float32, copy=True)
        mirrored[:, 0] *= -1.0
        return _readonly(mirrored)

    def summary(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "source_hand": self.source_hand,
            "source_sha256": self.source_sha256,
            "topology_sha256": self.topology_sha256,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "neutral_vertices_available": self.has_neutral_vertices,
            "left_winding": "reversed_after_x_reflection",
        }


def topology_from_faces(
    faces: Any,
    *,
    source_path: str | Path = "<memory>",
    source_hand: str = "RIGHT",
    neutral_vertices: Any | None = None,
    expected_face_count: int | None = MANO_FACE_COUNT,
) -> ManoTopology:
    """Build a validated topology object, primarily for tests and smoke tools."""

    normalized_hand = str(source_hand).upper()
    if normalized_hand not in {"LEFT", "RIGHT"}:
        raise ManoTopologyError(f"source_hand must be LEFT or RIGHT, got {source_hand!r}")
    validated_faces = validate_mano_faces(
        faces,
        expected_face_count=expected_face_count,
        require_full_vertex_coverage=True,
    )
    template = None
    if neutral_vertices is not None:
        template = _validate_neutral_vertices(neutral_vertices, MANO_VERTEX_COUNT)
    canonical_faces, canonical_template = _canonicalize_right(
        validated_faces,
        template,
        source_hand=normalized_hand,
    )
    topology_hash = hashlib.sha256(np.ascontiguousarray(canonical_faces).tobytes()).hexdigest()
    path = Path(source_path)
    source_hash = ""
    if path.is_file():
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return ManoTopology(
        faces=_readonly(canonical_faces),
        source_path=path,
        source_sha256=source_hash,
        source_hand=normalized_hand,
        neutral_vertices=canonical_template,
        topology_sha256=topology_hash,
    )


def load_mano_topology(
    model_path: str | Path | None,
    *,
    expected_vertex_count: int = MANO_VERTEX_COUNT,
    expected_face_count: int | None = MANO_FACE_COUNT,
) -> ManoTopology | None:
    """Load topology from one local MANO asset, or return ``None`` if omitted.

    The caller may treat ``None`` as the explicit point-cloud presentation
    mode.  An existing but malformed asset raises ``ManoTopologyError`` so a
    typo or wrong model cannot silently produce a misleading surface.
    """

    if model_path is None or not str(model_path).strip():
        return None
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise ManoAssetUnavailableError(f"MANO model asset not found: {path}")
    if expected_vertex_count != MANO_VERTEX_COUNT:
        raise ManoTopologyError(
            f"TASK-007F requires {MANO_VERTEX_COUNT} MANO vertices, got {expected_vertex_count}"
        )

    source_hand = _asset_hand(path)
    smplx_faces, smplx_template = _faces_from_smplx(path)
    payload = None
    if smplx_faces is None:
        payload = _load_pickle(path)
        smplx_faces = _field(payload, ("f", "faces", "face_indices"))
        smplx_template = _field(payload, ("v_template", "vertices_template", "template"))
    if smplx_faces is None:
        raise ManoTopologyError(
            f"MANO asset does not expose faces/model.faces: {path}"
        )

    faces = validate_mano_faces(
        smplx_faces,
        expected_vertex_count=expected_vertex_count,
        expected_face_count=expected_face_count,
        require_full_vertex_coverage=True,
    )
    template = None
    if smplx_template is not None:
        template = _validate_neutral_vertices(smplx_template, expected_vertex_count)
    canonical_faces, canonical_template = _canonicalize_right(
        faces,
        template,
        source_hand=source_hand,
    )
    topology_hash = hashlib.sha256(np.ascontiguousarray(canonical_faces).tobytes()).hexdigest()
    return ManoTopology(
        faces=_readonly(canonical_faces),
        source_path=path,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_hand=source_hand,
        neutral_vertices=canonical_template,
        topology_sha256=topology_hash,
    )


__all__ = [
    "MANO_FACE_COUNT",
    "MANO_VERTEX_COUNT",
    "ManoAssetUnavailableError",
    "ManoTopology",
    "ManoTopologyError",
    "load_mano_topology",
    "topology_from_faces",
    "validate_mano_faces",
]
