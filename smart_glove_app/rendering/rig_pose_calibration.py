"""Read-only presentation-rig calibration from the shipped hand GLB.

The Qt Quick 3D scene snapshots the local rotation of every loaded bone and
applies a presentation delta as ``rest * delta``.  This module exposes the
same immutable rest rotations to the Python-side landmark retargeter.  It is
deliberately a small GLB reader; it does not edit, save, or export an asset.

The calibration is presentation-only.  It never replaces the TASK-005
kinematic arrays and it is not used by the recognition bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .glb_index import GlbHierarchy, GlbIndexError


def quaternion_to_matrix_wxyz(value: object) -> np.ndarray:
    """Return a finite 3x3 rotation matrix for a WXYZ quaternion."""

    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise GlbIndexError("GLB rotation is not a finite WXYZ quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise GlbIndexError("GLB rotation quaternion has zero length")
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(value: object) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a deterministic WXYZ value."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise GlbIndexError("retarget rotation is not a finite 3x3 matrix")
    determinant = float(np.linalg.det(matrix))
    if abs(determinant - 1.0) > 1e-4 or not np.allclose(
        matrix.T @ matrix, np.eye(3), atol=1e-4
    ):
        raise GlbIndexError("retarget rotation is not a proper orthonormal matrix")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(max(1e-12, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(max(1e-12, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(max(1e-12, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.asarray((w, x, y, z), dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12 or not np.isfinite(quaternion).all():
        raise GlbIndexError("retarget rotation could not be converted to a quaternion")
    quaternion /= norm
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


@dataclass(frozen=True, slots=True)
class GlbPoseCalibration:
    """Immutable local/world rest rotations for one presentation GLB."""

    path: Path
    node_indices: Mapping[str, int]
    parent_names: Mapping[str, str | None]
    local_rotations: Mapping[str, np.ndarray]
    world_rotations: Mapping[str, np.ndarray]

    @classmethod
    def from_glb(cls, path: str | Path, required_bones: tuple[str, ...]) -> "GlbPoseCalibration":
        target = Path(path).expanduser().resolve()
        hierarchy = GlbHierarchy(target)
        by_name: dict[str, int] = {}
        parent_by_index: dict[int, int | None] = {}

        def visit(index: int, parent: int | None) -> None:
            node = hierarchy.nodes[index]
            if node.name in by_name:
                raise GlbIndexError(f"{target.name} contains duplicate node name {node.name!r}")
            by_name[node.name] = index
            parent_by_index[index] = parent
            for child in node.children:
                visit(child, index)

        for root in hierarchy.roots:
            visit(root, None)

        missing = [bone for bone in required_bones if bone not in by_name]
        if missing:
            raise GlbIndexError(
                f"{target.name} is missing retarget bones: {', '.join(missing)}"
            )

        local: dict[str, np.ndarray] = {}
        world: dict[str, np.ndarray] = {}
        world_by_index: dict[int, np.ndarray] = {}

        def solve_world(index: int, parent_world: np.ndarray) -> None:
            node = hierarchy.nodes[index]
            current = parent_world @ quaternion_to_matrix_wxyz(node.rotation_wxyz)
            world_by_index[index] = current
            local[node.name] = quaternion_to_matrix_wxyz(node.rotation_wxyz)
            world[node.name] = current
            for child in node.children:
                solve_world(child, current)

        for root in hierarchy.roots:
            solve_world(root, np.eye(3, dtype=np.float64))

        parents = {
            bone: (
                hierarchy.nodes[parent_by_index[by_name[bone]]].name
                if parent_by_index[by_name[bone]] is not None
                else None
            )
            for bone in required_bones
        }
        return cls(
            path=target,
            node_indices={bone: by_name[bone] for bone in required_bones},
            parent_names=parents,
            local_rotations={bone: local[bone].copy() for bone in required_bones},
            world_rotations={bone: world[bone].copy() for bone in required_bones},
        )

    def parent_world_rotation(self, bone: str) -> np.ndarray:
        """Return the immutable rest world rotation of a bone's parent."""

        parent = self.parent_names.get(str(bone))
        if parent is None:
            return np.eye(3, dtype=np.float64)
        try:
            return self.world_rotations[parent].copy()
        except KeyError as exc:
            raise GlbIndexError(
                f"parent {parent!r} of retarget bone {bone!r} is not in the calibration"
            ) from exc


__all__ = [
    "GlbPoseCalibration",
    "matrix_to_quaternion_wxyz",
    "quaternion_to_matrix_wxyz",
]
