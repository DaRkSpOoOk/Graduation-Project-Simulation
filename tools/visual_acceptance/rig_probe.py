"""Offline probe that reproduces the QML bone-application maths exactly.

The QML stage does ``node.rotation = restRotation.times(delta)`` for every bone
named by the rig profile.  This module walks the same glTF node hierarchy and
applies the identical composition, so the resulting world transforms and
skinned vertices can be measured without a GPU or a window.

It exists so the TASK-007G framing and orbit claims are reproducible rather
than anecdotal; the regression tests drive it directly.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from smart_glove_app.rendering.glb_index import GlbHierarchy as _BaseHierarchy
from smart_glove_app.rendering.glb_index import GlbIndexError, GltfNode


def quaternion_to_matrix(quaternion_wxyz: Any) -> np.ndarray:
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def compose(translation: Any, rotation_wxyz: Any, scale: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_matrix(rotation_wxyz) * np.asarray(scale, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


class GlbHierarchy(_BaseHierarchy):
    """Presentation GLB plus the transform maths the QML stage performs."""

    def world_transforms(self, overrides: Mapping[int, Any] | None = None) -> dict[int, np.ndarray]:
        """World matrix per node, with optional per-node local rotation overrides.

        ``overrides`` maps a node index to a replacement WXYZ local rotation,
        exactly like assigning ``node.rotation`` in QML.
        """

        replacements = overrides or {}
        result: dict[int, np.ndarray] = {}

        def recurse(index: int, parent: np.ndarray) -> None:
            node = self.nodes[index]
            rotation = replacements.get(index, node.rotation_wxyz)
            world = parent @ compose(node.translation, rotation, node.scale)
            result[index] = world
            for child in node.children:
                recurse(child, world)

        for root in self.roots:
            recurse(root, np.eye(4, dtype=np.float64))
        return result

    def accessor(self, index: int) -> np.ndarray:
        accessor = self.json["accessors"][index]
        view = self.json["bufferViews"][accessor["bufferView"]]
        dtype = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[
            accessor["componentType"]
        ]
        components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[accessor["type"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        data = np.frombuffer(self.binary, dtype=dtype, count=accessor["count"] * components, offset=start)
        return data.reshape(accessor["count"], components) if components > 1 else data


def skinned_vertices(
    hierarchy: GlbHierarchy,
    mesh_node: GltfNode,
    world: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Linear-blend-skin one mesh node with the supplied world transforms."""

    if mesh_node.skin is None or mesh_node.mesh is None:
        raise GlbIndexError(f"node {mesh_node.name!r} is not a skinned mesh")

    skin = hierarchy.json["skins"][int(mesh_node.skin)]
    joints = [int(v) for v in skin["joints"]]
    inverse_bind = hierarchy.accessor(int(skin["inverseBindMatrices"])).reshape(-1, 4, 4)
    inverse_bind = np.transpose(inverse_bind, (0, 2, 1))  # glTF matrices are column-major
    joint_matrices = np.stack([world[joint] @ inverse_bind[i] for i, joint in enumerate(joints)])

    primitive = hierarchy.json["meshes"][int(mesh_node.mesh)]["primitives"][0]
    positions = hierarchy.accessor(primitive["attributes"]["POSITION"]).astype(np.float64)
    joint_index = hierarchy.accessor(primitive["attributes"]["JOINTS_0"]).astype(np.int64)
    weights = hierarchy.accessor(primitive["attributes"]["WEIGHTS_0"]).astype(np.float64)

    homogeneous = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
    result = np.zeros((len(positions), 3), dtype=np.float64)
    for slot in range(joint_index.shape[1]):
        matrices = joint_matrices[joint_index[:, slot]]
        result += weights[:, slot : slot + 1] * np.einsum("nij,nj->ni", matrices, homogeneous)[:, :3]
    return result


__all__ = ["GlbHierarchy", "GlbIndexError", "GltfNode", "compose", "quaternion_to_matrix", "skinned_vertices"]
