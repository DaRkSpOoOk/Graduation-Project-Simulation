"""Read the exported presentation GLB and index it for the QML scene.

Qt Quick 3D's ``RuntimeLoader`` builds its node tree without carrying the glTF
node names across, so a QML-side lookup by name cannot work.  TASK-007F worked
around that with structural child-index paths hand-written into the rig
profile, which silently go stale the moment the asset is re-exported.

This module removes that failure mode: the paths are derived from the shipped
GLB at startup, so they always describe the asset actually being loaded.  The
QML side then locates the subtree those paths are relative to by validating
candidates, rather than trusting a hard-coded prefix.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


class GlbIndexError(ValueError):
    """Raised when a GLB cannot be indexed for presentation."""


@dataclass(frozen=True, slots=True)
class GltfNode:
    index: int
    name: str
    translation: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    scale: tuple[float, float, float]
    children: tuple[int, ...]
    mesh: int | None
    skin: int | None


class GlbHierarchy:
    """Read-only glTF node graph loaded straight from a binary container."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            payload = self.path.read_bytes()
        except OSError as exc:
            raise GlbIndexError(f"cannot read {self.path}: {exc}") from exc
        if payload[:4] != b"glTF":
            raise GlbIndexError(f"{self.path} is not a binary glTF container")

        offset = 12
        chunks: list[tuple[int, bytes]] = []
        while offset + 8 <= len(payload):
            length, chunk_type = struct.unpack_from("<II", payload, offset)
            chunks.append((chunk_type, payload[offset + 8 : offset + 8 + length]))
            offset += 8 + length
        if not chunks:
            raise GlbIndexError(f"{self.path} contains no glTF chunks")

        self.json: Mapping[str, Any] = json.loads(chunks[0][1].decode("utf-8"))
        self.binary = chunks[1][1] if len(chunks) > 1 else b""

        self.nodes: list[GltfNode] = []
        for index, node in enumerate(self.json.get("nodes", [])):
            xyzw = node.get("rotation", (0.0, 0.0, 0.0, 1.0))  # glTF is XYZW, the project uses WXYZ
            self.nodes.append(
                GltfNode(
                    index=index,
                    name=str(node.get("name", "")),
                    translation=tuple(float(v) for v in node.get("translation", (0.0, 0.0, 0.0))),
                    rotation_wxyz=(float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])),
                    scale=tuple(float(v) for v in node.get("scale", (1.0, 1.0, 1.0))),
                    children=tuple(int(v) for v in node.get("children", ())),
                    mesh=node.get("mesh"),
                    skin=node.get("skin"),
                )
            )
        scenes = self.json.get("scenes")
        if not scenes:
            raise GlbIndexError(f"{self.path} declares no scene")
        self.roots = tuple(int(v) for v in scenes[self.json.get("scene", 0)]["nodes"])

    def walk(self, index: int, depth: int = 0) -> Iterator[tuple[int, GltfNode]]:
        yield depth, self.nodes[index]
        for child in self.nodes[index].children:
            yield from self.walk(child, depth + 1)

    def path_to(self, root_position: int, wanted: str) -> list[int] | None:
        """Child-index path from ``scene.nodes[root_position]`` down to ``wanted``.

        The first element is the position inside the scene's root list; every
        further element is an index into that node's ``children``.
        """

        def search(index: int, prefix: list[int]) -> list[int] | None:
            node = self.nodes[index]
            if node.name == wanted:
                return prefix
            for position, child in enumerate(node.children):
                found = search(child, prefix + [position])
                if found is not None:
                    return found
            return None

        return search(self.roots[root_position], [root_position])


def build_scene_index(
    glb_path: str | Path,
    *,
    roots: Mapping[str, str],
    bones: tuple[str, ...],
) -> dict[str, Any]:
    """Describe the GLB in the form the QML stage needs.

    Returns per-side child-index paths for the presentation root and for every
    required bone, plus the scene-root count so QML can identify the container
    node those paths are relative to.
    """

    hierarchy = GlbHierarchy(glb_path)
    positions: dict[str, int] = {}
    for side, name in roots.items():
        matches = [i for i, node_index in enumerate(hierarchy.roots) if hierarchy.nodes[node_index].name == name]
        if not matches:
            raise GlbIndexError(f"{Path(glb_path).name} has no scene root named {name!r}")
        positions[side.upper()] = matches[0]

    index: dict[str, Any] = {
        "sceneRootCount": len(hierarchy.roots),
        "sides": {},
    }
    for side, position in positions.items():
        side_entry: dict[str, Any] = {"root": [position], "bones": {}}
        missing = []
        for bone in bones:
            path = hierarchy.path_to(position, bone)
            if path is None:
                missing.append(bone)
            else:
                side_entry["bones"][bone] = path
        if missing:
            raise GlbIndexError(
                f"{Path(glb_path).name} is missing {side} bones: {', '.join(sorted(missing))}"
            )
        index["sides"][side] = side_entry
    return index


__all__ = ["GlbHierarchy", "GlbIndexError", "GltfNode", "build_scene_index"]
