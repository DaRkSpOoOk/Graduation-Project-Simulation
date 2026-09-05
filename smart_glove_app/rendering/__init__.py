"""Presentation-only MANO, mesh, and Qt Quick 3D rendering surfaces."""

from .hand_mesh_state import (
    HandMeshState,
    HandRenderPayload,
    PersistentRenderScene,
    PresentationFrame,
    PresentationGeometryError,
    compute_vertex_normals,
)
from .mano_topology import (
    MANO_FACE_COUNT,
    MANO_VERTEX_COUNT,
    ManoAssetUnavailableError,
    ManoTopology,
    ManoTopologyError,
    load_mano_topology,
    topology_from_faces,
    validate_mano_faces,
)
from .qt_geometry import QT_QUICK3D_AVAILABLE, QtHandGeometry

__all__ = [
    "MANO_FACE_COUNT",
    "MANO_VERTEX_COUNT",
    "ManoAssetUnavailableError",
    "ManoTopology",
    "ManoTopologyError",
    "HandMeshState",
    "HandRenderPayload",
    "PersistentRenderScene",
    "PresentationFrame",
    "PresentationGeometryError",
    "QT_QUICK3D_AVAILABLE",
    "QtHandGeometry",
    "compute_vertex_normals",
    "load_mano_topology",
    "topology_from_faces",
    "validate_mano_faces",
]
