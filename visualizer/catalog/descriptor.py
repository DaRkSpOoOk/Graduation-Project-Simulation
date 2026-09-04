"""Renderer-neutral descriptors for one isolated TASK-008 sequence.

The descriptor names artifacts by repository-relative paths.  It carries no
GUI, mesh, tensor, or renderer object, so TASK-007A can consume it without
having to repeat label or filesystem resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _required(row: Mapping[str, str], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"manifest row is missing {key!r}")
    return value


def _relative(value: str, field: str) -> str:
    if not str(value).strip():
        raise ValueError(f"{field} must not be empty")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path: {value!r}")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class SequenceDescriptor:
    """The paths and provenance required to load one isolated sign sequence."""

    sample_id: str
    run_root: str
    pose_relative_path: str
    tracking_relative_path: str
    kinematics_relative_path: str
    virtual_glove_relative_path: str
    sequence_length: int
    source_relative_path: str = ""
    source_sha256: str = ""
    manifest_sha256: str = ""
    signer_id: str = ""
    official_partition: str = ""
    repetition_id: str = ""

    def __post_init__(self) -> None:
        if not self.sample_id or "/" in self.sample_id or "\\" in self.sample_id:
            raise ValueError(f"sample_id must be a safe identifier: {self.sample_id!r}")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        for field_name in (
            "pose_relative_path",
            "tracking_relative_path",
            "kinematics_relative_path",
            "virtual_glove_relative_path",
        ):
            _relative(str(getattr(self, field_name)), field_name)
        if self.source_relative_path:
            _relative(self.source_relative_path, "source_relative_path")

    @classmethod
    def from_manifest_row(cls, row: Mapping[str, str], run_root: str | Path) -> "SequenceDescriptor":
        """Build a descriptor from the authoritative virtual-glove manifest row."""

        sample_id = _required(row, "sample_id")
        sequence_text = row.get("sequence_length") or row.get("source_frame_count") or ""
        try:
            sequence_length = int(sequence_text)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid sequence_length for {sample_id!r}: {sequence_text!r}") from error
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive for {sample_id!r}")

        vg_path = _relative(_required(row, "virtual_glove_relative_path"), "virtual_glove_relative_path")
        return cls(
            sample_id=sample_id,
            run_root=str(Path(run_root).expanduser().resolve()),
            pose_relative_path=f"pose/raw/{sample_id}/wilor_raw.npz",
            tracking_relative_path=f"tracking/{sample_id}/wilor_tracked.npz",
            kinematics_relative_path=f"kinematics/{sample_id}/hand_kinematics.npz",
            virtual_glove_relative_path=vg_path,
            sequence_length=sequence_length,
            source_relative_path=str(row.get("source_relative_path", "")),
            source_sha256=str(row.get("source_sha256", "")),
            manifest_sha256=str(row.get("manifest_sha256", "")),
            signer_id=str(row.get("signer_id", "")),
            official_partition=str(row.get("official_partition", "")),
            repetition_id=str(row.get("repetition_id", "")),
        )

    def absolute_path(self, stage: str) -> Path:
        """Return an artifact path for a known stage without renderer coupling."""

        relative = {
            "pose": self.pose_relative_path,
            "tracking": self.tracking_relative_path,
            "kinematics": self.kinematics_relative_path,
            "virtual_glove": self.virtual_glove_relative_path,
        }.get(stage)
        if relative is None:
            raise ValueError(f"unknown sequence stage {stage!r}")
        return Path(self.run_root) / relative

    def to_dict(self) -> dict[str, object]:
        return {
            "kinematics_relative_path": self.kinematics_relative_path,
            "manifest_sha256": self.manifest_sha256,
            "official_partition": self.official_partition,
            "pose_relative_path": self.pose_relative_path,
            "repetition_id": self.repetition_id,
            "run_root": self.run_root,
            "sample_id": self.sample_id,
            "sequence_length": self.sequence_length,
            "signer_id": self.signer_id,
            "source_relative_path": self.source_relative_path,
            "source_sha256": self.source_sha256,
            "tracking_relative_path": self.tracking_relative_path,
            "virtual_glove_relative_path": self.virtual_glove_relative_path,
        }
