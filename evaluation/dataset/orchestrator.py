"""Resumable TASK-008A worker orchestration.

This module owns scheduling/state/provenance only.  The model-specific
mathematics remains in the frozen WiLoR, tracking, kinematics and virtual
glove packages.  A worker initializes WiLoR once, then processes its assigned
source videos in deterministic manifest order.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .manifest import load_manifest, manifest_sha256, validate_manifest_rows

STAGES: tuple[str, ...] = ("POSE", "TRACKING", "KINEMATICS", "VIRTUAL_GLOVE")
STAGE_STATUS = {
    "POSE": "POSE_DONE",
    "TRACKING": "TRACKING_DONE",
    "KINEMATICS": "KINEMATICS_DONE",
    "VIRTUAL_GLOVE": "VIRTUAL_GLOVE_DONE",
}
RUN_SCHEMA_VERSION = "task008a_run_v1"
STATE_SCHEMA_VERSION = "task008a_state_v1"
TASK005_CONTRACT = "TASK-005-final-v2"
TASK006_CONTRACT = "TASK-006-ideal-virtual-glove-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_shard_index(manifest_position: int, num_shards: int) -> int:
    """Assign a row using stable manifest ordering and modulo."""

    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if manifest_position < 0:
        raise ValueError("manifest_position must be non-negative")
    return manifest_position % num_shards


def assign_shards(rows: Iterable[Mapping[str, str]], num_shards: int) -> dict[int, list[dict[str, str]]]:
    materialized = [dict(row) for row in rows]
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    result = {index: [] for index in range(num_shards)}
    for position, row in enumerate(materialized):
        result[stable_shard_index(position, num_shards)].append(row)
    return result


def sample_ids_for_shard(rows: Iterable[Mapping[str, str]], num_shards: int, shard_index: int) -> list[str]:
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index {shard_index} outside [0, {num_shards})")
    return [row["sample_id"] for row in assign_shards(rows, num_shards)[shard_index]]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _safe_sample_id(sample_id: str) -> str:
    if not sample_id or sample_id in {".", ".."} or "/" in sample_id or "\\" in sample_id:
        raise ValueError(f"Unsafe sample_id: {sample_id!r}")
    return sample_id


@dataclass(slots=True)
class RunPaths:
    root: Path

    @property
    def pose(self) -> Path:
        return self.root / "pose" / "raw"

    @property
    def tracking(self) -> Path:
        return self.root / "tracking"

    @property
    def kinematics(self) -> Path:
        return self.root / "kinematics"

    @property
    def virtual_glove(self) -> Path:
        return self.root / "virtual_glove"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def failures(self) -> Path:
        return self.root / "failures"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance"

    @property
    def export(self) -> Path:
        return self.root / "export"

    def stage_dir(self, stage: str, sample_id: str) -> Path:
        _safe_sample_id(sample_id)
        return {
            "POSE": self.pose,
            "TRACKING": self.tracking,
            "KINEMATICS": self.kinematics,
            "VIRTUAL_GLOVE": self.virtual_glove,
        }[stage] / sample_id

    def stage_sidecar(self, stage: str, sample_id: str) -> Path:
        return self.stage_dir(stage, sample_id) / "task008a_stage.json"

    def state_file(self, shard_index: int) -> Path:
        return self.state / f"shard-{shard_index:02d}.json"


class StateStore:
    """Atomic per-shard state owned by one SLURM worker."""

    def __init__(self, paths: RunPaths, shard_index: int, num_shards: int, manifest_hash: str) -> None:
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError(f"shard_index {shard_index} outside [0, {num_shards})")
        self.path = paths.state_file(shard_index)
        self.payload: dict[str, Any] = {
            "schema_version": STATE_SCHEMA_VERSION,
            "run_schema_version": RUN_SCHEMA_VERSION,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "manifest_sha256": manifest_hash,
            "updated_utc": utc_now(),
            "samples": {},
        }
        if self.path.is_file():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if loaded.get("manifest_sha256") != manifest_hash:
                raise ValueError(f"State manifest hash mismatch: {self.path}")
            if int(loaded.get("shard_index", -1)) != shard_index or int(loaded.get("num_shards", -1)) != num_shards:
                raise ValueError(f"State shard configuration mismatch: {self.path}")
            self.payload = loaded

    @property
    def samples(self) -> dict[str, dict[str, Any]]:
        return self.payload.setdefault("samples", {})

    def get(self, sample_id: str) -> dict[str, Any]:
        return self.samples.setdefault(
            sample_id,
            {
                "status": "PENDING",
                "current_stage": None,
                "attempts": 0,
                "retry_count": 0,
                "frames_processed": 0,
                "last_error": None,
            },
        )

    def update(self, sample_id: str, **values: Any) -> None:
        self.get(sample_id).update(values)
        self.payload["updated_utc"] = utc_now()
        _atomic_json(self.path, self.payload)

    def persist(self) -> None:
        self.payload["updated_utc"] = utc_now()
        _atomic_json(self.path, self.payload)


def _stage_sidecar_payload(
    stage: str,
    row: Mapping[str, str],
    manifest_hash: str,
    source_sha256: str,
    *,
    frames: int,
    upstream_sha256: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "stage": stage.lower(),
        "status": STAGE_STATUS[stage],
        "sample_id": row["sample_id"],
        "source_relative_path": row["source_relative_path"],
        "source_sha256": source_sha256,
        "manifest_sha256": manifest_hash,
        "frames": frames,
        "task005_contract": TASK005_CONTRACT,
        "task006_contract": TASK006_CONTRACT,
        "created_utc": utc_now(),
    }
    if upstream_sha256:
        payload["upstream_sha256"] = upstream_sha256
    if extra:
        payload.update(extra)
    return payload


def _read_stage_sidecar(paths: RunPaths, stage: str, sample_id: str) -> dict[str, Any] | None:
    sidecar = paths.stage_sidecar(stage, sample_id)
    if not sidecar.is_file():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def validate_stage_artifact(
    paths: RunPaths,
    stage: str,
    row: Mapping[str, str],
    manifest_hash: str,
    *,
    source_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Validate a completed stage before ``--resume`` skips it."""

    stage = stage.upper()
    sample_id = row["sample_id"]
    expected_source = source_sha256 or row.get("source_sha256", "")
    sidecar = _read_stage_sidecar(paths, stage, sample_id)
    if not sidecar or sidecar.get("status") != STAGE_STATUS[stage]:
        return None
    if (
        sidecar.get("stage") != stage.lower()
        or sidecar.get("sample_id") != sample_id
        or sidecar.get("manifest_sha256") != manifest_hash
    ):
        return None
    if expected_source and sidecar.get("source_sha256") != expected_source:
        return None
    directory = paths.stage_dir(stage, sample_id)
    try:
        if stage == "POSE":
            artifact = directory / "wilor_raw.npz"
            if not artifact.is_file():
                return None
            import numpy as np

            with np.load(artifact, allow_pickle=False) as data:
                if "run_metadata_json" not in data or "frame_index" not in data:
                    return None
                run_metadata = json.loads(str(data["run_metadata_json"]))
                if run_metadata.get("mode") != "full" or len(data["frame_index"]) <= 0:
                    return None
                # The raw pose NPZ holds ONE ROW PER DETECTED HAND, so a
                # two-hand video has twice as many rows as frames. The sidecar
                # records video frames, so the comparison must be against the
                # number of distinct frame indices. Comparing row counts made
                # every already-complete POSE stage look stale, which silently
                # defeated --resume for the most expensive stage in the run.
                frames = int(np.unique(np.asarray(data["frame_index"])).size)
        elif stage == "TRACKING":
            artifact = directory / "wilor_tracked.npz"
            if not artifact.is_file():
                return None
            import numpy as np

            with np.load(artifact, allow_pickle=False) as data:
                if "frame_index" not in data or len(data["frame_index"]) <= 0:
                    return None
                frames = int(len(data["frame_index"]))
        elif stage == "KINEMATICS":
            artifact = directory / "hand_kinematics.npz"
            if not artifact.is_file():
                return None
            import numpy as np

            with np.load(artifact, allow_pickle=False) as data:
                required = {"frame_index", "flexion_deg", "adjacent_spread_deg", "valid_palm_frame"}
                if not required <= set(data.files) or len(data["frame_index"]) <= 0:
                    return None
                frames = int(len(data["frame_index"]))
        else:
            artifact = directory / "virtual_glove.npz"
            meta = directory / "virtual_glove_meta.json"
            if not artifact.is_file() or not meta.is_file():
                return None
            import numpy as np

            with np.load(artifact, allow_pickle=False) as data:
                required = {"frame_index", "bend_normalized", "spread_normalized", "palm_imu_valid"}
                if not required <= set(data.files) or len(data["frame_index"]) <= 0:
                    return None
                frames = int(len(data["frame_index"]))
        declared_artifact = sidecar.get("artifact")
        if declared_artifact and Path(str(declared_artifact)).resolve() != artifact.resolve():
            return None
        if stage != "POSE":
            upstream_stage = {
                "TRACKING": "POSE",
                "KINEMATICS": "TRACKING",
                "VIRTUAL_GLOVE": "KINEMATICS",
            }[stage]
            upstream = {
                "POSE": paths.pose / sample_id / "wilor_raw.npz",
                "TRACKING": paths.tracking / sample_id / "wilor_tracked.npz",
                "KINEMATICS": paths.kinematics / sample_id / "hand_kinematics.npz",
            }[upstream_stage]
            expected_upstream_sha = sidecar.get("upstream_sha256")
            if (
                not expected_upstream_sha
                or not upstream.is_file()
                or sha256_file(upstream) != expected_upstream_sha
            ):
                return None
        if int(sidecar.get("frames", -1)) != frames:
            return None
    except Exception:  # noqa: BLE001
        # A truncated or otherwise unreadable artifact (an interrupted write
        # raises zipfile.BadZipFile, which is not an OSError) means "not
        # usable", so the stage is recomputed. It must never abort the sample:
        # a long run has to survive whatever the last interruption left behind.
        return None
    return {"artifact": str(artifact), "frames": frames, "sidecar": sidecar}


def _run_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else result.stdout.strip() or result.stderr.strip()


def environment_provenance(
    *,
    manifest: Path,
    data_root: Path,
    run_root: Path,
    shard_index: int,
    num_shards: int,
    repository_root: Path,
    device: str,
) -> dict[str, Any]:
    """Collect compact, persistent worker provenance before model loading."""

    git_commit = _run_command(["git", "-C", str(repository_root), "rev-parse", "HEAD"])
    git_branch = _run_command(["git", "-C", str(repository_root), "branch", "--show-current"])
    nvidia = _run_command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    try:
        import numpy as np

        numpy_version = np.__version__
    except ImportError:
        numpy_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_runtime = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda_available else []
        tf32 = {
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if hasattr(torch.backends, "cuda") else None,
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32) if hasattr(torch.backends, "cudnn") else None,
        }
    except ImportError:
        torch_version = cuda_runtime = None
        cuda_available = False
        gpu_names = []
        tf32 = {}
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    ram_gb = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                ram_gb = round(float(line.split()[1]) / (1024 * 1024), 3)
                break
    except (OSError, ValueError, IndexError):
        pass
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "recorded_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "slurm": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION",
                "SLURM_JOB_ACCOUNT", "SLURM_JOB_QOS", "CUDA_VISIBLE_DEVICES",
            )
        },
        "hardware": {
            "cpu_model": cpu_model,
            "execution_device_requested": device,
        },
        "gpu": {
            "nvidia_smi": nvidia,
            "torch_cuda_available": cuda_available,
            "torch_device_names": gpu_names,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "memory": {"ram_gb": ram_gb},
        "packages": {"numpy": numpy_version, "torch": torch_version, "cuda_runtime": cuda_runtime, "tf32": tf32},
        "wilor": {
            "upstream_commit": "fcb911312a38fa8badd30d9656a167485d61b8f9",
            "mode": "full FP32; fast/FP16 disabled",
            "detector_confidence": 0.3,
            "rescale_factor": 2.0,
            "assets_dir": os.environ.get("WILOR_ASSETS_DIR"),
        },
        "contracts": {"task005": TASK005_CONTRACT, "task006": TASK006_CONTRACT},
        "manifest": {
            "path": str(manifest),
            "sha256": manifest_sha256(manifest),
            "data_root": str(data_root),
            "run_root": str(run_root),
        },
        "shard": {"index": shard_index, "count": num_shards},
    }


def _append_failure(paths: RunPaths, shard_index: int, payload: Mapping[str, Any]) -> None:
    paths.failures.mkdir(parents=True, exist_ok=True)
    path = paths.failures / f"shard-{shard_index:02d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _write_stage_sidecar(paths: RunPaths, stage: str, sample_id: str, payload: Mapping[str, Any]) -> None:
    destination = paths.stage_sidecar(stage, sample_id)
    _atomic_json(destination, payload)


def _video_source_sha(row: Mapping[str, str], data_root: Path) -> str:
    expected = row.get("source_sha256", "").strip()
    path = data_root / row["source_relative_path"]
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Source video missing or empty: {path}")
    actual = sha256_file(path)
    if expected and actual.lower() != expected.lower():
        raise ValueError(f"Source SHA-256 mismatch for {row['sample_id']}: {actual} != {expected}")
    return actual


def _run_pose(row: Mapping[str, str], paths: RunPaths, data_root: Path, manifest_hash: str, pipeline: Any) -> dict[str, Any]:
    from pose.wilor.frame_extraction import EXTRACTOR_NAME
    from pose.wilor.npz_io import save_raw_video_output
    from pose.wilor.video_processing import process_video_full
    from video_io.reader import inspect_video

    source = data_root / row["source_relative_path"]
    inspection = inspect_video(source)
    if not inspection.decoder_success or not inspection.fps or inspection.fps <= 0:
        raise ValueError(f"Video preflight failed: {inspection.error or source}")
    source_sha = _video_source_sha(row, data_root)
    result = process_video_full(
        source,
        row["sample_id"],
        pipeline,
        runtime_confidence=0.3,
        rescale_factor=2.0,
        extractor_version=f"{EXTRACTOR_NAME}@fcb911312a38fa8badd30d9656a167485d61b8f9",
        checkpoint_id=str(pipeline.checkpoint_paths.wilor_checkpoint),
    )
    if result.total_frames_decoded <= 0:
        raise ValueError("WiLoR decoded zero frames")
    output = save_raw_video_output(
        paths.pose,
        row["sample_id"],
        result.frames,
        result.vertices_by_hand,
        run_metadata={
            "mode": "full",
            "stage": "pose",
            "task": "TASK-008A",
            "sample_id": row["sample_id"],
            "source_video": row["source_relative_path"],
            "source_video_sha256": source_sha,
            "manifest_sha256": manifest_hash,
            "extractor_version": f"{EXTRACTOR_NAME}@fcb911312a38fa8badd30d9656a167485d61b8f9",
            "wilor_upstream_commit": "fcb911312a38fa8badd30d9656a167485d61b8f9",
            "runtime_mode": "full FP32",
            "detector_confidence": 0.3,
            "rescale_factor": 2.0,
            "video_inspection": inspection.to_dict(),
        },
    )
    payload = _stage_sidecar_payload(
        "POSE", row, manifest_hash, source_sha, frames=result.total_frames_decoded,
        extra={
            "artifact": str(output),
            "decoder_success": inspection.decoder_success,
            "reported_frame_count": inspection.reported_frame_count,
            "decoded_frame_count": inspection.decoded_frame_count,
            "fps": inspection.fps,
            "width": inspection.width,
            "height": inspection.height,
            "duration_seconds": inspection.duration_seconds,
            "frame_errors": result.frame_errors,
            "inference_seconds": result.inference_seconds,
            "effective_fps": result.effective_fps,
        },
    )
    _write_stage_sidecar(paths, "POSE", row["sample_id"], payload)
    return payload


def _run_tracking(row: Mapping[str, str], paths: RunPaths, manifest_hash: str, source_sha: str) -> dict[str, Any]:
    from tracking.wilor import TrackerConfig, compute_metrics, load_raw_sequence, save_tracked_sequence, track_sequence

    sample = row["sample_id"]
    raw_path = paths.pose / sample / "wilor_raw.npz"
    raw = load_raw_sequence(raw_path, sample)
    source = {
        "raw_npz": str(raw_path),
        "raw_npz_sha256": sha256_file(raw_path),
        "source_video": row["source_relative_path"],
        "source_video_sha256": source_sha,
        "manifest_sha256": manifest_hash,
        "task008a": True,
    }
    config = TrackerConfig()
    sequence = track_sequence(raw, config, source=source)
    metrics = compute_metrics(sequence)
    output, _ = save_tracked_sequence(paths.tracking, sequence, metrics.to_dict())
    payload = _stage_sidecar_payload(
        "TRACKING", row, manifest_hash, source_sha, frames=raw.total_frames,
        upstream_sha256=sha256_file(raw_path),
        extra={"artifact": str(output), "metrics": metrics.to_dict()},
    )
    _write_stage_sidecar(paths, "TRACKING", sample, payload)
    return payload


def _run_kinematics(row: Mapping[str, str], paths: RunPaths, manifest_hash: str, source_sha: str) -> dict[str, Any]:
    from kinematics import build_metadata, extract_sequence, save_kinematics
    from tracking.wilor import load_tracked_sequence

    sample = row["sample_id"]
    tracked_dir = paths.tracking / sample
    tracked_npz = tracked_dir / "wilor_tracked.npz"
    arrays, tracked_meta = load_tracked_sequence(tracked_dir)
    sequence = extract_sequence(arrays, tracked_meta, sample)
    output = save_kinematics(
        paths.kinematics / sample,
        sequence,
        build_metadata(
            sequence,
            tracked_dir=tracked_dir,
            tracked_sha256=sha256_file(tracked_npz),
            tracked_metadata=tracked_meta,
            implementation_commit=_run_command(["git", "rev-parse", "HEAD"]) or "unknown",
        ),
    )
    payload = _stage_sidecar_payload(
        "KINEMATICS", row, manifest_hash, source_sha, frames=int(sequence.frame_index.shape[0]),
        upstream_sha256=sha256_file(tracked_npz),
        extra={"artifact": str(output)},
    )
    _write_stage_sidecar(paths, "KINEMATICS", sample, payload)
    return payload


def _run_virtual_glove(row: Mapping[str, str], paths: RunPaths, manifest_hash: str, source_sha: str) -> dict[str, Any]:
    import numpy as np

    from kinematics import KINEMATICS_META_NAME, KINEMATICS_NPZ_NAME
    from virtual_glove import build_metadata, extract_glove_sequence, save_glove_sequence

    sample = row["sample_id"]
    kinematics_dir = paths.kinematics / sample
    kinematics_npz = kinematics_dir / KINEMATICS_NPZ_NAME
    with np.load(kinematics_npz, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads((kinematics_dir / KINEMATICS_META_NAME).read_text(encoding="utf-8"))
    sequence = extract_glove_sequence(arrays, metadata, sample)
    output = save_glove_sequence(
        paths.virtual_glove / sample,
        sequence,
        build_metadata(
            sequence,
            kinematics_dir=kinematics_dir,
            kinematics_sha256=sha256_file(kinematics_npz),
            kinematics_metadata=metadata,
            implementation_commit=_run_command(["git", "rev-parse", "HEAD"]) or "unknown",
        ),
    )
    payload = _stage_sidecar_payload(
        "VIRTUAL_GLOVE", row, manifest_hash, source_sha, frames=int(sequence.frame_index.shape[0]),
        upstream_sha256=sha256_file(kinematics_npz),
        extra={
            "artifact": str(output),
            "bend_valid": int(sequence.bend_valid.sum()),
            "bend_total": int(sequence.bend_valid.size),
            "spread_valid": int(sequence.spread_valid.sum()),
            "spread_total": int(sequence.spread_valid.size),
            "imu_valid": int(sequence.palm_imu_valid.sum()),
            "imu_total": int(sequence.palm_imu_valid.size),
            "contract_violation_count": len(sequence.contract_violations),
        },
    )
    _write_stage_sidecar(paths, "VIRTUAL_GLOVE", sample, payload)
    return payload


def process_sample(
    row: Mapping[str, str],
    *,
    paths: RunPaths,
    data_root: Path,
    manifest_hash: str,
    state: StateStore,
    pipeline: Any,
    resume: bool,
    retry_failed: bool,
    requested_stage: str = "ALL",
) -> dict[str, Any]:
    """Process one sample, preserving valid earlier stages on resume."""

    sample = row["sample_id"]
    entry = state.get(sample)
    if entry.get("status") == "FAILED" and not retry_failed:
        return {
            "sample_id": sample,
            "status": "SKIPPED_FAILED",
            "stage": entry.get("current_stage"),
            "frames": int(entry.get("frames_processed", 0)),
        }
    requested = requested_stage.upper()
    if requested not in {"ALL", *STAGES}:
        raise ValueError(f"Unknown stage: {requested_stage}")
    start_index = 0 if requested == "ALL" else STAGES.index(requested)
    final_payload: dict[str, Any] = {
        "sample_id": sample,
        "status": entry.get("status", "PENDING"),
        "stage": entry.get("current_stage"),
        "frames": 0,
    }
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    entry["last_error"] = None
    source_sha = ""
    # Per-stage wall time, so a benchmark can attribute cost rather than treat
    # model inference as the whole job. Skipped (resumed) stages are recorded
    # as such and never counted as compute.
    stage_seconds: dict[str, float] = {}
    stage_skipped: list[str] = []
    sample_started = time.perf_counter()
    state.persist()
    try:
        source_sha = _video_source_sha(row, data_root)
        for stage_index, stage in enumerate(STAGES[start_index:], start=start_index):
            stage_started = time.perf_counter()
            existing = validate_stage_artifact(paths, stage, row, manifest_hash, source_sha256=source_sha) if resume else None
            if existing:
                stage_skipped.append(stage)
                final_payload["frames"] = existing["frames"]
                entry["status"] = STAGE_STATUS[stage]
                entry["frames_processed"] = existing["frames"]
                final_payload["stage"] = stage
                continue
            entry["current_stage"] = stage
            final_payload["stage"] = stage
            state.persist()
            if stage == "POSE":
                payload = _run_pose(row, paths, data_root, manifest_hash, pipeline)
            elif stage == "TRACKING":
                payload = _run_tracking(row, paths, manifest_hash, source_sha)
            elif stage == "KINEMATICS":
                payload = _run_kinematics(row, paths, manifest_hash, source_sha)
            else:
                payload = _run_virtual_glove(row, paths, manifest_hash, source_sha)
            stage_seconds[stage] = time.perf_counter() - stage_started
            entry["status"] = STAGE_STATUS[stage]
            entry["current_stage"] = None
            entry["frames_processed"] = int(payload.get("frames", 0))
            final_payload["frames"] = int(payload.get("frames", 0))
            state.persist()
        final_payload["status"] = entry.get("status", "PENDING")
        final_payload["stage_seconds"] = stage_seconds
        final_payload["stages_skipped"] = stage_skipped
        final_payload["sample_seconds"] = time.perf_counter() - sample_started
        return final_payload
    except Exception as error:  # one bad sample is recorded, not fatal to the shard
        entry["status"] = "FAILED"
        entry["current_stage"] = entry.get("current_stage") or "UNKNOWN"
        entry["retry_count"] = int(entry.get("retry_count", 0)) + 1
        entry["last_error"] = {"type": type(error).__name__, "message": str(error), "utc": utc_now()}
        state.persist()
        _append_failure(
            paths,
            int(state.payload["shard_index"]),
            {
                "sample_id": sample,
                "shard": state.payload["shard_index"],
                "stage": entry["current_stage"],
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp_utc": utc_now(),
                "retry_count": entry["retry_count"],
            },
        )
        return {
            "sample_id": sample,
            "status": "FAILED",
            "stage": entry.get("current_stage"),
            "frames": int(entry.get("frames_processed", 0)),
            "error": str(error),
            "stage_seconds": stage_seconds,
            "stages_skipped": stage_skipped,
            "sample_seconds": time.perf_counter() - sample_started,
        }


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


class ProgressDisplay:
    def __init__(self, shard_index: int, num_shards: int, total: int) -> None:
        self.shard_index = shard_index
        self.num_shards = num_shards
        self.total = total
        self.started = time.perf_counter()
        self.success = 0
        self.failed = 0
        self.frames = 0
        self.last_sample = "-"

    def update(self, result: Mapping[str, Any]) -> None:
        self.last_sample = str(result.get("sample_id", "-"))
        if result.get("status") in {"FAILED", "SKIPPED_FAILED"}:
            self.failed += 1
        else:
            self.success += 1
        self.frames += int(result.get("frames", 0) or 0)
        completed = self.success + self.failed
        elapsed = time.perf_counter() - self.started
        rate = self.frames / elapsed if elapsed > 0 else 0.0
        remaining = max(self.total - completed, 0)
        eta = remaining * elapsed / completed if completed and rate else None
        print(
            f"TASK-008A shard {self.shard_index:02d}/{self.num_shards}: "
            f"{completed}/{self.total} ({100.0 * completed / self.total if self.total else 100.0:.1f}%) "
            f"current={self.last_sample} stage={result.get('stage') or result.get('status')} "
            f"success={self.success} failed={self.failed} frames={self.frames} "
            f"rolling_fps={rate:.2f} elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}"
        , flush=True)


def read_state_files(run_root: str | Path) -> list[dict[str, Any]]:
    root = Path(run_root).resolve() / "state"
    result = []
    for path in sorted(root.glob("shard-*.json")):
        try:
            result.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            result.append({"state_file": str(path), "malformed": True})
    return result


def status_snapshot(run_root: str | Path, manifest: str | Path | None = None) -> dict[str, Any]:
    """Aggregate state read-only; it never imports WiLoR or changes files."""

    rows = []
    if manifest and Path(manifest).is_file():
        from .manifest import load_manifest

        rows = load_manifest(manifest)
    states = read_state_files(run_root)
    sample_states: dict[str, dict[str, Any]] = {}
    for state in states:
        sample_states.update(state.get("samples", {}))
    if rows:
        for row in rows:
            sample_states.setdefault(row["sample_id"], {"status": "PENDING", "frames_processed": 0})
    counts = {name: 0 for name in ("PENDING", "POSE_DONE", "TRACKING_DONE", "KINEMATICS_DONE", "VIRTUAL_GLOVE_DONE", "FAILED")}
    in_progress = 0
    for entry in sample_states.values():
        status = entry.get("status", "PENDING")
        counts[status] = counts.get(status, 0) + 1
        if entry.get("current_stage") and status not in {"FAILED", "VIRTUAL_GLOVE_DONE"}:
            in_progress += 1
    total_frames = sum(int(entry.get("frames_processed", 0) or 0) for entry in sample_states.values())
    completed = counts.get("VIRTUAL_GLOVE_DONE", 0)
    return {
        "run_root": str(Path(run_root).resolve()),
        "total_samples": len(rows) if rows else len(sample_states),
        "complete": completed,
        "pending": counts.get("PENDING", 0),
        "failed": counts.get("FAILED", 0),
        "in_progress": in_progress,
        "frames_processed": total_frames,
        "stage_counts": counts,
        "shards": [
            {
                "shard_index": state.get("shard_index"),
                "num_shards": state.get("num_shards"),
                "manifest_sha256": state.get("manifest_sha256"),
                "samples": len(state.get("samples", {})),
                "updated_utc": state.get("updated_utc"),
            }
            for state in states
        ],
    }


def format_status(snapshot: Mapping[str, Any]) -> str:
    counts = snapshot.get("stage_counts", {})
    lines = [
        "TASK-008A CORE-28",
        "===============================",
        f"Total samples:      {snapshot.get('total_samples', 0)}",
        f"Complete:           {snapshot.get('complete', 0)}",
        f"Pending:            {snapshot.get('pending', 0)}",
        f"Failed:             {snapshot.get('failed', 0)}",
        f"In progress:        {snapshot.get('in_progress', 0)}",
        f"Frames processed:   {snapshot.get('frames_processed', 0)}",
        "",
        f"POSE_DONE:          {counts.get('POSE_DONE', 0)}",
        f"TRACKING_DONE:      {counts.get('TRACKING_DONE', 0)}",
        f"KINEMATICS_DONE:    {counts.get('KINEMATICS_DONE', 0)}",
        f"VIRTUAL_GLOVE_DONE: {counts.get('VIRTUAL_GLOVE_DONE', 0)}",
        "",
        "Per-shard:",
    ]
    for shard in snapshot.get("shards", []):
        lines.append(
            f"{int(shard.get('shard_index', -1)):02d} "
            f"samples={shard.get('samples', 0)} updated={shard.get('updated_utc', 'unknown')}"
        )
    return "\n".join(lines)


def run_worker(
    *,
    manifest: Path,
    data_root: Path,
    run_root: Path,
    num_shards: int,
    shard_index: int,
    resume: bool,
    retry_failed: bool,
    limit: int | None = None,
    sample_id: str | None = None,
    stage: str = "ALL",
    device: str = "cuda",
) -> dict[str, Any]:
    rows = validate_manifest_rows(load_manifest(manifest))
    if not rows:
        raise ValueError("Cannot run a worker with an empty/schema-only manifest")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index {shard_index} outside [0, {num_shards})")
    manifest_hash = manifest_sha256(manifest)
    shards = assign_shards(rows, num_shards)
    selected = shards[shard_index]
    if sample_id:
        selected = [row for row in selected if row["sample_id"] == sample_id]
        if not selected:
            raise ValueError(f"Sample {sample_id!r} is not assigned to shard {shard_index}")
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        selected = selected[:limit]
    paths = RunPaths(run_root.resolve())
    paths.root.mkdir(parents=True, exist_ok=True)
    state = StateStore(paths, shard_index, num_shards, manifest_hash)
    provenance = environment_provenance(
        manifest=manifest.resolve(), data_root=data_root.resolve(), run_root=paths.root,
        shard_index=shard_index, num_shards=num_shards,
        repository_root=Path(__file__).resolve().parents[2], device=device,
    )
    _atomic_json(paths.provenance / f"shard-{shard_index:02d}.json", provenance)

    if not selected:
        state.persist()
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "manifest_sha256": manifest_hash,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "results": [],
            "completed": 0,
            "failed": 0,
        }

    # Heavy imports and model construction occur only in worker mode, once per shard.
    from pose.wilor.config import WilorAssetPaths, WilorRuntimeConfig
    from pose.wilor.model_loader import check_assets, load_pipeline

    assets = WilorAssetPaths.resolve()
    check_assets(assets)
    # The model is constructed ONCE here and reused for every sample in the
    # shard; it is never rebuilt per video.
    model_load_started = time.perf_counter()
    pipeline = load_pipeline(assets, WilorRuntimeConfig(device=device, fast_mode=False, detector_confidence=0.3, rescale_factor=2.0))
    model_load_seconds = time.perf_counter() - model_load_started
    progress = ProgressDisplay(shard_index, num_shards, len(selected))
    results = []
    for row in selected:
        result = process_sample(
            row, paths=paths, data_root=data_root.resolve(), manifest_hash=manifest_hash,
            state=state, pipeline=pipeline, resume=resume, retry_failed=retry_failed, requested_stage=stage,
        )
        results.append(result)
        progress.update(result)
    state.persist()
    peak_vram_bytes = None
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            peak_vram_bytes = int(torch.cuda.max_memory_allocated())
    except Exception:  # noqa: BLE001 - diagnostics must never fail the run
        peak_vram_bytes = None
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "manifest_sha256": manifest_hash,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "results": results,
        "completed": sum(result.get("status") not in {"FAILED", "SKIPPED_FAILED"} for result in results),
        "failed": sum(result.get("status") in {"FAILED", "SKIPPED_FAILED"} for result in results),
        "model_load_seconds": model_load_seconds,
        "model_loads": 1,
        "peak_vram_bytes": peak_vram_bytes,
        "worker_wall_seconds": time.perf_counter() - model_load_started,
    }


def write_index(
    path: str | Path,
    rows: Iterable[Mapping[str, str]],
    run_root: str | Path,
    *,
    manifest_hash_value: str,
) -> None:
    """Write a compact authoritative output index from external run artifacts."""

    root = Path(run_root).resolve()
    if not manifest_hash_value or len(manifest_hash_value) != 64:
        raise ValueError("manifest_hash_value must be the SHA-256 of the source manifest")
    fields = (
        "sample_id", "sign_id", "label_ar", "label_index", "signer_id", "official_partition",
        "source_relative_path", "source_sha256", "frame_count", "pose_status", "tracking_status",
        "kinematics_status", "virtual_glove_status", "bend_valid_fraction", "spread_valid_fraction",
        "imu_valid_fraction",
    )
    output: list[dict[str, object]] = []
    for row in validate_manifest_rows(rows):
        sample = row["sample_id"]
        values: dict[str, object] = {field: row.get(field, "") for field in fields}
        for stage, status in (("POSE", "pose_status"), ("TRACKING", "tracking_status"), ("KINEMATICS", "kinematics_status"), ("VIRTUAL_GLOVE", "virtual_glove_status")):
            values[status] = (
                STAGE_STATUS[stage]
                if validate_stage_artifact(
                    RunPaths(root),
                    stage,
                    row,
                    manifest_hash_value,
                    source_sha256=row.get("source_sha256", ""),
                )
                else "PENDING"
            )
        glove = root / "virtual_glove" / sample / "virtual_glove.npz"
        if glove.is_file():
            try:
                import numpy as np

                with np.load(glove, allow_pickle=False) as data:
                    values["frame_count"] = int(len(data["frame_index"]))
                    values["bend_valid_fraction"] = float(data["bend_valid"].mean())
                    values["spread_valid_fraction"] = float(data["spread_valid"].mean())
                    values["imu_valid_fraction"] = float(data["palm_imu_valid"].mean())
            except (OSError, KeyError, ValueError):
                pass
        output.append(values)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
