"""Discovery and audit of an ALREADY-DOWNLOADED local KArSL dataset.

TASK-008A was written while the official network source was unavailable, so its
acquisition path assumes a download. This module covers the case that now
applies: the official archives are already extracted on disk and must be
discovered, parsed and audited in place. Nothing is downloaded, renamed,
reorganized or copied into the repository.

Discovered layout (not assumed -- see the TASK-008B report):

    <signer>/<train|test>/videos/<SignID>/<chapter>_<signer>_<SignID>_(<stamp>)_c.mp4

The leading filename field is a chapter code: "01" for SignIDs 1-31 (numbers)
and "02" for 32-70 (letters), which independently corroborates the official
workbook's own chapter split.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

VIDEO_SUFFIX = ".mp4"
PARTITIONS = ("train", "test")

# <chapter>_<signer>_<signid>_(DD_MM_YY_HH_MM_SS)_c.mp4
FILENAME_RE = re.compile(
    r"^(?P<chapter>\d{2})_(?P<signer>\d{2})_(?P<sign_id>\d{4})_"
    r"\((?P<stamp>\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\)_(?P<modality>[a-z])\.mp4$"
)


class LocalSourceError(RuntimeError):
    """The local dataset does not match the expected official layout."""


@dataclass(frozen=True, slots=True)
class LocalVideo:
    """One discovered source video, described only by portable data."""

    relative_path: str          # portable, beneath the dataset root
    file_name: str
    signer_id: str              # from the directory, authoritative
    official_partition: str
    sign_id: int                # from the directory, authoritative
    chapter: str
    filename_signer: str
    filename_sign_id: int
    timestamp_token: str
    modality: str
    size_bytes: int
    path_sign_id_matches_filename: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VideoProbe:
    relative_path: str
    ok: bool
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    packet_count: int | None = None
    duration_seconds: float | None = None
    codec: str | None = None
    container: str | None = None
    sha256: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_video_filename(name: str) -> dict[str, Any] | None:
    match = FILENAME_RE.match(name)
    if not match:
        return None
    return {
        "chapter": match.group("chapter"),
        "signer": match.group("signer"),
        "sign_id": int(match.group("sign_id")),
        "timestamp_token": match.group("stamp"),
        "modality": match.group("modality"),
    }


def discover_local_videos(
    dataset_root: str | Path, *, sign_ids: Iterable[int] | None = None
) -> list[LocalVideo]:
    """Enumerate the official videos in a deterministic order.

    The directory is trusted over the filename for signer / SignID identity:
    the official archive places each clip under its own class directory, and a
    filename field can disagree (the TASK-008B audit found one such block). The
    disagreement is recorded per row rather than resolved silently.
    """

    root = Path(dataset_root)
    if not root.is_dir():
        raise LocalSourceError(f"Dataset root does not exist: {root}")

    wanted = set(sign_ids) if sign_ids is not None else None
    videos: list[LocalVideo] = []
    for path in sorted(root.rglob(f"*{VIDEO_SUFFIX}"), key=lambda p: p.as_posix()):
        relative = path.relative_to(root)
        parts = relative.parts
        if len(parts) < 5:
            continue
        signer, partition, marker, sign_token = parts[0], parts[1].lower(), parts[2], parts[3]
        if partition not in PARTITIONS or marker.lower() != "videos":
            continue
        if not (signer.isdigit() and sign_token.isdigit()):
            continue
        sign_id = int(sign_token)
        if wanted is not None and sign_id not in wanted:
            continue
        parsed = parse_video_filename(path.name)
        if parsed is None:
            raise LocalSourceError(f"Unrecognized official filename: {relative.as_posix()}")
        videos.append(
            LocalVideo(
                relative_path=relative.as_posix(),
                file_name=path.name,
                signer_id=signer,
                official_partition=partition,
                sign_id=sign_id,
                chapter=parsed["chapter"],
                filename_signer=parsed["signer"],
                filename_sign_id=parsed["sign_id"],
                timestamp_token=parsed["timestamp_token"],
                modality=parsed["modality"],
                size_bytes=path.stat().st_size,
                path_sign_id_matches_filename=parsed["sign_id"] == sign_id,
            )
        )
    return videos


def _ffprobe(path: Path, *, count_packets: bool) -> dict[str, Any]:
    entries = "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,duration"
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_packets:
        command.append("-count_packets")
        entries += ",nb_read_packets"
    command += ["-show_entries", entries, "-show_entries", "format=duration,format_name",
                "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:200] or "ffprobe failed")
    return json.loads(result.stdout or "{}")


def _rate(value: str | None) -> float | None:
    if not value or "/" not in value:
        return None
    num, den = value.split("/", 1)
    try:
        numerator, denominator = float(num), float(den)
    except ValueError:
        return None
    return numerator / denominator if denominator else None


def probe_video(
    dataset_root: Path, relative_path: str, *, count_packets: bool = False, hash_file: bool = True
) -> VideoProbe:
    path = dataset_root / relative_path
    try:
        payload = _ffprobe(path, count_packets=count_packets)
        streams = payload.get("streams") or []
        if not streams:
            return VideoProbe(relative_path, False, error="no_video_stream")
        stream = streams[0]
        fmt = payload.get("format") or {}
        frames = stream.get("nb_frames")
        packets = stream.get("nb_read_packets")
        duration = stream.get("duration") or fmt.get("duration")
        digest = None
        if hash_file:
            hasher = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
        return VideoProbe(
            relative_path=relative_path,
            ok=True,
            width=int(stream["width"]) if stream.get("width") else None,
            height=int(stream["height"]) if stream.get("height") else None,
            fps=_rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate")),
            frame_count=int(frames) if frames not in (None, "N/A") else None,
            packet_count=int(packets) if packets not in (None, "N/A") else None,
            duration_seconds=float(duration) if duration not in (None, "N/A") else None,
            codec=stream.get("codec_name"),
            container=fmt.get("format_name"),
            sha256=digest,
        )
    except Exception as error:  # noqa: BLE001 - a bad file must not abort the audit
        return VideoProbe(relative_path, False, error=f"{type(error).__name__}: {error}")


def probe_videos(
    dataset_root: str | Path,
    videos: Sequence[LocalVideo],
    *,
    workers: int = 8,
    count_packets: bool = False,
    hash_files: bool = True,
    progress: bool = False,
) -> dict[str, VideoProbe]:
    """Probe many videos in parallel, preserving deterministic output order."""

    root = Path(dataset_root)
    results: dict[str, VideoProbe] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(probe_video, root, video.relative_path,
                        count_packets=count_packets, hash_file=hash_files): video.relative_path
            for video in videos
        }
        done = 0
        for future in concurrent.futures.as_completed(futures):
            probe = future.result()
            results[probe.relative_path] = probe
            done += 1
            if progress and done % 500 == 0:
                print(f"  probed {done}/{len(futures)}", flush=True)
    return {video.relative_path: results[video.relative_path] for video in videos}


# ---------------------------------------------------------------------------
# Deterministic benchmark subset (TASK-008B Phase J)
# ---------------------------------------------------------------------------

SMOKE_TARGET_ROWS = 24


def select_smoke_subset(
    rows: Sequence[dict[str, Any]], *, target: int = SMOKE_TARGET_ROWS
) -> list[dict[str, str]]:
    """Choose a deterministic, length-diverse benchmark subset.

    Nothing is sampled randomly. The rule, applied to Core-28 manifest rows:

    1. Keep only rows with a known positive frame count.
    2. Form the 6 (signer, partition) cells -- 3 signers x {train, test} -- so
       every signer and both official partitions are represented.
    3. Within each cell, order by frame count then ``sample_id`` and take the
       shortest, the median and the longest sequence. Sequence-length spread is
       deliberate: this subset is also the throughput benchmark, and a
       uniform-length subset would misestimate the full run.
    4. Fill the remaining budget by walking the cells round-robin, taking the
       next unused row nearest each cell's own median, so added rows stay
       representative rather than extreme.
    5. Emit in a stable order: signer, partition, frame count, sample_id.

    The selection is a pure function of the manifest, so the same manifest
    always yields the same subset in the same order.
    """

    usable = [row for row in rows if str(row.get("frame_count", "")).strip().isdigit()
              and int(row["frame_count"]) > 0]
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in usable:
        cells.setdefault((row["signer_id"], row["official_partition"]), []).append(row)
    for key in cells:
        cells[key].sort(key=lambda item: (int(item["frame_count"]), item["sample_id"]))

    chosen: dict[str, dict[str, Any]] = {}
    ordered_cells = sorted(cells)
    for key in ordered_cells:
        bucket = cells[key]
        if not bucket:
            continue
        for index in (0, len(bucket) // 2, len(bucket) - 1):
            candidate = bucket[index]
            chosen.setdefault(candidate["sample_id"], candidate)

    # round-robin fill from each cell's median outwards
    offsets = 1
    while len(chosen) < target and offsets < 10_000:
        added = False
        for key in ordered_cells:
            if len(chosen) >= target:
                break
            bucket = cells[key]
            middle = len(bucket) // 2
            for position in (middle + offsets, middle - offsets):
                if 0 <= position < len(bucket):
                    candidate = bucket[position]
                    if candidate["sample_id"] not in chosen:
                        chosen[candidate["sample_id"]] = candidate
                        added = True
                        break
        if not added:
            break
        offsets += 1

    selected = sorted(
        chosen.values(),
        key=lambda item: (item["signer_id"], item["official_partition"],
                          int(item["frame_count"]), item["sample_id"]),
    )
    return [dict(row) for row in selected[:target]]
