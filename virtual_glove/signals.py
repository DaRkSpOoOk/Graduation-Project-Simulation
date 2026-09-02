"""Angle -> ideal sensor value transfer for the virtual glove.

Three representations are kept deliberately distinct and are never conflated:

1. **geometric angle** -- the frozen TASK-005 value in degrees, copied verbatim;
2. **normalized ideal sensor value** -- the authoritative ML-facing signal;
3. **optional ADC-like encoding** -- a hardware-looking view, NOT authoritative.
"""

from __future__ import annotations

import numpy as np

# TASK-005 flexion and spread are unsigned angles contractually bounded by
# 0..180 degrees. The normalizer is that contract, nothing else: it is fixed a
# priori, is identical for every channel, hand, subject and dataset, and has no
# fitted parameter of any kind.
CONTRACT_MIN_DEG = 0.0
CONTRACT_MAX_DEG = 180.0
NORMALIZATION_DIVISOR = 180.0

# Optional ideal ADC. 12-bit, full scale, unlike the old physical prototype's
# ~850-1700 counts, which described that specific hardware and is not the
# ideal simulated-glove contract.
ADC_BITS = 12
ADC_MIN = 0
ADC_MAX = 4095
ADC_INVALID_SENTINEL = -1


class SensorContractViolation(ValueError):
    """A finite angle fell outside the frozen TASK-005 0..180 degree contract."""


def contract_violation_mask(angles_deg: np.ndarray) -> np.ndarray:
    """Finite entries outside [0, 180]. NaN is absence, not a violation."""

    angles = np.asarray(angles_deg, dtype=np.float64)
    finite = np.isfinite(angles)
    return finite & ((angles < CONTRACT_MIN_DEG) | (angles > CONTRACT_MAX_DEG))


def describe_violations(
    angles_deg: np.ndarray, channel: str, limit: int = 10
) -> list[dict]:
    mask = contract_violation_mask(angles_deg)
    angles = np.asarray(angles_deg, dtype=np.float64)
    detail: list[dict] = []
    for position in zip(*np.nonzero(mask)):
        if len(detail) >= limit:
            break
        detail.append(
            {"channel": channel, "index": [int(i) for i in position],
             "value_deg": float(angles[position])}
        )
    return detail


def normalize_angles(
    angles_deg: np.ndarray, *, channel: str = "angle", on_violation: str = "raise"
) -> tuple[np.ndarray, np.ndarray]:
    """Map degrees onto the ideal [0, 1] sensor scale.

    ``normalized = degrees / 180``.

    Returns ``(normalized, violation_mask)``. NaN in, NaN out: an absent angle
    stays absent and is never substituted.

    A finite value outside the contract is surfaced, never silently repaired.
    With ``on_violation="raise"`` (the default) it raises
    :class:`SensorContractViolation`; with ``"flag"`` it is returned in the mask
    so the caller can invalidate that channel. The value is NOT clamped in
    either case, and the original degrees remain available untouched.

    No pilot-derived or per-subject normalization is applied anywhere: there is
    no min/max fitting, no per-finger neutral-offset subtraction, and no
    dataset-dependent term.
    """

    if on_violation not in ("raise", "flag"):
        raise ValueError(f"on_violation must be 'raise' or 'flag', got {on_violation!r}")

    angles = np.asarray(angles_deg, dtype=np.float64)
    violations = contract_violation_mask(angles)
    if violations.any() and on_violation == "raise":
        raise SensorContractViolation(
            f"{int(violations.sum())} finite {channel} value(s) outside the frozen "
            f"TASK-005 contract [{CONTRACT_MIN_DEG}, {CONTRACT_MAX_DEG}] degrees; "
            f"first offenders: {describe_violations(angles, channel, limit=3)}"
        )
    return angles / NORMALIZATION_DIVISOR, violations


def to_adc_12bit(
    normalized: np.ndarray, valid: np.ndarray, *, integer: bool = True
) -> np.ndarray:
    """Optional hardware-looking encoding. Not authoritative for ML.

    ``adc = normalized * 4095`` over the valid channels. Rounding, when integer
    output is requested, is half-up (``floor(x + 0.5)``): deterministic,
    monotone, and exact at both rails -- 0.0 -> 0 and 1.0 -> 4095.

    Invalid channels stay explicitly invalid. They carry
    :data:`ADC_INVALID_SENTINEL` (-1), which is outside the 0..4095 range and so
    cannot be mistaken for a reading, and they remain masked by the same
    ``bend_valid``/``spread_valid`` arrays that govern every other view.
    """

    normalized = np.asarray(normalized, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    scaled = np.where(valid, normalized * ADC_MAX, np.nan)
    if not integer:
        return np.where(valid, scaled, np.nan).astype(np.float32)
    counts = np.full(normalized.shape, ADC_INVALID_SENTINEL, dtype=np.int32)
    if valid.any():
        counts[valid] = np.floor(scaled[valid] + 0.5).astype(np.int32)
    return counts
