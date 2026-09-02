"""Palm IMU channels for the virtual glove.

The authoritative IMU output is the frozen TASK-005 palm orientation, copied
verbatim. The evaluation-only LEFT/RIGHT comparison bases that live in
``evaluation.kinematics.final_contract`` map the production frame into the
independent benchmark's fixture convention; they are an integration convention
for comparison and are deliberately NOT applied to stored production
orientation here.

Angular velocity is DERIVED. There is no accelerometer channel; see
``accelerometer_feasibility`` for why.
"""

from __future__ import annotations

import numpy as np

DERIVED_ANGULAR_VELOCITY_UNITS = "rad/s"


def angular_velocity_body_frame(
    rotations: np.ndarray,
    timestamps: np.ndarray,
    valid: np.ndarray,
    frame_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame angular velocity between consecutive valid palm orientations.

    For a consecutive pair the relative rotation in the earlier body frame is

        R_rel = R[k]^T @ R[k+1]

    whose axis-angle ``(axis, theta)`` gives ``omega = axis * theta / dt``. This
    is what a gyroscope rigidly mounted on the palm measures: a body-frame rate,
    which is dimensionless in position and therefore unaffected by the
    uncalibrated translational scale discussed in
    :func:`accelerometer_feasibility`.

    A sample is emitted at index ``k+1`` only when ALL of the following hold, so
    nothing is ever bridged across a gap:

    * both ``k`` and ``k+1`` have a valid palm orientation;
    * the two frames are genuinely adjacent (``frame_index`` differs by 1);
    * the actual timestamp delta is finite and strictly positive.

    Otherwise the entry stays NaN with its validity False. Index 0 is always
    invalid: a rate needs two samples. No smoothing, filtering or interpolation
    is applied -- TASK-005 records large frame-to-frame orientation changes and
    they must remain visible here rather than be averaged away.
    """

    rotations = np.asarray(rotations, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    frame_index = np.asarray(frame_index)

    frames = rotations.shape[0]
    omega = np.full((frames, 3), np.nan, dtype=np.float64)
    omega_valid = np.zeros((frames,), dtype=bool)

    for k in range(frames - 1):
        if not (valid[k] and valid[k + 1]):
            continue
        if int(frame_index[k + 1]) - int(frame_index[k]) != 1:
            continue
        dt = float(timestamps[k + 1] - timestamps[k])
        if not np.isfinite(dt) or dt <= 0.0:
            continue
        first, second = rotations[k], rotations[k + 1]
        if not (np.isfinite(first).all() and np.isfinite(second).all()):
            continue
        relative = first.T @ second
        # Axis-angle via the well-conditioned atan2 form rather than
        # arccos((tr-1)/2), whose derivative is unbounded near theta = 0 -- and
        # a nearly still hand is the common case.
        vector = np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ],
            dtype=np.float64,
        )
        sin_term = 0.5 * float(np.linalg.norm(vector))
        cos_term = 0.5 * (float(np.trace(relative)) - 1.0)
        theta = float(np.arctan2(sin_term, np.clip(cos_term, -1.0, 1.0)))
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            if theta < 1e-9:
                omega[k + 1] = np.zeros(3)
                omega_valid[k + 1] = True
            # theta near pi with a vanishing axis vector is genuinely ambiguous
            # in sign; it is left invalid rather than guessed.
            continue
        omega[k + 1] = (vector / norm) * (theta / dt)
        omega_valid[k + 1] = True

    return omega, omega_valid


def accelerometer_feasibility() -> dict:
    """Why TASK-006A emits no accelerometer channel.

    A physical IMU normally carries one, but fabricating values to complete the
    analogy would be dishonest. The evidence, measured on this pilot:

    * The frozen TASK-005 kinematics output contains NO position channel at
      all -- only orientation. Any position would have to be pulled back from
      the TASK-004 tracked stage.
    * That stage's ``camera_translation`` is a monocular weak-perspective
      estimate tied to WiLoR's assumed focal length, not a calibrated metric
      depth. Measured on the pilot, |t_z| is about 397x the reconstructed palm
      length: a hand 38 units away while measuring 0.095 across would be far
      below one pixel if those units were metres. The absolute scale is
      therefore arbitrary, and acceleration is not scale-free.
    * Acceleration is a second derivative. TASK-005A records palm-orientation
      changes up to 102 degrees between adjacent frames; differentiating a
      noisier position signal twice at 30 fps amplifies that error by roughly
      900x.
    * A real accelerometer measures specific force including gravity. No
      gravity direction and no metric scale are available, so the output would
      not be comparable to any physical device.

    Angular velocity has none of these problems: orientation is independently
    validated, a body-frame rate is scale-free, and the timestamps are real.
    """

    return {
        "accelerometer": "DEFER ACCELEROMETER",
        "reasons": [
            "frozen TASK-005 output carries no position channel, only orientation",
            "upstream camera_translation is uncalibrated weak-perspective scale, "
            "not metric depth (|t_z| ~= 397x palm length on this pilot)",
            "second derivative amplifies known orientation/position jitter by ~900x at 30 fps",
            "no gravity reference, so specific force is not reproducible",
        ],
        "gyroscope": "IMPLEMENTED (derived, body frame, unsmoothed, never bridged)",
        "gyroscope_justification": (
            "body-frame angular rate is dimensionless in position, so the "
            "uncalibrated translational scale does not affect it; orientation is "
            "already validated and timestamps are real"
        ),
    }
