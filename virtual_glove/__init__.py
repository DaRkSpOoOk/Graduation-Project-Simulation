"""TASK-006A: ideal virtual smart-glove sensor model.

Converts the frozen TASK-005 geometric kinematics contract into an ideal
simulated glove of 19 Hall-type channels plus one palm IMU per hand.

No LSTM, no training dataset, no sensor-ablation study, no tracking or
kinematics mathematics is changed here.
"""

from .extractor import (
    GloveInputError,
    GloveSequence,
    REQUIRED_KINEMATICS_ARRAYS,
    extract_glove_sequence,
)
from .imu import (
    DERIVED_ANGULAR_VELOCITY_UNITS,
    accelerometer_feasibility,
    angular_velocity_body_frame,
)
from .io import (
    ARRAY_ORDER,
    GLOVE_META_NAME,
    GLOVE_NPZ_NAME,
    SCHEMA_VERSION,
    SENSOR_LAYOUT_NAME,
    build_metadata,
    load_glove_sequence,
    save_glove_sequence,
    sha256_file,
)
from .layout import (
    BEND_SENSOR_IDS,
    CHAIN_ORDER,
    EXPECTED_BEND_SENSORS,
    EXPECTED_HALL_SENSORS,
    EXPECTED_IMU_SENSORS,
    EXPECTED_SENSOR_PACKAGES,
    EXPECTED_SPREAD_SENSORS,
    FINGER_ORDER,
    HALL_SENSOR_IDS,
    IMU_SENSOR_IDS,
    LAYOUT_VERSION,
    MARKER_HALL,
    MARKER_IMU,
    SENSOR_LAYOUT,
    SPREAD_PAIRS,
    SPREAD_SENSOR_IDS,
    TRACK_ORDER,
    SensorSpec,
    layout_document,
    sensor_by_id,
)
from .signals import (
    ADC_BITS,
    ADC_INVALID_SENTINEL,
    ADC_MAX,
    ADC_MIN,
    CONTRACT_MAX_DEG,
    CONTRACT_MIN_DEG,
    NORMALIZATION_DIVISOR,
    SensorContractViolation,
    contract_violation_mask,
    normalize_angles,
    to_adc_12bit,
)

__all__ = [
    "ADC_BITS", "ADC_INVALID_SENTINEL", "ADC_MAX", "ADC_MIN", "ARRAY_ORDER",
    "BEND_SENSOR_IDS", "CHAIN_ORDER", "CONTRACT_MAX_DEG", "CONTRACT_MIN_DEG",
    "DERIVED_ANGULAR_VELOCITY_UNITS", "EXPECTED_BEND_SENSORS",
    "EXPECTED_HALL_SENSORS", "EXPECTED_IMU_SENSORS", "EXPECTED_SENSOR_PACKAGES",
    "EXPECTED_SPREAD_SENSORS", "FINGER_ORDER", "GLOVE_META_NAME", "GLOVE_NPZ_NAME",
    "GloveInputError", "GloveSequence", "HALL_SENSOR_IDS", "IMU_SENSOR_IDS",
    "LAYOUT_VERSION", "MARKER_HALL", "MARKER_IMU", "NORMALIZATION_DIVISOR",
    "REQUIRED_KINEMATICS_ARRAYS", "SCHEMA_VERSION", "SENSOR_LAYOUT",
    "SENSOR_LAYOUT_NAME", "SPREAD_PAIRS", "SPREAD_SENSOR_IDS",
    "SensorContractViolation", "SensorSpec", "TRACK_ORDER", "accelerometer_feasibility",
    "angular_velocity_body_frame", "build_metadata", "contract_violation_mask",
    "extract_glove_sequence", "layout_document", "load_glove_sequence",
    "normalize_angles", "save_glove_sequence", "sensor_by_id", "sha256_file",
    "to_adc_12bit",
]
