"""TASK-009C all-signers deployment training."""

from .plan import (
    DEPLOYMENT_FOLD_TAG,
    DEPLOYMENT_TRAINING_ROLE,
    EPOCH_POLICY,
    EXPECTED_SAMPLES,
    PLAN_SCHEMA_VERSION,
    PRIMARY_FEATURE_SET,
    PRIMARY_INPUT_POLICY,
    PRIMARY_POOLING,
    PRIMARY_QUATERNION_POLICY,
    PRIMARY_SEED,
    DeploymentPlanError,
    audit_deployment_dataset,
    build_deployment_plan,
    derive_epoch_budget,
    load_plan,
    read_primary_loso_evidence,
    write_plan,
)
from .verify import (
    HASHED_FILES,
    REQUIRED_FILES,
    DeploymentVerificationError,
    sha256_file,
    verify_checkpoint_metadata,
    verify_files,
    verify_history,
    verify_plan,
    verify_status,
)
from .train import (
    DEPLOYMENT_CHECKPOINT,
    DEPLOYMENT_METRIC_NAME,
    RESUME_CHECKPOINT,
    deployment_metadata,
    deployment_spec,
    train_deployment_model,
)

__all__ = [
    "DEPLOYMENT_FOLD_TAG", "DEPLOYMENT_TRAINING_ROLE", "EPOCH_POLICY", "EXPECTED_SAMPLES",
    "PLAN_SCHEMA_VERSION", "PRIMARY_FEATURE_SET", "PRIMARY_INPUT_POLICY", "PRIMARY_POOLING",
    "PRIMARY_QUATERNION_POLICY", "PRIMARY_SEED", "DeploymentPlanError",
    "audit_deployment_dataset", "build_deployment_plan", "derive_epoch_budget", "load_plan",
    "read_primary_loso_evidence", "write_plan", "DEPLOYMENT_CHECKPOINT",
    "DEPLOYMENT_METRIC_NAME", "RESUME_CHECKPOINT", "deployment_metadata", "deployment_spec",
    "train_deployment_model", "HASHED_FILES", "REQUIRED_FILES",
    "DeploymentVerificationError", "sha256_file", "verify_checkpoint_metadata",
    "verify_files", "verify_history", "verify_plan", "verify_status",
]
