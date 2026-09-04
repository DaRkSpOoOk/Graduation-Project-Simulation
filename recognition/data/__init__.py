"""TASK-009A frozen sequence-input contract and PyTorch data pipeline."""

from .collate import (
    BatchError,
    collate_sequences,
    concat_features_and_masks,
    key_padding_mask,
    make_collate_fn,
)
from .contract import (
    CONTRACT_VERSION,
    FEATURE_SETS,
    HAND_ORDER,
    NUM_CLASSES,
    QUATERNION_POLICIES,
    SequenceInputConfig,
    channel_index,
    channel_names,
    contract_document,
    family_slice,
    feature_dimension,
    hand_slice,
)
from .loso import FOLD_SIGNERS, LosoFold, load_all_folds, load_fold
from .sequence_dataset import (
    SequenceContractError,
    SequenceRecord,
    VirtualGloveSequenceDataset,
    build_feature_tensor,
    load_index,
    load_sequence_arrays,
    relative_to_first_valid,
    verify_sensor_layout,
)

__all__ = [
    "BatchError", "CONTRACT_VERSION", "FEATURE_SETS", "FOLD_SIGNERS", "HAND_ORDER",
    "LosoFold", "NUM_CLASSES", "QUATERNION_POLICIES", "SequenceContractError",
    "SequenceInputConfig", "SequenceRecord", "VirtualGloveSequenceDataset",
    "build_feature_tensor", "channel_index", "channel_names", "collate_sequences",
    "concat_features_and_masks", "contract_document", "family_slice",
    "feature_dimension", "hand_slice", "key_padding_mask", "load_all_folds",
    "load_fold", "load_index", "load_sequence_arrays", "make_collate_fn",
    "relative_to_first_valid", "verify_sensor_layout",
]
