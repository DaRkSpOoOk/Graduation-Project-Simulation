"""TASK-009A variable-length batching.

Original sequence length is preserved: nothing is cropped, resampled or fitted
to a fixed window. Batches are right-padded dense tensors plus an explicit
``frame_valid`` mask and integer ``lengths``, which is directly compatible with
``torch.nn.utils.rnn.pack_padded_sequence(..., enforce_sorted=False)``.

Three states stay distinct at every position:

* real observation      -- frame_valid=True,  feature_valid=True  (value may be 0.0)
* invalid/missing sensor-- frame_valid=True,  feature_valid=False
* batch padding         -- frame_valid=False, feature_valid=False
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

try:  # torch is required to build batches, but not to import the contract.
    import torch
except ImportError:  # pragma: no cover - exercised only in torch-less installs
    torch = None  # type: ignore[assignment]

from .contract import HAND_ORDER, SequenceInputConfig


class BatchError(ValueError):
    """A batch could not be assembled from the supplied items."""


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise BatchError("PyTorch is required to build batches")


def collate_sequences(
    items: Sequence[Mapping[str, Any]], config: SequenceInputConfig | None = None
) -> dict[str, Any]:
    """Collate variable-length sequences into one padded batch.

    Padding uses ``config.padding_fill_value`` and is marked False in
    ``frame_valid``, ``feature_valid`` and ``hand_present`` simultaneously, so a
    padded step can never be mistaken for an observed one no matter which mask a
    downstream model happens to consult.
    """

    _require_torch()
    config = config or SequenceInputConfig()
    if not items:
        raise BatchError("cannot collate an empty list of items")

    lengths = [int(item["length"]) for item in items]
    if any(length <= 0 for length in lengths):
        raise BatchError("every sequence must have at least one frame")
    dims = {int(np.asarray(item["values"]).shape[1]) for item in items}
    if len(dims) != 1:
        raise BatchError(f"items disagree on feature dimension: {sorted(dims)}")
    feature_dim = dims.pop()
    if feature_dim != config.feature_dim:
        raise BatchError(
            f"item feature dimension {feature_dim} != contract dimension {config.feature_dim} "
            f"for feature_set={config.feature_set!r}"
        )

    batch_size = len(items)
    max_length = max(lengths)
    num_hands = len(HAND_ORDER)

    values = np.full((batch_size, max_length, feature_dim), config.padding_fill_value, dtype=np.float32)
    feature_valid = np.zeros((batch_size, max_length, feature_dim), dtype=bool)
    hand_present = np.zeros((batch_size, max_length, num_hands), dtype=bool)
    frame_valid = np.zeros((batch_size, max_length), dtype=bool)
    tracking_state = np.zeros((batch_size, max_length, num_hands), dtype=np.int16)
    frame_index = np.full((batch_size, max_length), -1, dtype=np.int64)

    for row, item in enumerate(items):
        length = lengths[row]
        values[row, :length] = np.asarray(item["values"], dtype=np.float32)
        feature_valid[row, :length] = np.asarray(item["feature_valid"], dtype=bool)
        hand_present[row, :length] = np.asarray(item["hand_present"], dtype=bool)
        tracking_state[row, :length] = np.asarray(item["tracking_state_code"], dtype=np.int16)
        frame_index[row, :length] = np.asarray(item["frame_index"], dtype=np.int64)
        frame_valid[row, :length] = True

    batch: dict[str, Any] = {
        "values": torch.from_numpy(values),
        "feature_valid": torch.from_numpy(feature_valid),
        "hand_present": torch.from_numpy(hand_present),
        "frame_valid": torch.from_numpy(frame_valid),
        # int64 on CPU, unsorted: exactly what pack_padded_sequence expects with
        # enforce_sorted=False.
        "lengths": torch.tensor(lengths, dtype=torch.int64),
        "labels": torch.tensor([int(item["label_index"]) for item in items], dtype=torch.int64),
        "sample_ids": [str(item["sample_id"]) for item in items],
        "signer_ids": [str(item["signer_id"]) for item in items],
        "sign_ids": [str(item["sign_id"]) for item in items],
        "labels_ar": [str(item.get("label_ar", "")) for item in items],
        "frame_index": torch.from_numpy(frame_index),
        "feature_set": config.feature_set,
        "quaternion_policy": config.quaternion_policy,
    }
    if config.include_tracking_state:
        batch["tracking_state_code"] = torch.from_numpy(tracking_state)
    if not config.include_hand_present:
        batch.pop("hand_present")
    return batch


def make_collate_fn(config: SequenceInputConfig | None = None):
    """Return a ``collate_fn`` bound to one frozen configuration."""

    resolved = config or SequenceInputConfig()

    def _collate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return collate_sequences(items, resolved)

    return _collate


def concat_features_and_masks(batch: Mapping[str, Any]) -> "torch.Tensor":
    """Concatenate ``values`` with ``feature_valid`` along the channel axis.

    Masks are delivered as separate tensors by default so that ``feature_dim``
    keeps meaning exactly what the contract says and ablations stay clean. A
    model that prefers mask channels in its input calls this instead of the
    dataset growing a second code path: the result is ``[B, T, 2D]`` with the
    original channels first and their validity flags, in the same order, second.
    """

    _require_torch()
    return torch.cat((batch["values"], batch["feature_valid"].to(batch["values"].dtype)), dim=-1)


def key_padding_mask(batch: Mapping[str, Any]) -> "torch.Tensor":
    """``True`` where a step is padding -- the transformer/attention convention.

    Provided as an explicit derivation rather than as a second stored mask, so
    the contract keeps exactly one polarity and cannot ship an inverted twin.
    """

    _require_torch()
    return ~batch["frame_valid"]


__all__ = [
    "BatchError", "collate_sequences", "make_collate_fn",
    "concat_features_and_masks", "key_padding_mask",
]
