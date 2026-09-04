"""TASK-009B: a deliberately modest LSTM baseline over the frozen TASK-009A input.

The scientific question is representation quality, not leaderboard performance,
so this is an ordinary unidirectional LSTM with a linear head. No transformer, no
convolutional front end, no attention, no architecture search.

The one non-obvious design choice is temporal pooling. Sequence duration in this
corpus identifies the *signer* far better than it identifies the *letter*
(TASK-009A measured 78.1% vs 7.6% oracle accuracy from length alone), and an
LSTM's final hidden state is implicitly a function of how many steps it consumed.
The primary pooling is therefore a masked mean over real source frames, which
weights every observed frame equally and is invariant to how many of them there
are. ``final_hidden`` is retained as a controlled comparison, not as the default.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal, Mapping

import torch
from torch import nn

from ..data.contract import CONTRACT_VERSION, NUM_CLASSES, feature_dimension

Pooling = Literal["masked_mean", "final_hidden"]
InputPolicy = Literal["values_and_feature_valid", "values_only"]

POOLING_POLICIES: tuple[str, ...] = ("masked_mean", "final_hidden")
INPUT_POLICIES: tuple[str, ...] = ("values_and_feature_valid", "values_only")


def model_input_dimension(feature_set: str, input_policy: str) -> int:
    """Channels the model actually receives.

    ``values_and_feature_valid`` doubles the contract dimension because a numeric
    zero cannot by itself express "no measurement": the validity flag has to be a
    real input channel, not an implicit convention.
    """

    if input_policy not in INPUT_POLICIES:
        raise ValueError(f"input_policy must be one of {INPUT_POLICIES}")
    base = feature_dimension(feature_set)
    return 2 * base if input_policy == "values_and_feature_valid" else base


@dataclass(frozen=True)
class LSTMBaselineConfig:
    """Everything needed to rebuild this model from a checkpoint."""

    feature_set: str = "full"
    input_policy: InputPolicy = "values_and_feature_valid"
    pooling: Pooling = "masked_mean"
    hidden_size: int = 192
    num_layers: int = 2
    dropout: float = 0.3
    # An explicit projection before the LSTM is redundant with the LSTM's own
    # input-to-hidden matrix, so the default is None. Kept configurable because
    # the brief asks for an optional projection, not because it is expected to
    # help.
    input_projection: int | None = None
    bidirectional: bool = False
    num_classes: int = NUM_CLASSES
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.pooling not in POOLING_POLICIES:
            raise ValueError(f"pooling must be one of {POOLING_POLICIES}")
        if self.input_policy not in INPUT_POLICIES:
            raise ValueError(f"input_policy must be one of {INPUT_POLICIES}")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def input_dim(self) -> int:
        return model_input_dimension(self.feature_set, self.input_policy)

    @property
    def pooled_dim(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model_input(batch: Mapping[str, Any], input_policy: str) -> torch.Tensor:
    """Assemble the model input from a frozen TASK-009A batch.

    ``hand_present`` is deliberately NOT fed: per-channel ``feature_valid``
    already carries strictly more information about what is missing, and in this
    corpus a hand's presence is exactly equal to its channels being valid, so
    adding it would be two duplicated constant-correlated channels.
    ``tracking_state_code`` stays diagnostics-only.
    """

    values = batch["values"]
    if input_policy == "values_only":
        return values
    if input_policy != "values_and_feature_valid":
        raise ValueError(f"input_policy must be one of {INPUT_POLICIES}")
    return torch.cat((values, batch["feature_valid"].to(values.dtype)), dim=-1)


def masked_mean(outputs: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
    """Mean of ``outputs`` over real frames only.

    Padding contributes exactly zero weight -- it is zeroed before summing and
    excluded from the denominator -- so a batch's padding amount cannot change
    any sequence's representation.
    """

    mask = frame_valid.unsqueeze(-1).to(outputs.dtype)
    summed = (outputs * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class LSTMBaseline(nn.Module):
    """Masked-input LSTM classifier over variable-length glove sequences."""

    def __init__(self, config: LSTMBaselineConfig | None = None) -> None:
        super().__init__()
        self.config = config or LSTMBaselineConfig()
        lstm_input = self.config.input_dim
        self.projection: nn.Module | None = None
        if self.config.input_projection:
            self.projection = nn.Sequential(
                nn.Linear(self.config.input_dim, self.config.input_projection), nn.ReLU()
            )
            lstm_input = self.config.input_projection
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=self.config.bidirectional,
            # nn.LSTM only applies dropout between stacked layers.
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(self.config.pooled_dim, self.config.num_classes)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        features = build_model_input(batch, self.config.input_policy)
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"batch input dimension {features.shape[-1]} != model input dimension "
                f"{self.config.input_dim} (feature_set={self.config.feature_set!r}, "
                f"input_policy={self.config.input_policy!r})"
            )
        if self.projection is not None:
            features = self.projection(features)

        lengths = batch["lengths"].to("cpu", dtype=torch.int64)
        packed = nn.utils.rnn.pack_padded_sequence(
            features, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, (hidden, _) = self.lstm(packed)

        if self.config.pooling == "masked_mean":
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=features.shape[1]
            )
            pooled = masked_mean(outputs, batch["frame_valid"])
        else:
            # h_n is [layers * directions, B, H]; the packed LSTM already returns
            # the state at each sequence's own final REAL step, so this is never
            # contaminated by padding.
            if self.config.bidirectional:
                pooled = torch.cat((hidden[-2], hidden[-1]), dim=-1)
            else:
                pooled = hidden[-1]
        return self.classifier(self.dropout(pooled))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = [
    "POOLING_POLICIES", "INPUT_POLICIES", "LSTMBaselineConfig", "LSTMBaseline",
    "build_model_input", "masked_mean", "model_input_dimension",
]
