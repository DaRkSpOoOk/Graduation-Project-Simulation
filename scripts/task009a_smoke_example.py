#!/usr/bin/env python3
"""TASK-009A handoff smoke example. No training, no optimizer, no model.

Builds one LOSO dataset, one DataLoader, pulls one batch, prints the shapes and
lengths, and verifies the three-way distinction between a real observation, an
invalid sensor and batch padding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.data import (  # noqa: E402
    SequenceInputConfig,
    VirtualGloveSequenceDataset,
    channel_names,
    key_padding_mask,
    load_fold,
    load_index,
    make_collate_fn,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "datasets/manifests/karsl_core28_virtual_glove.csv")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "datasets/splits")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--held-out-signer", default="01")
    parser.add_argument("--feature-set", default="full", choices=("bend_only", "bend_spread", "full"))
    parser.add_argument("--quaternion-policy", default="absolute",
                        choices=("absolute", "relative_first_valid"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    records = load_index(args.index)
    fold = load_fold(args.splits_dir, args.held_out_signer, records)
    config = SequenceInputConfig(feature_set=args.feature_set,
                                 quaternion_policy=args.quaternion_policy)
    dataset = VirtualGloveSequenceDataset(fold.roles["train"], args.run_root, config)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=make_collate_fn(config))
    batch = next(iter(loader))

    print(f"contract          : {config.to_dict()['contract_version']}")
    print(f"fold              : held-out signer {args.held_out_signer}, {fold.counts()}")
    print(f"feature_set       : {config.feature_set}  (D = {config.feature_dim})")
    print(f"quaternion_policy : {config.quaternion_policy}")
    print()
    for name in ("values", "feature_valid", "hand_present", "frame_valid", "lengths", "labels"):
        tensor = batch[name]
        print(f"{name:<16}: {tuple(tensor.shape)}  {tensor.dtype}")
    print(f"{'sample_ids':<16}: {len(batch['sample_ids'])} strings")
    print()
    print("lengths           :", batch["lengths"].tolist())
    print("labels            :", batch["labels"].tolist())
    print("signers           :", sorted(set(batch["signer_ids"])))

    frame_valid = batch["frame_valid"]
    feature_valid = batch["feature_valid"]
    values = batch["values"]
    padded = int((~frame_valid).sum())
    assert torch.equal(frame_valid.sum(dim=1), batch["lengths"]), "frame_valid disagrees with lengths"
    assert not feature_valid[~frame_valid].any(), "a padded step is marked as a valid measurement"
    assert torch.equal(key_padding_mask(batch), ~frame_valid)
    real_zero = int(((values == 0.0) & feature_valid).sum())
    filled_zero = int(((values == 0.0) & ~feature_valid).sum())
    print()
    print(f"padded timesteps               : {padded}")
    print(f"zeros that ARE measurements    : {real_zero}   (feature_valid=True)")
    print(f"zeros that are placeholders    : {filled_zero}   (feature_valid=False)")
    print("masks verified                 : OK")
    print()
    print("first three channel names      :", channel_names(config.feature_set)[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
