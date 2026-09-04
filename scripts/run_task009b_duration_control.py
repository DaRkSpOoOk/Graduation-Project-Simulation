#!/usr/bin/env python3
"""TASK-009B Duration Control A: the length-only 28-class baseline.

Fits a scikit-learn logistic regression on sequence length alone, per LOSO fold,
using TRAIN only. Test labels never influence the fit. Cheap: no GPU, seconds.

The number this produces is the floor a real recognizer must clear before its
accuracy can be attributed to hand dynamics rather than to signing tempo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.data import load_all_folds, load_index  # noqa: E402
from recognition.training import LengthOnlyClassifier, oracle_accuracy_from_length  # noqa: E402

DEFAULT_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task009b-core28-lstm")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path,
                        default=ROOT / "datasets/manifests/karsl_core28_virtual_glove.csv")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "datasets/splits")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    records = load_index(args.index)
    folds = load_all_folds(args.splits_dir, records)
    payload: dict[str, object] = {
        "schema_version": "task009b_duration_control_v1",
        "control": "A",
        "description": "28-class classification from sequence length alone",
        "fit_policy": "fitted on TRAIN only; test labels never used in any role",
        "folds": {},
    }
    for signer, fold in sorted(folds.items()):
        lengths = {role: [r.sequence_length for r in fold.roles[role]] for role in fold.roles}
        labels = {role: [r.label_index for r in fold.roles[role]] for role in fold.roles}
        model = LengthOnlyClassifier(seed=args.seed).fit(lengths["train"], labels["train"])
        entry = {
            "model": model.to_dict(),
            "train": model.evaluate(lengths["train"], labels["train"]),
            "validation": model.evaluate(lengths["validation"], labels["validation"]),
            "test": model.evaluate(lengths["test"], labels["test"]),
            "oracle_class_from_length_test": oracle_accuracy_from_length(
                lengths["test"], labels["test"]),
            "oracle_signer_from_length_all": oracle_accuracy_from_length(
                [r.sequence_length for r in records], [r.signer_id for r in records]),
            "chance_accuracy": 1.0 / 28,
        }
        for block in ("train", "validation", "test"):
            entry[block].pop("confusion_matrix", None)
        payload["folds"][signer] = entry
        print(f"fold S{signer}: test acc {entry['test']['accuracy']*100:6.2f}%  "
              f"macroF1 {entry['test']['macro_f1']:.4f}  "
              f"(chance {100/28:.2f}%, in-sample oracle "
              f"{entry['oracle_class_from_length_test']*100:.2f}%)", flush=True)

    destination = args.run_root / "duration_controls"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "control_a_length_only.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nRESULT JSON: {output}")
    print("STATUS:")
    print("COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
