#!/usr/bin/env python3
"""TASK-009B: analyse the completed sweep and emit the final tables and plots.

Reads only persisted artifacts. Trains nothing, mutates no checkpoint, and never
touches the frozen TASK-008 production tree.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.analysis import (  # noqa: E402
    EXPECTED_MATRIX,
    FOLDS,
    PRIMARY_CONFIG,
    audit_run_root,
    confusion_pairs,
    confusion_universality,
    fold_table,
    history_table,
    paired_class_delta,
    paired_delta,
    per_class_table,
    summary_table,
)
from recognition.data.contract import CONTRACT_VERSION  # noqa: E402
from recognition.data.labels import load_label_table  # noqa: E402

DEFAULT_RUN_ROOT = Path("/home/hatim/graduation-project-runs/task009b-core28-lstm")
# One-factor contrasts: each pair differs in exactly one design decision.
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("bend_spread", "bend_only", "spread channels added"),
    ("full_absolute_masked_mean", "bend_spread", "palm orientation added"),
    ("full_absolute_masked_mean", "bend_only", "full glove vs bend only"),
    ("full_relative_masked_mean", "full_absolute_masked_mean", "relative vs absolute quaternion"),
    ("full_absolute_final_hidden", "full_absolute_masked_mean", "final_hidden vs masked_mean"),
)
PALETTE = {"01": "#4C78A8", "02": "#F58518", "03": "#54A24B"}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
    })


def plot_headline(folds: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    configs = [entry["config"] for entry in EXPECTED_MATRIX]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for axis, metric, title in (
        (axes[0], "test_accuracy", "Held-out signer accuracy"),
        (axes[1], "test_macro_f1", "Held-out signer macro F1"),
    ):
        positions = np.arange(len(configs))
        width = 0.26
        for offset, fold in enumerate(FOLDS):
            values = [float(folds[(folds.config == c) & (folds.fold == fold)][metric].iloc[0]) * 100
                      for c in configs]
            axis.bar(positions + (offset - 1) * width, values, width,
                     label=f"held-out S{fold}", color=PALETTE[fold])
        means = [float(summary[summary.config == c][f"{metric}_mean"].iloc[0]) * 100
                 for c in configs]
        axis.plot(positions, means, "k_", markersize=26, markeredgewidth=2.0,
                  label="mean over folds")
        axis.axhline(100 / 28, color="#B0B0B0", linestyle="--", linewidth=1,
                     label="28-way chance")
        axis.set_xticks(positions)
        axis.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=7.5)
        axis.set_ylabel("percent")
        axis.set_title(title)
        axis.set_ylim(0, 100)
    axes[0].legend(fontsize=7.5, loc="upper left")
    figure.suptitle("TASK-009B Core-28 LSTM baseline — leave-one-signer-out (seed 1337)",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_pooling_control(folds: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    subset = folds[folds.config.isin(["full_absolute_masked_mean", "full_absolute_final_hidden"])]
    positions = np.arange(len(FOLDS))
    for offset, (config, colour, label) in enumerate((
        ("full_absolute_masked_mean", "#4C78A8", "masked_mean (primary)"),
        ("full_absolute_final_hidden", "#E45756", "final_hidden (Control B)"),
    )):
        values = [float(subset[(subset.config == config) & (subset.fold == f)]
                        ["test_accuracy"].iloc[0]) * 100 for f in FOLDS]
        axis.bar(positions + (offset - 0.5) * 0.36, values, 0.36, color=colour, label=label)
    axis.set_xticks(positions)
    axis.set_xticklabels([f"S{f}\nlen ratio "
                          f"{float(folds[(folds.config==PRIMARY_CONFIG)&(folds.fold==f)]['test_mean_length'].iloc[0]) / float(folds[(folds.config==PRIMARY_CONFIG)&(folds.fold==f)]['train_mean_length'].iloc[0]):.2f}"
                          for f in FOLDS])
    axis.set_ylabel("held-out accuracy (%)")
    axis.set_title("Control B — pooling under duration shift\nS02 is the severely shifted fold")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_training_curves(history: pd.DataFrame, path: Path) -> None:
    configs = [entry["config"] for entry in EXPECTED_MATRIX]
    figure, axes = plt.subplots(2, len(configs), figsize=(17, 6), sharex=True)
    for column, config in enumerate(configs):
        for fold in FOLDS:
            block = history[(history.config == config) & (history.fold == fold)]
            axes[0, column].plot(block.epoch, block.train_loss, color=PALETTE[fold],
                                 linewidth=1.1, label=f"S{fold} train")
            axes[0, column].plot(block.epoch, block.validation_loss, color=PALETTE[fold],
                                 linewidth=1.1, linestyle="--")
            axes[1, column].plot(block.epoch, block.validation_macro_f1, color=PALETTE[fold],
                                 linewidth=1.2, label=f"held-out S{fold}")
        axes[0, column].set_title(config.replace("_", "\n"), fontsize=8)
        axes[1, column].set_xlabel("epoch")
        axes[1, column].set_ylim(0, 1.02)
    axes[0, 0].set_ylabel("loss (solid train, dashed val)")
    axes[1, 0].set_ylabel("validation macro F1")
    axes[1, 0].legend(fontsize=7)
    figure.suptitle("TASK-009B training dynamics — validation uses non-held-out signers", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_class_deltas(per_class: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for axis, (treatment, baseline, title) in zip(axes, (
        ("bend_spread", "bend_only", "spread added (bend_spread − bend_only)"),
        ("full_absolute_masked_mean", "bend_spread", "orientation added (full − bend_spread)"),
    )):
        delta = paired_class_delta(per_class, treatment, baseline).sort_values("mean_f1_delta")
        colours = ["#E45756" if v < 0 else "#54A24B" for v in delta.mean_f1_delta]
        axis.barh(delta.label_ar.astype(str), delta.mean_f1_delta, color=colours)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("mean per-class F1 change (over 3 folds)")
        axis.set_title(title, fontsize=9)
        axis.tick_params(axis="y", labelsize=8)
    figure.suptitle("TASK-009B per-class sensor ablation effects", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_domain_gap(folds: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    configs = [entry["config"] for entry in EXPECTED_MATRIX]
    for fold in FOLDS:
        block = folds[folds.fold == fold].set_index("config").reindex(configs)
        axis.plot(range(len(configs)), block.best_validation_macro_f1, "o--",
                  color=PALETTE[fold], alpha=0.55, markersize=5)
        axis.plot(range(len(configs)), block.test_macro_f1, "o-",
                  color=PALETTE[fold], markersize=6, label=f"held-out S{fold}")
    axis.set_xticks(range(len(configs)))
    axis.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=7.5)
    axis.set_ylabel("macro F1")
    axis.set_ylim(0, 1.02)
    axis.set_title("Validation (dashed, seen signers) vs held-out signer test (solid)\n"
                   "the gap is cross-signer generalization, not ordinary overfitting",
                   fontsize=9.5)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_primary_confusion(run_root: Path, labels_ar: dict[int, str], path: Path) -> None:
    from recognition.analysis import load_result
    entry = next(e for e in EXPECTED_MATRIX if e["config"] == PRIMARY_CONFIG)
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    names = [labels_ar[i] for i in range(28)]
    for axis, fold in zip(axes, FOLDS):
        matrix = np.asarray(load_result(run_root, entry, fold)["test_metrics"]["confusion_matrix"],
                            dtype=np.float64)
        normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        image = axis.imshow(normalized, cmap="magma", vmin=0, vmax=1)
        axis.set_xticks(range(28)); axis.set_yticks(range(28))
        axis.set_xticklabels(names, fontsize=6); axis.set_yticklabels(names, fontsize=6)
        axis.set_title(f"held-out S{fold}", fontsize=9)
        axis.set_xlabel("predicted"); axis.set_ylabel("true")
        axis.grid(False)
    figure.colorbar(image, ax=axes, fraction=0.015, label="row-normalized rate")
    figure.suptitle(f"TASK-009B primary model ({PRIMARY_CONFIG}) — test confusion", fontsize=11)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--labels", type=Path,
                        default=ROOT / "datasets/manifests/karsl_core28_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/recognition")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "reports/recognition/figures")
    args = parser.parse_args(argv)

    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    labels_ar = {index: label.label_ar
                 for index, label in load_label_table(args.labels).items()}

    audit = audit_run_root(args.run_root)
    print(f"[audit] {audit['complete_experiments']}/{audit['expected_experiments']} complete, "
          f"{audit['problem_count']} problems", flush=True)
    if not audit["all_complete"]:
        print("[audit] REFUSING to write a results report over an incomplete sweep.")
        (args.output_dir / "TASK-009B-completeness-audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        return 1

    folds = fold_table(args.run_root)
    summary = summary_table(folds)
    per_class = per_class_table(args.run_root, labels_ar)
    history = history_table(args.run_root)

    folds.to_csv(args.output_dir / "TASK-009B-results-by-fold.csv", index=False)
    per_class.to_csv(args.output_dir / "TASK-009B-results-by-class.csv", index=False)

    ablation_rows = [paired_delta(folds, treatment, baseline).assign(contrast=name)
                     for treatment, baseline, name in CONTRASTS]
    ablation = pd.concat(ablation_rows, ignore_index=True)
    ablation.to_csv(args.output_dir / "TASK-009B-sensor-ablation.csv", index=False)

    pooling = folds[folds.config.isin(["full_absolute_masked_mean",
                                       "full_absolute_final_hidden"])].copy()
    pooling["length_shift_ratio"] = pooling.test_mean_length / pooling.train_mean_length
    pooling[["config", "fold", "length_shift_ratio", "train_mean_length", "test_mean_length",
             "best_validation_macro_f1", "test_accuracy", "test_macro_f1", "best_epoch"]].to_csv(
        args.output_dir / "TASK-009B-duration-pooling-analysis.csv", index=False)

    class_deltas = pd.concat(
        [paired_class_delta(per_class, treatment, baseline).assign(contrast=name)
         for treatment, baseline, name in CONTRASTS], ignore_index=True)
    class_deltas.to_csv(args.output_dir / "TASK-009B-class-ablation.csv", index=False)

    pairs = confusion_pairs(args.run_root, PRIMARY_CONFIG, labels_ar=labels_ar)
    pairs.to_csv(args.output_dir / "TASK-009B-top-confusions.csv", index=False)

    plot_headline(folds, summary, args.figure_dir / "task009b-headline.png")
    plot_pooling_control(folds, args.figure_dir / "task009b-pooling-control.png")
    plot_training_curves(history, args.figure_dir / "task009b-training-curves.png")
    plot_class_deltas(per_class, args.figure_dir / "task009b-class-ablation.png")
    plot_domain_gap(folds, args.figure_dir / "task009b-domain-gap.png")
    plot_primary_confusion(args.run_root, labels_ar,
                           args.figure_dir / "task009b-primary-confusion.png")

    control_path = args.run_root / "duration_controls" / "control_a_length_only.json"
    control = json.loads(control_path.read_text(encoding="utf-8")) if control_path.is_file() else None

    special = (per_class[per_class.label_ar.isin(["ض", "ي", "ص"])]
               .pivot_table(index="label_ar", columns="config", values="f1",
                            aggfunc="mean", observed=True))

    def config_block(name: str) -> dict:
        row = summary[summary.config == name].iloc[0]
        return {
            "test_accuracy_mean": float(row.test_accuracy_mean),
            "test_accuracy_std_across_folds": float(row.test_accuracy_std),
            "test_macro_f1_mean": float(row.test_macro_f1_mean),
            "test_macro_f1_std_across_folds": float(row.test_macro_f1_std),
            "per_fold": {
                r.fold: {"accuracy": float(r.test_accuracy), "macro_f1": float(r.test_macro_f1)}
                for r in folds[folds.config == name].itertuples()
            },
            "domain_gap_macro_f1_mean": float(row.domain_gap_macro_f1_mean),
        }

    payload = {
        "schema_version": "task009b_results_summary_v1",
        "contract_version": CONTRACT_VERSION,
        "run_root": str(args.run_root),
        "seed": 1337,
        "folds": list(FOLDS),
        "primary_config": PRIMARY_CONFIG,
        "primary_config_preregistered": True,
        "completeness": {k: v for k, v in audit.items() if k != "experiments"},
        "results": {entry["config"]: config_block(entry["config"]) for entry in EXPECTED_MATRIX},
        "contrasts": {
            name: {
                "treatment": treatment, "baseline": baseline,
                "per_fold_accuracy_delta_pp": {
                    r.fold: float(r.accuracy_delta_pp)
                    for r in paired_delta(folds, treatment, baseline).itertuples()},
                "mean_accuracy_delta_pp": float(
                    paired_delta(folds, treatment, baseline).accuracy_delta_pp.mean()),
                "mean_macro_f1_delta_points": float(
                    paired_delta(folds, treatment, baseline).macro_f1_delta_points.mean()),
                "folds_improved": int(
                    (paired_delta(folds, treatment, baseline).accuracy_delta_pp > 0).sum()),
                "classes_improved_of_28": int(
                    (paired_class_delta(per_class, treatment, baseline).mean_f1_delta > 0).sum()),
            }
            for treatment, baseline, name in CONTRASTS
        },
        "duration_shift": {
            fold: {
                "train_mean_length": float(
                    folds[(folds.config == PRIMARY_CONFIG) & (folds.fold == fold)]
                    .train_mean_length.iloc[0]),
                "test_mean_length": float(
                    folds[(folds.config == PRIMARY_CONFIG) & (folds.fold == fold)]
                    .test_mean_length.iloc[0]),
            } for fold in FOLDS
        },
        "duration_control_a": control,
        "domain_gap": {
            "overall_mean_macro_f1": float(folds.domain_gap_macro_f1.mean()),
            "by_fold_mean": {k: float(v) for k, v in
                             folds.groupby("fold").domain_gap_macro_f1.mean().items()},
            "by_config_mean": {k: float(v) for k, v in
                               folds.groupby("config", observed=True)
                               .domain_gap_macro_f1.mean().items()},
        },
        "confusion_universality": confusion_universality(pairs),
        "low_spread_validity_classes": {
            label: {config: float(value) for config, value in row.items()}
            for label, row in special.iterrows()
        },
        "runtime": {
            "total_wall_seconds": float(folds.wall_seconds.sum()),
            "mean_seconds_per_epoch": float(folds.seconds_per_epoch.mean()),
            "peak_gpu_mib": float(folds.peak_gpu_mib.max()),
            "parameter_count": int(folds.parameter_count.max()),
        },
        "statistical_note": (
            "3 LOSO folds and ONE seed (1337). The reported standard deviation is across "
            "held-out signers and therefore measures signer variation, not initialization "
            "or training variance. No significance test is reported: with n=3 folds and no "
            "seed replication, a p-value would be theatre."
        ),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True, check=False).stdout.strip(),
    }
    summary_path = args.output_dir / "TASK-009B-results-summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                            encoding="utf-8")
    (args.output_dir / "TASK-009B-completeness-audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[write] {summary_path}")
    print(f"[write] {args.output_dir}/TASK-009B-results-by-fold.csv (+ by-class, ablation, "
          f"pooling, confusions)")
    print(f"[write] {args.figure_dir}/*.png (6 figures)")
    print("STATUS:\nCOMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
