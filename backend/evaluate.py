"""
evaluate.py
-----------
Held-out evaluation for the moderation classifiers.

Metrics per task: accuracy, macro/weighted F1, per-class precision / recall /
F1 / support and the confusion matrix. Accuracy is reported but never used
alone - with imbalanced classes it hides exactly the failures that matter.

Library use (from train.py)::
    compute_metrics(y_true, y_pred, labels, positive_class)

CLI - re-evaluate the SAVED model on the same deterministic held-out split and
regenerate figures, per-class reports and error-analysis files::

    cd backend && python evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
FIGURES_DIR = ROOT / "reports" / "figures"


def compute_metrics(y_true, y_pred, labels, positive_class=None) -> dict:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        lab: {"precision": float(p[i]), "recall": float(r[i]),
              "f1": float(f[i]), "support": int(s[i])}
        for i, lab in enumerate(labels)
    }
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
    }
    if positive_class:
        out["positive_class"] = positive_class
        out["positive_precision"] = per_class[positive_class]["precision"]
        out["positive_recall"] = per_class[positive_class]["recall"]
        out["positive_f1"] = per_class[positive_class]["f1"]
    return out


def choose_uncertainty_threshold(y_true, proba, classes, target_acc=0.90,
                                 min_coverage=0.30) -> dict:
    """
    Pick a confidence threshold from *out-of-fold training predictions*.

    Smallest t in {0.50, 0.55, ... 0.95} such that predictions with top-class
    probability >= t are at least ``target_acc`` accurate while still covering
    at least ``min_coverage`` of the data. If no t qualifies, fall back to the
    t with the highest accuracy that satisfies the coverage floor.
    """
    classes = np.asarray(classes)
    pred = classes[np.argmax(proba, axis=1)]
    conf = proba.max(axis=1)
    correct = pred == np.asarray(y_true)
    rows = []
    for t in np.round(np.arange(0.50, 0.96, 0.05), 2):
        keep = conf >= t
        cov = float(keep.mean())
        acc = float(correct[keep].mean()) if keep.any() else float("nan")
        rows.append({"threshold": float(t), "coverage": cov, "accuracy": acc})
    ok = [r for r in rows if r["coverage"] >= min_coverage and r["accuracy"] >= target_acc]
    if ok:
        chosen, rule = ok[0], f"smallest t with accuracy>={target_acc} and coverage>={min_coverage}"
    else:
        cand = [r for r in rows if r["coverage"] >= min_coverage and not np.isnan(r["accuracy"])]
        chosen = max(cand, key=lambda r: r["accuracy"]) if cand else rows[0]
        rule = f"fallback: highest accuracy with coverage>={min_coverage} (target {target_acc} not reachable)"
    return {"threshold": chosen["threshold"], "rule": rule, "table": rows}


def threshold_check_on_test(y_true, proba, classes, threshold) -> dict:
    classes = np.asarray(classes)
    pred = classes[np.argmax(proba, axis=1)]
    conf = proba.max(axis=1)
    correct = pred == np.asarray(y_true)
    hi, lo = conf >= threshold, conf < threshold
    return {
        "threshold": threshold,
        "coverage_confident": float(hi.mean()),
        "accuracy_confident": float(correct[hi].mean()) if hi.any() else None,
        "accuracy_uncertain": float(correct[lo].mean()) if lo.any() else None,
    }


# ---------------------------------------------------------------- figures --

def plot_confusion_matrix(cm, labels, title, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(df["task"].unique())
    models = list(df["model"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.8 / len(models)
    for k, m in enumerate(models):
        vals = [df[(df.task == t) & (df.model == m)]["test_macro_f1"].iloc[0] for t in tasks]
        xs = np.arange(len(tasks)) + k * width
        bars = ax.bar(xs, vals, width, label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(tasks)) + width * (len(models) - 1) / 2, tasks)
    ax.set_ylabel("Macro F1 (held-out test set)"); ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on held-out test data"); ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------- error analysis --

def error_table(texts, y_true, y_pred, proba, classes, n=15) -> pd.DataFrame:
    """Most confident misclassifications (the most informative errors)."""
    classes = np.asarray(classes)
    conf = proba.max(axis=1)
    df = pd.DataFrame({"text": texts, "actual": y_true, "predicted": y_pred, "confidence": conf})
    wrong = df[df.actual != df.predicted].sort_values("confidence", ascending=False)
    return wrong.head(n).reset_index(drop=True)


# --------------------------------------------------------------- summary --

def write_summary(report: dict, meta: dict) -> None:
    """Generate experiments/evaluation_summary.md from MEASURED results only."""
    comp = pd.read_csv(EXPERIMENTS_DIR / "model_comparison.csv")
    abl = pd.read_csv(EXPERIMENTS_DIR / "preprocessing_ablation.csv")
    L = [
        "# Evaluation summary",
        "",
        "_Generated by `backend/evaluate.py` from the saved model - do not edit by hand._",
        "",
        f"- Model version: **{meta['model_version']}**, trained {meta['trained_at']}",
        f"- Split: stratified {int((1 - meta['test_size']) * 100)}/{int(meta['test_size'] * 100)} "
        f"train/test, seed {meta['split_seed']}; model selection by {meta['cv_folds']}-fold CV "
        f"macro-F1 on the training split only",
        f"- Environment: Python {meta['environment']['python']}, "
        f"scikit-learn {meta['environment']['scikit_learn']}",
        "",
        "## Datasets",
        "",
        "| Task | Source file | After de-dup | Train | Test | Class counts |",
        "|---|---|---:|---:|---:|---|",
    ]
    for task, t in meta["tasks"].items():
        d = t["dataset"]
        L.append(f"| {task} | `{d['file']}` | {d['n_after_dedup']} | {d['n_train']} | {d['n_test']} | "
                 f"{d['class_counts']} |")

    L += ["", "## Model comparison (held-out test set)", "",
          "| Task | Model | CV macro-F1 | Test accuracy | Test macro-F1 | Key class | Key-class precision | Key-class recall |",
          "|---|---|---:|---:|---:|---|---:|---:|"]
    for _, r in comp.iterrows():
        sel = " **(selected)**" if r["model"] == meta["tasks"][r["task"]]["selected_model"] else ""
        L.append(f"| {r['task']} | {r['model']}{sel} | {r['cv_macro_f1']:.3f} | {r['test_accuracy']:.3f} | "
                 f"{r['test_macro_f1']:.3f} | {r['key_class']} | {r['test_key_class_precision']:.3f} | "
                 f"{r['test_key_class_recall']:.3f} |")
    L += ["", "Selection used CV macro-F1 only; test scores are reported for all candidates but did not "
          "influence which model was chosen.", "",
          "![Model comparison](../reports/figures/model_comparison.png)", "",
          "## Preprocessing ablation (5-fold CV on training split, Logistic Regression C=3)", "",
          "| Task | Preprocessing | CV macro-F1 (mean ± std) |", "|---|---|---:|"]
    for _, r in abl.iterrows():
        L.append(f"| {r['task']} | {r['preprocessing']} | {r['cv_macro_f1_mean']:.4f} ± {r['cv_macro_f1_std']:.4f} |")

    L += ["", "## Per-class results for the deployed model (held-out test set)"]
    for task, m in report.items():
        L += ["", f"### {task} - {meta['tasks'][task]['selected_model']}", "",
              f"Accuracy {m['accuracy']:.3f} · macro-F1 {m['macro_f1']:.3f} · weighted-F1 {m['weighted_f1']:.3f}", "",
              "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        for lab, c in m["per_class"].items():
            L.append(f"| {lab} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['support']} |")
        u = m["uncertainty_check"]
        acc_c = "n/a" if u["accuracy_confident"] is None else f"{u['accuracy_confident']:.3f}"
        acc_u = "n/a (no predictions below threshold)" if u["accuracy_uncertain"] is None else f"{u['accuracy_uncertain']:.3f}"
        L += ["", f"Uncertainty threshold {u['threshold']} (chosen from out-of-fold training predictions): "
              f"{u['coverage_confident']:.1%} of test predictions are at/above it with accuracy {acc_c}; "
              f"accuracy of predictions below it: {acc_u}.", "",
              f"![Confusion matrix: {task}](../reports/figures/confusion_matrix_{task}.png)"]
    L += ["", "Most confident misclassifications per task are in `experiments/error_analysis_<task>.csv`. "
          "See `docs/methodology.md` for interpretation and `docs/limitations.md` for caveats.", ""]
    (EXPERIMENTS_DIR / "evaluation_summary.md").write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------- CLI --

def evaluate_saved_models() -> dict:
    """Re-evaluate the saved artifact on the deterministic held-out split."""
    from data_loading import POSITIVE_CLASS, TASK_LABELS, train_test
    from model import MODEL_PATH, load_models

    bundle = load_models(MODEL_PATH)
    models, meta = bundle["models"], bundle["metadata"]
    report = {}
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    for task, pipe in models.items():
        _, X_te, _, y_te = train_test(task, seed=meta["split_seed"], test_size=meta["test_size"])
        proba = pipe.predict_proba(X_te)
        pred = pipe.classes_[np.argmax(proba, axis=1)]
        m = compute_metrics(y_te, pred, TASK_LABELS[task], POSITIVE_CLASS[task])
        m["uncertainty_check"] = threshold_check_on_test(
            y_te, proba, pipe.classes_, meta["tasks"][task]["uncertainty_threshold"])
        report[task] = m

        plot_confusion_matrix(m["confusion_matrix"], m["labels"],
                              f"{task}: {meta['tasks'][task]['selected_model']}",
                              FIGURES_DIR / f"confusion_matrix_{task}.png")
        error_table(X_te, y_te, pred, proba, pipe.classes_).to_csv(
            EXPERIMENTS_DIR / f"error_analysis_{task}.csv", index=False)

        stored = meta["tasks"][task]["test_metrics"]["macro_f1"]
        print(f"[evaluate] {task:<9} macro-F1={m['macro_f1']:.4f} "
              f"(stored at train time: {stored:.4f})")

    (EXPERIMENTS_DIR / "test_metrics.json").write_text(json.dumps(report, indent=2))
    if (EXPERIMENTS_DIR / "model_comparison.csv").exists():
        write_summary(report, meta)
    return report


if __name__ == "__main__":
    evaluate_saved_models()
