"""
train.py
--------
Reproducible training + model comparison for the three moderation tasks.

For every task:
  1. Load the public dataset (see data/README.md), de-duplicate, then make a
     stratified 80/20 train/test split with a fixed seed.
  2. Preprocessing ablation (legacy vs enhanced cleaning) using 5-fold CV on
     the TRAINING split only; the better variant is used for that task.
  3. Compare Logistic Regression, Multinomial Naive Bayes and Linear SVC
     (calibrated, so it yields probabilities). Hyper-parameters are tuned with
     5-fold stratified CV on the training split, scoring macro-F1.
  4. Select the best model by CV macro-F1 ONLY. The held-out test set is not
     used for any selection decision; it is scored once per candidate to report
     results.
  5. Choose an uncertainty threshold from out-of-fold training predictions.
  6. Save the selected pipelines + metadata and write experiment reports.

Usage (from backend/):
    python train.py

The TF-IDF vocabulary lives inside each Pipeline, so it is learned from
training folds only (no leakage), and the same preprocessing function is
applied at training time and at inference time.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from functools import partial

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import evaluate as ev
from data_loading import (
    POSITIVE_CLASS, SPLIT_SEED, TASK_LABELS, TEST_SIZE, file_sha256, load_task, train_test,
)
from model import MODEL_PATH, TASKS
from preprocess import PREPROCESSORS, ensure_nltk_resources, get_preprocessor

MODEL_VERSION = "2.0.0"
CV_FOLDS = 5
SEED = SPLIT_SEED


def tfidf(variant: str) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=get_preprocessor(variant),
        ngram_range=(1, 2), min_df=2, max_features=30_000, sublinear_tf=True,
    )


def candidates(variant: str) -> dict:
    """name -> (pipeline, parameter grid)"""
    return {
        "Logistic Regression": (
            Pipeline([("tfidf", tfidf(variant)),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            {"clf__C": [0.5, 1.0, 3.0, 10.0]},
        ),
        "Multinomial Naive Bayes": (
            Pipeline([("tfidf", tfidf(variant)), ("clf", MultinomialNB())]),
            {"clf__alpha": [0.05, 0.1, 0.5, 1.0]},
        ),
        "Linear SVC (calibrated)": (
            Pipeline([("tfidf", tfidf(variant)),
                      ("clf", CalibratedClassifierCV(
                          LinearSVC(class_weight="balanced"), cv=3))]),
            {"clf__estimator__C": [0.05, 0.1, 0.5, 1.0]},
        ),
    }


def cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def run_task(task: str, ablation_rows: list, comparison_rows: list):
    print(f"\n=== {task} ===")
    df = load_task(task)
    X_tr, X_te, y_tr, y_te = train_test(task)
    labels, pos = TASK_LABELS[task], POSITIVE_CLASS[task]
    print(f"  samples after de-duplication: {len(df)} "
          f"(train {len(X_tr)}, test {len(X_te)}); classes: {df.label.value_counts().to_dict()}")

    # ---- 2. preprocessing ablation (baseline LR, CV on train only) --------
    scores = {}
    for variant in PREPROCESSORS:
        base = Pipeline([("tfidf", tfidf(variant)),
                         ("clf", LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced"))])
        s = cross_val_score(base, X_tr, y_tr, cv=cv(), scoring="f1_macro")
        scores[variant] = float(s.mean())
        ablation_rows.append({"task": task, "preprocessing": variant,
                              "cv_macro_f1_mean": round(s.mean(), 4), "cv_macro_f1_std": round(s.std(), 4)})
        print(f"  preprocessing={variant:<8} CV macro-F1 = {s.mean():.4f} ± {s.std():.4f}")
    variant = max(scores, key=lambda v: (scores[v], v == "enhanced"))
    print(f"  -> using '{variant}' preprocessing for {task}")

    # ---- 3/4. model comparison + selection by CV ---------------------------
    fitted = {}
    for name, (pipe, grid) in candidates(variant).items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv(), n_jobs=1, refit=True)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        pred = best.predict(X_te)
        m = ev.compute_metrics(y_te, pred, labels, pos)
        fitted[name] = (best, gs.best_score_, gs.best_params_, m)
        comparison_rows.append({
            "task": task, "model": name, "preprocessing": variant,
            "best_params": json.dumps(gs.best_params_),
            "cv_macro_f1": round(gs.best_score_, 4),
            "test_accuracy": round(m["accuracy"], 4),
            "test_macro_f1": round(m["macro_f1"], 4),
            "test_weighted_f1": round(m["weighted_f1"], 4),
            "key_class": pos,
            "test_key_class_precision": round(m["positive_precision"], 4),
            "test_key_class_recall": round(m["positive_recall"], 4),
            "test_key_class_f1": round(m["positive_f1"], 4),
        })
        print(f"  {name:<26} CV F1={gs.best_score_:.4f}  test acc={m['accuracy']:.4f} "
              f"macroF1={m['macro_f1']:.4f}  {pos} recall={m['positive_recall']:.3f}  "
              f"({time.time() - t0:.0f}s)")

    selected = max(fitted, key=lambda n: fitted[n][1])      # CV score only
    best, cv_score, params, test_metrics = fitted[selected]
    print(f"  -> selected '{selected}' by CV macro-F1 ({cv_score:.4f})")

    # ---- 5. uncertainty threshold from out-of-fold TRAIN predictions -------
    oof = cross_val_predict(best, X_tr, y_tr, cv=cv(), method="predict_proba")
    thr = ev.choose_uncertainty_threshold(y_tr, oof, best.classes_)
    proba_te = best.predict_proba(X_te)
    thr_test = ev.threshold_check_on_test(y_te, proba_te, best.classes_, thr["threshold"])
    print(f"  uncertainty threshold = {thr['threshold']} ({thr['rule']})")

    info = {
        "selected_model": selected,
        "preprocessing": variant,
        "best_params": params,
        "cv_macro_f1": cv_score,
        "test_metrics": test_metrics,
        "uncertainty_threshold": thr["threshold"],
        "uncertainty_rule": thr["rule"],
        "uncertainty_oof_table": thr["table"],
        "uncertainty_test_check": thr_test,
        "dataset": {
            "file": __import__("data_loading").DATASET_FILES[task],
            "sha256": file_sha256(task),
            "n_after_dedup": int(len(df)),
            "n_train": len(X_tr), "n_test": len(X_te),
            "class_counts": {k: int(v) for k, v in df.label.value_counts().items()},
        },
    }
    return best, info


def main() -> None:
    ensure_nltk_resources(download=False)
    ablation_rows, comparison_rows, models, tasks_meta = [], [], {}, {}

    for task in TASKS:
        models[task], tasks_meta[task] = run_task(task, ablation_rows, comparison_rows)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "split_seed": SEED, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS,
        "scoring": "f1_macro",
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
        "tasks": tasks_meta,
    }

    # ---- 6. persist ---------------------------------------------------------
    MODEL_PATH_DIR = __import__("os").path.dirname(MODEL_PATH)
    __import__("os").makedirs(MODEL_PATH_DIR, exist_ok=True)
    joblib.dump({"models": models, "metadata": metadata}, MODEL_PATH, compress=3)
    (ev.EXPERIMENTS_DIR).mkdir(exist_ok=True)
    (ev.EXPERIMENTS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    comp = pd.DataFrame(comparison_rows)
    comp.to_csv(ev.EXPERIMENTS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(ev.EXPERIMENTS_DIR / "preprocessing_ablation.csv", index=False)
    ev.plot_model_comparison(comp, ev.FIGURES_DIR / "model_comparison.png")
    print(f"\nSaved model -> {MODEL_PATH}")

    ev.evaluate_saved_models()      # confusion matrices, error analysis, test_metrics.json


if __name__ == "__main__":
    main()
