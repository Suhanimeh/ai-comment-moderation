"""
model.py
--------
Inference layer for the multi-task comment-moderation model.

Three independent scikit-learn pipelines (TF-IDF -> classifier), each trained
on its own dataset by ``train.py``:

  * toxicity  : Toxic / Non-Toxic
  * spam      : Spam / Not Spam
  * sentiment : Positive / Negative / Neutral

The saved artifact (``models/moderation_model.joblib``) is a dict
``{"models": {task: Pipeline}, "metadata": {...}}``. It is produced by
``python train.py`` - this module never trains on hidden seed data.

SECURITY: joblib/pickle files can execute code when loaded. Only load model
files you trained yourself or otherwise trust.

Probabilities
-------------
Each task returns the probability of ITS OWN predicted class. These are
model-estimated class probabilities, not guarantees of correctness, and the
three tasks are NOT combined into one score (averaging probabilities of
unrelated tasks has no defensible interpretation).
"""

from __future__ import annotations

import os

import joblib
import numpy as np

from preprocess import clean_text

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "moderation_model.joblib"),
)
TASKS = ("toxicity", "spam", "sentiment")
ALLOWED_LABELS = {
    "toxicity": {"Toxic", "Non-Toxic"},
    "spam": {"Spam", "Not Spam"},
    "sentiment": {"Positive", "Negative", "Neutral"},
}


class ModelNotAvailableError(RuntimeError):
    """The trained model artifact is missing or invalid."""


class InsufficientTextError(ValueError):
    """The comment contains no analysable text once cleaned."""


def load_models(path: str = MODEL_PATH) -> dict:
    """Load and sanity-check the trained artifact."""
    if not os.path.exists(path):
        raise ModelNotAvailableError(
            f"No trained model at {path}. Train one with:  "
            "python scripts/download_data.py && python scripts/setup_nltk.py "
            "&& cd backend && python train.py"
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "models" not in bundle or "metadata" not in bundle:
        raise ModelNotAvailableError(
            f"{path} is not a v2 model artifact (expected keys 'models' and "
            "'metadata'). Re-run python train.py."
        )
    for task in TASKS:
        if task not in bundle["models"]:
            raise ModelNotAvailableError(f"Model artifact is missing task '{task}'.")
        found = {str(c) for c in bundle["models"][task].classes_}
        if found != ALLOWED_LABELS[task]:
            raise ModelNotAvailableError(f"Task '{task}' has unexpected labels: {found}")
    return bundle


def predict_comment(comment: str, bundle: dict) -> dict:
    """
    Classify one comment.

    Raises InsufficientTextError if nothing usable remains after cleaning
    (e.g. punctuation-only input) rather than labelling it "safe".
    """
    if not clean_text(comment):
        raise InsufficientTextError("Comment contains no analysable text.")

    models, meta = bundle["models"], bundle["metadata"]
    labels, scores, all_scores = {}, {}, {}

    for task in TASKS:
        pipe = models[task]
        probs = pipe.predict_proba([comment])[0]
        idx = int(np.argmax(probs))
        label = str(pipe.classes_[idx])
        prob = float(probs[idx])
        if label not in ALLOWED_LABELS[task] or not (0.0 <= prob <= 1.0):
            raise RuntimeError(f"Invalid model output for task '{task}'")

        threshold = meta["tasks"][task]["uncertainty_threshold"]
        labels[task] = label
        scores[task] = {
            "label": label,
            "probability": round(prob, 4),
            "uncertain": prob < threshold,
        }
        all_scores[task] = {str(c): round(float(p), 4) for c, p in zip(pipe.classes_, probs)}

    return {
        "comment": comment,
        "status": "ok",
        "toxicity": labels["toxicity"],
        "spam": labels["spam"],
        "sentiment": labels["sentiment"],
        "scores": scores,
        "all_scores": all_scores,
        "model_version": meta["model_version"],
    }
