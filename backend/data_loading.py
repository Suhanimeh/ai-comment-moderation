"""
datasets.py
-----------
Loads and prepares the three public datasets used to train the classifiers.
Raw files live in ``data/raw/`` (not committed - see ``data/README.md`` and
``scripts/download_data.py``).

Task            Source                                   Label mapping used here
--------------  ---------------------------------------  ------------------------------------------
toxicity        Davidson et al. 2017 hate-speech /       hate_speech or offensive_language -> Toxic,
                offensive-language tweets (24,783)       neither -> Non-Toxic
spam            SMS Spam Collection (5,572)              spam -> Spam, ham -> Not Spam
sentiment       Sanders Analytics Twitter sentiment      positive/negative/neutral kept,
                corpus (5,113)                           "irrelevant" dropped

Every loader strips exact duplicates (after light normalisation) BEFORE any
train/test split so identical texts (e.g. retweets) cannot appear on both sides
of the split.
"""

from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

DATASET_FILES = {
    "toxicity": "davidson_labeled_data.csv",
    "spam": "sms_spam.tsv",
    "sentiment": "sanders_full_corpus.csv",
}

TASK_LABELS = {
    "toxicity": ["Non-Toxic", "Toxic"],
    "spam": ["Not Spam", "Spam"],
    "sentiment": ["Negative", "Neutral", "Positive"],
}

# The label whose recall/precision matters most when moderating content.
POSITIVE_CLASS = {"toxicity": "Toxic", "spam": "Spam", "sentiment": "Negative"}

_RT_PREFIX = re.compile(r"^rt\s+@\w+:\s*")
_WS = re.compile(r"\s+")


class DatasetMissingError(FileNotFoundError):
    pass


def _path(task: str) -> Path:
    p = DATA_DIR / DATASET_FILES[task]
    if not p.exists():
        raise DatasetMissingError(
            f"Dataset file not found: {p}\n"
            "Run:  python scripts/download_data.py   (see data/README.md)"
        )
    return p


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    key = (
        df["text"].str.lower()
        .str.replace(_RT_PREFIX, "", regex=True)
        .str.replace(_WS, " ", regex=True)
        .str.strip()
    )
    return df.loc[~key.duplicated()].reset_index(drop=True)


def file_sha256(task: str) -> str:
    return hashlib.sha256(_path(task).read_bytes()).hexdigest()


def load_task(task: str) -> pd.DataFrame:
    """Return a DataFrame with columns ``text`` and ``label`` for *task*."""
    if task not in DATASET_FILES:
        raise ValueError(f"Unknown task: {task!r}")
    path = _path(task)

    if task == "toxicity":
        raw = pd.read_csv(path)
        label = raw["class"].map({0: "Toxic", 1: "Toxic", 2: "Non-Toxic"})
        df = pd.DataFrame({"text": raw["tweet"], "label": label})
    elif task == "spam":
        raw = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
        df = pd.DataFrame({
            "text": raw["text"],
            "label": raw["label"].map({"spam": "Spam", "ham": "Not Spam"}),
        })
    else:  # sentiment
        raw = pd.read_csv(path)
        raw = raw[raw["Sentiment"].isin(["positive", "negative", "neutral"])]
        df = pd.DataFrame({
            "text": raw["TweetText"],
            "label": raw["Sentiment"].str.capitalize(),
        })

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).map(html.unescape)
    df = df[df["text"].str.strip().astype(bool)]
    return _dedupe(df)


SPLIT_SEED = 42
TEST_SIZE = 0.20


def train_test(task: str, seed: int = SPLIT_SEED, test_size: float = TEST_SIZE):
    """Deterministic stratified split -> (X_train, X_test, y_train, y_test)."""
    from sklearn.model_selection import train_test_split

    df = load_task(task)
    return train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=test_size, random_state=seed, stratify=df["label"],
    )
