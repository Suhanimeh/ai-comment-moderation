"""
preprocess.py
-------------
Text cleaning for the comment-moderation pipelines.

Two variants are implemented so the choice can be *measured* instead of assumed
(see ``experiments/preprocessing_ablation.csv``):

* ``legacy``   - the original behaviour: URLs, @mentions, #hashtags and digits
                 are deleted outright.
* ``enhanced`` - URLs, @mentions and numbers are replaced by indicator tokens
                 (``urltoken``, ``mentiontoken``, ``numtoken``) instead of being
                 deleted, because their *presence* can be a spam signal even
                 when their content is not. Repeated characters
                 ("soooo") are also collapsed.

Both variants lowercase, strip punctuation/emoji, drop stopwords (keeping
negations and a few polarity words) and lemmatise.

The functions here are used as the ``preprocessor`` of the TF-IDF vectoriser
inside each scikit-learn Pipeline, so training and inference are guaranteed to
apply identical cleaning.

NLTK resources
--------------
Resources are NOT downloaded at import time. Run once during setup::

    python scripts/setup_nltk.py        # or: python -m preprocess

If they are missing, :func:`clean_text` raises :class:`NLTKResourceError` with
instructions instead of failing obscurely or silently downloading.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache, partial

REQUIRED_NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
}

# Negations / polarity words that matter for toxicity and sentiment, so they
# are kept even though NLTK lists most of them as stopwords.
KEEP_WORDS = {
    "not", "no", "never", "nothing", "nobody", "nowhere", "nor",
    "neither", "against", "hate", "love", "good", "bad",
}

_URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b\S+\.(?:com|net|org|io|ly|ee)\b\S*")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_DIGITS_RE = re.compile(r"\d+")
_PUNCT_RE = re.compile(r"[^\w\s]|_")          # anything that is not a letter/digit/space
_REPEAT_RE = re.compile(r"(.)\1{2,}")          # 3+ identical chars -> 2


class NLTKResourceError(RuntimeError):
    """Raised when a required NLTK corpus has not been installed."""


def ensure_nltk_resources(download: bool = False) -> None:
    """Check (and optionally download) the NLTK corpora needed for cleaning."""
    import nltk

    missing = []
    for name, path in REQUIRED_NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            missing.append(name)

    if missing and download:
        for name in missing:
            nltk.download(name, quiet=True)
        return ensure_nltk_resources(download=False)

    if missing:
        raise NLTKResourceError(
            f"Missing NLTK resources: {', '.join(missing)}. "
            "Install them once with:  python scripts/setup_nltk.py"
        )


@lru_cache(maxsize=1)
def _resources():
    ensure_nltk_resources(download=False)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop = set(stopwords.words("english")) - KEEP_WORDS
    return stop, WordNetLemmatizer()


@lru_cache(maxsize=100_000)
def _lemma(token: str) -> str:
    return _resources()[1].lemmatize(token)


def _clean(text: str, variant: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    stop, _ = _resources()

    text = text.lower()
    if variant == "enhanced":
        text = _URL_RE.sub(" urltoken ", text)
        text = _MENTION_RE.sub(" mentiontoken ", text)
        text = _HASHTAG_RE.sub(" ", text)
        text = _DIGITS_RE.sub(" numtoken ", text)
        text = _REPEAT_RE.sub(r"\1\1", text)
    else:  # legacy
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"[@#]\w+", "", text)
        text = _DIGITS_RE.sub("", text)

    text = _PUNCT_RE.sub(" ", text)
    tokens = [_lemma(t) for t in text.split() if t not in stop and len(t) > 1]
    return " ".join(tokens)


def clean_legacy(text: str) -> str:
    """Original cleaning: delete URLs, mentions, hashtags, digits."""
    return _clean(text, "legacy")


def clean_enhanced(text: str) -> str:
    """Indicator tokens for URLs/mentions/numbers; collapse repeated chars."""
    return _clean(text, "enhanced")


PREPROCESSORS = {"legacy": clean_legacy, "enhanced": clean_enhanced}
DEFAULT_VARIANT = "enhanced"


def clean_text(text: str, variant: str = DEFAULT_VARIANT) -> str:
    """Normalise a raw comment string. Returns '' if nothing usable remains."""
    if variant not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocessing variant: {variant!r}")
    return PREPROCESSORS[variant](text)


def get_preprocessor(variant: str):
    """Picklable preprocessor callable for use inside a TfidfVectorizer."""
    if variant not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocessing variant: {variant!r}")
    return PREPROCESSORS[variant]


if __name__ == "__main__":
    ensure_nltk_resources(download=True)
    print("NLTK resources ready:", ", ".join(REQUIRED_NLTK_RESOURCES))
    sys.exit(0)
