"""Bangla keyword extraction — project scaffold.

This module defines the public interface of the extraction pipeline. Nothing is
implemented yet: every function is a placeholder that raises
``NotImplementedError``. Fill in the bodies as the project progresses.

Planned pipeline
----------------
``bangla_news.csv``
    -> :func:`load_dataset`
    -> :func:`clean_text`
    -> :func:`tokenize`
    -> :func:`remove_stopwords`
    -> :func:`stem_tokens`             (needs a verified Bengali NLP library)
    -> :func:`extract_keywords_tfidf`
    -> :func:`save_results`
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Intended dataset locations, checked in order. The dataset is not generated
#: or shipped by this scaffold.
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "bangla_news.csv"
FALLBACK_DATASET_PATH = PROJECT_ROOT / "bangla_news.csv"

#: Column in ``bangla_news.csv`` that holds the article body.
DEFAULT_TEXT_COLUMN = "content"

#: Intended directory for generated result files.
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


# ---------------------------------------------------------------------------
# Function stubs
# ---------------------------------------------------------------------------

def load_dataset(
    path: str | Path | None = None,
    text_column: str = DEFAULT_TEXT_COLUMN,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load ``bangla_news.csv`` into a pandas DataFrame.

    Intended to read from :data:`DEFAULT_DATASET_PATH`, fall back to
    :data:`FALLBACK_DATASET_PATH`, validate ``text_column`` and drop blank rows.
    Not implemented yet.
    """
    raise NotImplementedError


def clean_text(text: str, keep_latin: bool = False) -> str:
    """Clean and normalise a single Bengali text field.

    Intended to remove URLs and HTML, normalise Unicode, and strip punctuation,
    digits and non-Bengali characters. Not implemented yet.
    """
    raise NotImplementedError


def tokenize(text: str) -> list[str]:
    """Split text into Bengali word tokens.

    Not implemented yet.
    """
    raise NotImplementedError


def remove_stopwords(
    tokens: Iterable[str],
    stopwords: Iterable[str] | None = None,
) -> list[str]:
    """Remove stop words from a token sequence.

    Not implemented yet.
    """
    raise NotImplementedError


def stem_tokens(
    tokens: Sequence[str],
    stemmer: Callable[[str], str] | None = None,
) -> list[str]:
    """Apply Bengali stemming to a token sequence.

    Intended to be backed by a verified Bengali NLP library (for example
    ``bnlp-toolkit`` or ``indic-nlp-library``) rather than hand-written suffix
    rules. Not implemented yet.
    """
    raise NotImplementedError


def preprocess(
    text: str,
    stopwords: Iterable[str] | None = None,
    stemmer: Callable[[str], str] | None = None,
) -> list[str]:
    """Run cleaning, tokenisation and stop-word removal on one document.

    Not implemented yet.
    """
    raise NotImplementedError


def extract_keywords_tfidf(
    documents: Sequence[str],
    top_n: int = 10,
    **vectorizer_kwargs: object,
) -> list[list[tuple[str, float]]]:
    """Rank the top ``top_n`` keywords per document using TF-IDF.

    Intended to use ``sklearn.feature_extraction.text.TfidfVectorizer``.
    Not implemented yet.
    """
    raise NotImplementedError


def save_results(
    results: pd.DataFrame,
    output_path: str | Path = DEFAULT_OUTPUT_DIR / "keywords.csv",
) -> Path:
    """Write a results table (e.g. keywords with scores) to CSV.

    Not implemented yet.
    """
    raise NotImplementedError
