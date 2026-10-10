"""Bangla keyword extraction with a title-weighted TF-IDF model.

Public API
----------
Preprocessing:
    :func:`load_dataset`, :func:`clean_text`, :func:`tokenize`,
    :func:`remove_stopwords`, :func:`preprocess`, :func:`build_document`.
Extraction:
    :class:`KeywordExtractor` (fit once on the corpus, reuse for single
    articles and batches), :func:`extract_keywords` (one article),
    :func:`extract_keywords_tfidf` (corpus-level ranking).
Output:
    :func:`save_results`.

Finalised pipeline
------------------
``title (repeated x3) + content``
    -> :func:`clean_text` -> :func:`tokenize` -> :func:`remove_stopwords`
    -> ``TfidfVectorizer(tokenizer=str.split, max_features=100000)``
    -> top-k terms by TF-IDF weight.

No stemming, POS filtering or graph ranking is applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

import re
import unicodedata

from bnlp import BasicTokenizer, BengaliCorpus
from sklearn.feature_extraction.text import TfidfVectorizer

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

#: Column holding the article headline.
DEFAULT_TITLE_COLUMN = "title"

#: Intended directory for generated result files.
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

#: Title repetition used by the finalised model: the headline is repeated this
#: many times before the body when building a TF-IDF document.
DEFAULT_TITLE_WEIGHT = 3

#: Vocabulary cap used by the finalised model.
DEFAULT_MAX_FEATURES = 100000


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(
    path: str | Path | None = None,
    text_column: str = DEFAULT_TEXT_COLUMN,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load ``bangla_news.csv`` into a pandas DataFrame.

    Reads from :data:`DEFAULT_DATASET_PATH`, falls back to
    :data:`FALLBACK_DATASET_PATH`, validates ``text_column``, drops blank rows,
    duplicate URLs and the known metadata-only entry.
    """
    if path is None:
        if DEFAULT_DATASET_PATH.exists():
            path = DEFAULT_DATASET_PATH
        elif FALLBACK_DATASET_PATH.exists():
            path = FALLBACK_DATASET_PATH
        else:
            raise FileNotFoundError(
                f"Dataset not found: {DEFAULT_DATASET_PATH}"
            )
    path = Path(path)
    df = pd.read_csv(path, encoding=encoding)
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    df[text_column] = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    df = df[df[text_column] != ""].copy()
    if "url" in df.columns:
        df = df.drop_duplicates(subset="url", keep="first")
    df = df[
        ~df[text_column].isin({"আল মাহফুজ"})
    ].copy()
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str, keep_latin: bool = False) -> str:
    """Clean and normalise a single Bengali text field.

    Drops zero-width joiners, removes URLs and HTML, applies Unicode NFC and
    collapses whitespace. With ``keep_latin=False`` only Bengali-script
    characters and spaces are kept.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    # Drop zero-width joiners/non-joiners (used to render conjuncts) so they
    # cannot split or pollute tokens; NFC alone leaves them in place.
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if keep_latin:
        return text
    return re.sub(r"[^\u0980-\u09FF\s]", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split text into Bengali word tokens via bnlp's ``BasicTokenizer``.

    Punctuation-only and standalone-numeric tokens are dropped.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    tokenizer = BasicTokenizer()
    tokens = tokenizer(text)

    cleaned_tokens = []
    for token in tokens:
        # Remove punctuation-only tokens
        if all(
            not char.isalnum()
            and not ("\u0980" <= char <= "\u09FF")
            for char in token
        ):
            continue
        # Remove standalone numeric tokens
        if re.fullmatch(r"\d+", token):
            continue
        cleaned_tokens.append(token)

    return cleaned_tokens


def remove_stopwords(
    tokens: Iterable[str],
    stopwords: Iterable[str] | None = None,
) -> list[str]:
    """Remove stop words from a token sequence.

    Defaults to bnlp's Bengali stop-word list (``BengaliCorpus.stopwords``).
    """
    if stopwords is None:
        stopwords = BengaliCorpus.stopwords
    stopword_set = set(stopwords)
    return [token for token in tokens if token not in stopword_set]


def stem_tokens(
    tokens: Sequence[str],
    stemmer: Callable[[str], str] | None = None,
) -> list[str]:
    """Apply Bengali stemming to a token sequence.

    No stemmer ships with this project (no verified Bengali method was
    available), so tokens are returned unchanged unless a caller supplies one.
    """
    if stemmer is None:
        return list(tokens)
    return [stemmer(token) for token in tokens]


def preprocess(
    text: str,
    stopwords: Iterable[str] | None = None,
    stemmer: Callable[[str], str] | None = None,
) -> list[str]:
    """Clean, tokenise and remove stop words from one text field."""
    cleaned = clean_text(text, keep_latin=True)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens, stopwords)
    tokens = stem_tokens(tokens, stemmer)
    return tokens


def build_document(
    title: str | None,
    text: str | None,
    title_weight: int = DEFAULT_TITLE_WEIGHT,
    stopwords: Iterable[str] | None = None,
    stemmer: Callable[[str], str] | None = None,
) -> str:
    """Build the TF-IDF document string for one article (title + body).

    The headline is repeated ``title_weight`` times before the body, because
    headline terms are the most salient. ``title_weight=3`` is the finalised
    model configuration.

    ``title`` and ``text`` may be ``None``, empty or whitespace-only. A blank
    title contributes **no** title tokens, so the document is built from the body
    alone (and vice-versa when the body is blank); if both are blank the document
    is an empty string.
    """
    if title_weight < 1:
        raise ValueError("title_weight must be at least 1")
    title_tokens = preprocess(title, stopwords=stopwords, stemmer=stemmer)
    body_tokens = preprocess(text, stopwords=stopwords, stemmer=stemmer)
    return " ".join(title_tokens * title_weight + body_tokens)


def _top_keywords(
    scores, feature_names, top_k: int
) -> list[tuple[str, float]]:
    """Rank one TF-IDF score row and return up to ``top_k`` positive terms."""
    order = scores.argsort()[::-1][:top_k]
    return [
        (feature_names[index], float(scores[index]))
        for index in order
        if scores[index] > 0
    ]


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

class KeywordExtractor:
    """Title-weighted TF-IDF keyword extractor with a reusable fitted model.

    Fit once on the corpus so that every article is scored with the *same*
    vocabulary and IDF weights, then call :meth:`extract` per article or
    :meth:`extract_batch` for a whole DataFrame.
    """

    def __init__(
        self,
        title_weight: int = DEFAULT_TITLE_WEIGHT,
        max_features: int = DEFAULT_MAX_FEATURES,
    ) -> None:
        if title_weight < 1:
            raise ValueError("title_weight must be at least 1")
        self.title_weight = title_weight
        self.max_features = max_features
        self.vectorizer: TfidfVectorizer | None = None
        self.feature_names = None

    @property
    def is_fitted(self) -> bool:
        """Whether the corpus vectorizer has been fitted."""
        return self.vectorizer is not None

    def _document(self, title: str, content: str) -> str:
        return build_document(title, content, title_weight=self.title_weight)

    def fit(self, documents: Sequence[str]) -> "KeywordExtractor":
        """Fit the TF-IDF vocabulary and IDF weights on a corpus of documents."""
        if not any(str(doc).strip() for doc in documents):
            raise ValueError("cannot fit: every document is empty")
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            max_features=self.max_features,
        )
        self.vectorizer.fit(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self

    def fit_dataframe(
        self,
        df: pd.DataFrame,
        title_column: str = DEFAULT_TITLE_COLUMN,
        text_column: str = DEFAULT_TEXT_COLUMN,
    ) -> "KeywordExtractor":
        """Fit on a DataFrame of articles (title + content columns)."""
        documents = [
            self._document(title, content)
            for title, content in zip(df[title_column], df[text_column])
        ]
        return self.fit(documents)

    def extract(
        self,
        title: str | None = None,
        content: str = "",
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return up to ``top_k`` (keyword, score) pairs for ONE article.

        Uses the corpus vocabulary/IDF from :meth:`fit`; terms unseen in the
        corpus are ignored.

        ``title`` may be ``None``, blank or whitespace-only, in which case no
        title tokens are added and keywords come from ``content`` alone. A blank
        ``content`` with a title extracts from the title alone; both blank
        returns ``[]``.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.is_fitted:
            raise RuntimeError("fit() the extractor before calling extract()")
        row = self.vectorizer.transform([self._document(title, content)])
        return _top_keywords(row.toarray().ravel(), self.feature_names, top_k)

    def extract_batch(
        self,
        df: pd.DataFrame,
        top_k: int = 10,
        title_column: str = DEFAULT_TITLE_COLUMN,
        text_column: str = DEFAULT_TEXT_COLUMN,
    ) -> list[list[tuple[str, float]]]:
        """Extract keywords for every row of a DataFrame.

        Uses the same fitted model and ranking routine as :meth:`extract`, so
        batch and single-article results are identical. Rows whose title is
        blank fall back to their content alone, exactly as :meth:`extract` does.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.is_fitted:
            raise RuntimeError("fit() the extractor before calling extract_batch()")
        documents = [
            self._document(title, content)
            for title, content in zip(df[title_column], df[text_column])
        ]
        matrix = self.vectorizer.transform(documents)
        return [
            _top_keywords(matrix[i].toarray().ravel(), self.feature_names, top_k)
            for i in range(matrix.shape[0])
        ]

    def extract_batch_dataframe(
        self,
        df: pd.DataFrame,
        top_k: int = 10,
        title_column: str = DEFAULT_TITLE_COLUMN,
        text_column: str = DEFAULT_TEXT_COLUMN,
    ) -> pd.DataFrame:
        """Batch extraction as a results table (matches ``output/keywords.csv``).

        Columns: ``title, category, url, keywords, keyword_scores`` (the latter
        two only when the corresponding source columns exist).
        """
        keywords = self.extract_batch(
            df, top_k=top_k, title_column=title_column, text_column=text_column
        )
        table: dict[str, list] = {"title": list(df[title_column])}
        if "category" in df.columns:
            table["category"] = list(df["category"])
        if "url" in df.columns:
            table["url"] = list(df["url"])
        table["keywords"] = [", ".join(w for w, _ in row) for row in keywords]
        table["keyword_scores"] = [
            "; ".join(f"{w}:{s:.4f}" for w, s in row) for row in keywords
        ]
        return pd.DataFrame(table)


def _has_text(value: object) -> bool:
    """True when ``value`` is a string containing something other than spaces."""
    return isinstance(value, str) and value.strip() != ""


#: Process-wide default extractor, fitted once on the shipped dataset and reused.
_DEFAULT_EXTRACTOR: "KeywordExtractor | None" = None


def _get_default_extractor() -> "KeywordExtractor":
    """Return the default extractor, fitting it once on the shipped dataset.

    Built lazily on first use and cached for the process, so individual requests
    reuse the same vocabulary and IDF weights instead of refitting a vectorizer.
    """
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = KeywordExtractor().fit_dataframe(load_dataset())
    return _DEFAULT_EXTRACTOR


def extract_keywords(
    title: str | None = None,
    content: str = "",
    top_k: int = 10,
    extractor: KeywordExtractor | None = None,
) -> list[tuple[str, float]]:
    """Return the top ``top_k`` (keyword, score) pairs for a single article.

    ``title`` is **optional**. Pass ``None``, ``""`` or whitespace to extract
    from the body alone (no title tokens are added). When a title is present the
    finalised model repeats it ``DEFAULT_TITLE_WEIGHT`` (3) times before the
    body::

        extractor = KeywordExtractor().fit_dataframe(load_dataset())

        # title + content (finalised title x3 model)
        extract_keywords(title, content, top_k=10, extractor=extractor)

        # content only -- omit/blank the title
        extract_keywords(content=content, top_k=10, extractor=extractor)

    ``extractor`` should be a corpus-fitted :class:`KeywordExtractor`. When it is
    omitted, a default extractor fitted once on the shipped dataset is used
    (cached for the process), so no vectorizer is refitted per request.

    Returns an empty list when both ``title`` and ``content`` are blank.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not _has_text(title) and not _has_text(content):
        return []
    if extractor is None:
        extractor = _get_default_extractor()
    return extractor.extract(title, content, top_k)


def extract_keywords_tfidf(
    documents: Sequence[str],
    top_n: int = 10,
    **vectorizer_kwargs: object,
) -> list[list[tuple[str, float]]]:
    """Rank the top ``top_n`` keywords per document using TF-IDF.

    Corpus-level helper: fits one ``TfidfVectorizer`` on ``documents`` (already
    preprocessed, space-joined strings) and ranks each row. Kept for batch use;
    :class:`KeywordExtractor` is the reusable equivalent.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        **vectorizer_kwargs,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    results = []
    for row_index in range(tfidf_matrix.shape[0]):
        scores = tfidf_matrix[row_index].toarray().flatten()
        results.append(_top_keywords(scores, feature_names, top_n))

    return results


def save_results(
    results: pd.DataFrame,
    output_path: str | Path = DEFAULT_OUTPUT_DIR / "keywords.csv",
) -> Path:
    """Write a results table (e.g. keywords with scores) to CSV.

    Creates parent directories if needed; UTF-8 with BOM for Excel.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    return output_path
