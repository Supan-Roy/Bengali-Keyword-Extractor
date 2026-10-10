"""Tests for the public single-article / batch keyword extraction API."""

import pandas as pd
import pytest

import src.keyword_extraction as ke
from src.keyword_extraction import (
    KeywordExtractor,
    build_document,
    extract_keywords,
    extract_keywords_tfidf,
)

CORPUS = pd.DataFrame({
    "title": ["ক্রিকেট বিশ্বকাপ", "নদী দূষণ", "ক্রিকেট দল", "বিশ্বকাপ ফাইনাল"],
    "content": [
        "ক্রিকেট বিশ্বকাপ আজ শুরু",
        "নদীর পানি দূষণ বেড়েছে",
        "দল জিতেছে ক্রিকেট ম্যাচ",
        "ফাইনালে বিশ্বকাপ জিতল দল",
    ],
    "category": ["sports"] * 4,
    "url": ["u1", "u2", "u3", "u4"],
})


def _names(result):
    return [word for word, _ in result]


def _fitted():
    return KeywordExtractor().fit_dataframe(CORPUS)


# --------------------------------------------------------------------------
# build_document
# --------------------------------------------------------------------------

def test_build_document_repeats_title_by_weight():
    tokens = build_document("ক্রিকেট বিশ্বকাপ", "নদী", title_weight=3).split()
    assert tokens.count("ক্রিকেট") == 3
    assert tokens.count("বিশ্বকাপ") == 3


def test_build_document_defaults_to_finalised_weight():
    assert build_document("ক্রিকেট", "নদী").split().count("ক্রিকেট") == 3


def test_build_document_rejects_invalid_title_weight():
    with pytest.raises(ValueError):
        build_document("ক", "খ", title_weight=0)


# --------------------------------------------------------------------------
# KeywordExtractor
# --------------------------------------------------------------------------

def test_extract_returns_ranked_top_k():
    result = _fitted().extract("ক্রিকেট বিশ্বকাপ", CORPUS.loc[0, "content"], top_k=3)
    assert 0 < len(result) <= 3
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)


def test_extract_requires_fitted_model():
    with pytest.raises(RuntimeError):
        KeywordExtractor().extract("ক্রিকেট", "ক্রিকেট")
    with pytest.raises(RuntimeError):
        KeywordExtractor().extract_batch(CORPUS)


def test_invalid_top_k_raises():
    extractor = _fitted()
    for bad in (0, -1):
        with pytest.raises(ValueError):
            extractor.extract("ক্রিকেট", "ক্রিকেট", top_k=bad)
        with pytest.raises(ValueError):
            extractor.extract_batch(CORPUS, top_k=bad)
        with pytest.raises(ValueError):
            extract_keywords("ক্রিকেট", "ক্রিকেট", top_k=bad)


def test_fit_rejects_all_empty_documents():
    with pytest.raises(ValueError):
        KeywordExtractor().fit(["", "   ", ""])


def test_empty_article_and_fields_are_handled():
    extractor = _fitted()
    assert extractor.extract("", "", top_k=5) == []
    assert isinstance(extractor.extract("ক্রিকেট", "", top_k=5), list)
    assert isinstance(extractor.extract("", "নদীর পানি", top_k=5), list)


def test_extractor_ignores_terms_unseen_in_corpus():
    extractor = _fitted()
    result = extractor.extract("সম্পূর্ণঅজানাশব্দ", "অন্যন্যন্যবর্ণ", top_k=5)
    assert result == []


# --------------------------------------------------------------------------
# public single-article function
# --------------------------------------------------------------------------

def test_extract_keywords_with_corpus_extractor_matches_extractor():
    extractor = _fitted()
    title, content = "ক্রিকেট বিশ্বকাপ", CORPUS.loc[0, "content"]
    assert extract_keywords(title, content, top_k=5, extractor=extractor) == \
        extractor.extract(title, content, top_k=5)


def test_extract_keywords_blank_inputs_return_empty():
    assert extract_keywords("", "", top_k=5) == []
    assert extract_keywords(None, "   ", top_k=5) == []
    assert extract_keywords(top_k=5) == []      # both omitted; no extractor needed


def test_default_extractor_is_fitted_once_and_reused(monkeypatch):
    stub = KeywordExtractor().fit_dataframe(CORPUS)
    monkeypatch.setattr(ke, "_DEFAULT_EXTRACTOR", stub)

    def _boom(*args, **kwargs):
        raise AssertionError("default extractor must not be refitted per request")

    monkeypatch.setattr(ke, "load_dataset", _boom)
    assert ke.extract_keywords("ক্রিকেট বিশ্বকাপ", "ক্রিকেট বিশ্বকাপ আজ শুরু", top_k=5)
    assert ke.extract_keywords(content="নদীর পানি দূষণ", top_k=5)


# --------------------------------------------------------------------------
# optional / blank title
# --------------------------------------------------------------------------

def test_blank_title_adds_no_title_tokens():
    extractor = _fitted()
    content = CORPUS.loc[1, "content"]           # নদীর পানি দূষণ বেড়েছে
    for blank in (None, "", "   "):
        words = _names(extractor.extract(blank, content, top_k=5))
        assert words
        assert "ক্রিকেট" not in words and "বিশ্বকাপ" not in words


def test_blank_title_variants_are_equivalent():
    extractor = _fitted()
    content = CORPUS.loc[0, "content"]
    assert (
        extractor.extract(None, content, top_k=5)
        == extractor.extract("", content, top_k=5)
        == extractor.extract("   ", content, top_k=5)
    )


def test_title_only_extraction_when_content_blank():
    extractor = _fitted()
    for blank in (None, "", "   "):
        words = _names(extractor.extract("ক্রিকেট বিশ্বকাপ", blank, top_k=5))
        assert "ক্রিকেট" in words


def test_both_blank_returns_empty():
    extractor = _fitted()
    assert extractor.extract("", "", top_k=5) == []
    assert extractor.extract(None, "   ", top_k=5) == []
    assert extract_keywords(None, None, top_k=5, extractor=extractor) == []


def test_title_plus_content_behaviour_is_preserved():
    extractor = _fitted()
    title, content = "ক্রিকেট বিশ্বকাপ", CORPUS.loc[0, "content"]
    assert extractor.extract(title, content, top_k=5) == \
        extractor.extract(title=title, content=content, top_k=5)
    # title weighting still applied when a title is given (content has no "ক্রিকেট")
    assert build_document(title, CORPUS.loc[1, "content"]).split().count("ক্রিকেট") == 3


def test_batch_falls_back_to_content_for_blank_titles():
    df = CORPUS.copy()
    df.loc[0, "title"] = ""
    extractor = KeywordExtractor().fit_dataframe(df)
    batch = extractor.extract_batch(df, top_k=5)
    singles = [
        extractor.extract(row["title"], row["content"], top_k=5)
        for _, row in df.iterrows()
    ]
    assert [_names(r) for r in batch] == [_names(r) for r in singles]
    assert batch[0]


# --------------------------------------------------------------------------
# batch consistency
# --------------------------------------------------------------------------

def test_batch_matches_per_article_extraction():
    extractor = _fitted()
    batch = extractor.extract_batch(CORPUS, top_k=5)
    singles = [
        extractor.extract(row["title"], row["content"], top_k=5)
        for _, row in CORPUS.iterrows()
    ]
    assert [_names(r) for r in batch] == [_names(r) for r in singles]
    for got, expected in zip(batch, singles):
        assert [s for _, s in got] == pytest.approx([s for _, s in expected])


def test_batch_matches_corpus_level_extract_keywords_tfidf():
    extractor = _fitted()
    documents = [
        build_document(title, content)
        for title, content in zip(CORPUS["title"], CORPUS["content"])
    ]
    expected = extract_keywords_tfidf(documents, top_n=5, max_features=100000)
    got = extractor.extract_batch(CORPUS, top_k=5)
    assert [_names(r) for r in got] == [_names(r) for r in expected]
    for got_row, expected_row in zip(got, expected):
        assert [s for _, s in got_row] == pytest.approx([s for _, s in expected_row])


def test_batch_dataframe_matches_keywords_csv_shape():
    out = _fitted().extract_batch_dataframe(CORPUS, top_k=3)
    assert list(out.columns) == ["title", "category", "url", "keywords", "keyword_scores"]
    assert len(out) == len(CORPUS)
    assert ", " in out.loc[0, "keywords"]
    assert ":" in out.loc[0, "keyword_scores"]
