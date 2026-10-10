"""Initial tests for the keyword-extraction scaffold.

Covers the implemented preprocessing helpers (``load_dataset``, ``clean_text``,
``tokenize``, ``remove_stopwords``, ``stem_tokens``, ``preprocess``). The
remaining functions are still stubs and are checked to report themselves as
unimplemented.
"""

import pytest

from bnlp import BengaliCorpus

from src import keyword_extraction as ke

PLACEHOLDERS = [
    ke.extract_keywords_tfidf,
    ke.save_results,
]


@pytest.mark.parametrize("func", PLACEHOLDERS, ids=lambda func: func.__name__)
def test_placeholder_is_not_implemented(func):
    with pytest.raises(NotImplementedError):
        func("বাংলা")


def test_clean_text_removes_html_and_urls():
    raw = "<p>খবর দেখুন https://example.com/news</p>"
    assert ke.clean_text(raw, keep_latin=True) == "খবর দেখুন"


def test_clean_text_keeps_only_bengali_by_default():
    assert ke.clean_text("Djokovic বাংলা") == "বাংলা"


def test_clean_text_keeps_latin_when_requested():
    assert ke.clean_text("Djokovic US Open", keep_latin=True) == "Djokovic US Open"


def test_clean_text_handles_non_string_input():
    assert ke.clean_text(None) == ""


def test_tokenize_splits_bengali_words():
    assert ke.tokenize("বাংলা ভাষা।") == ["বাংলা", "ভাষা"]


def test_tokenize_drops_standalone_numbers():
    assert ke.tokenize("বাংলা ২০২৪") == ["বাংলা"]


def test_tokenize_handles_empty_input():
    assert ke.tokenize("   ") == []


def test_remove_stopwords_accepts_custom_list():
    tokens = ["বাংলা", "ভাষা", "সাহিত্য"]
    assert ke.remove_stopwords(tokens, stopwords={"ভাষা"}) == ["বাংলা", "সাহিত্য"]


def test_remove_stopwords_uses_bnlp_default_list():
    sample = BengaliCorpus.stopwords[0]
    assert ke.remove_stopwords(["বাংলা", sample]) == ["বাংলা"]


def test_stem_tokens_returns_tokens_unchanged_without_stemmer():
    assert ke.stem_tokens(["বাংলা", "ভাষা"]) == ["বাংলা", "ভাষা"]


def test_stem_tokens_applies_supplied_stemmer():
    assert ke.stem_tokens(["খেলার"], stemmer=lambda t: t.rstrip("র")) == ["খেলা"]


def test_preprocess_runs_full_pipeline():
    result = ke.preprocess("বাংলা ভাষা ও সাহিত্য", stopwords={"ও"})
    assert result == ["বাংলা", "ভাষা", "সাহিত্য"]


def test_preprocess_applies_stemmer_when_supplied():
    result = ke.preprocess("খেলার", stopwords=set(), stemmer=lambda t: t.rstrip("র"))
    assert result == ["খেলা"]
