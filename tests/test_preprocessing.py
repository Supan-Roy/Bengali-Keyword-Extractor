"""Tests for the Bengali keyword-extraction pipeline."""

import pandas as pd
import pytest

from bnlp import BengaliCorpus

from src import keyword_extraction as ke


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


def test_extract_keywords_tfidf_ranks_terms_and_returns_top_n():
    documents = [
        "নদী দূষণ নদী পানি",
        "ক্রিকেট ক্রিকেট খেলা দল",
    ]
    results = ke.extract_keywords_tfidf(documents, top_n=2)
    assert len(results) == 2
    assert len(results[0]) == 2
    assert results[0][0][0] == "নদী"
    assert results[1][0][0] == "ক্রিকেট"


def test_extract_keywords_tfidf_rejects_non_positive_top_n():
    with pytest.raises(ValueError):
        ke.extract_keywords_tfidf(["নদী পানি"], top_n=0)


def test_extract_keywords_tfidf_empty_document_has_no_keywords():
    results = ke.extract_keywords_tfidf(["", "নদী পানি"], top_n=5)
    assert results[0] == []


def test_save_results_writes_utf8_bom_csv_and_creates_directories(tmp_path):
    df = pd.DataFrame({"keyword": ["নদী"], "score": [0.5]})
    output_path = ke.save_results(df, tmp_path / "nested" / "keywords.csv")
    assert output_path.exists()
    assert output_path.read_bytes()[:3] == b"\xef\xbb\xbf"
    assert "নদী" in output_path.read_text(encoding="utf-8-sig")


def test_clean_text_strips_zero_width_joiners():
    assert ke.clean_text("র\u200c্যাংকিং", keep_latin=True) == "র্যাংকিং"


def test_build_document_includes_title_terms_before_body():
    tokens = ke.build_document("ক্রিকেট বিশ্বকাপ", "আজ শুরু হলো বিশ্বকাপ").split()
    assert "ক্রিকেট" in tokens
    assert "বিশ্বকাপ" in tokens
    assert tokens.index("ক্রিকেট") < tokens.index("বিশ্বকাপ")
