# Developing a Keyword Extraction System for Bangla Text Using TF-IDF and NLP Techniques

A lightweight, reproducible pipeline that extracts ranked keywords from Bengali
news articles using a **title-weighted TF-IDF** model over classical NLP
preprocessing.

The finalised model is a single, reusable extractor: preprocess each article →
build a TF-IDF document with the headline repeated ×3 → rank terms by TF-IDF
weight → return the top-k.

## Project structure

```text
Bengali Keyword Extractor/
├── data/
│   ├── README.md                  # Dataset notes
│   └── bangla_news.csv            # Dataset (supplied separately, not committed)
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_bengali_preprocessing.ipynb
│   └── 03_tfidf_keyword_extraction.ipynb
├── output/
│   ├── keywords.csv               # Corpus-level predictions (11,844 articles)
│   ├── manual_evaluation.csv      # 50-article evaluation reference set
│   └── keyword_evaluation_audit.csv
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py           # Dataset loading used by the notebooks
│   └── keyword_extraction.py      # Public API (preprocessing + extraction)
├── tests/
│   ├── test_preprocessing.py      # Preprocessing helpers
│   └── test_keyword_extraction_api.py  # Public API + batch consistency
├── paper/                         # Reference paper (Showrov & Sobhan, ICAEE 2019)
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Requires **Python 3.11–3.13**. (Python 3.14 is unsupported: `bnlp-toolkit`
depends on `gensim`, which has no prebuilt wheel for 3.14.)

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset

Place `bangla_news.csv` at `data/bangla_news.csv` (the project root also works).
See `data/README.md` for the expected columns; the model uses `title` and
`content`. The dataset is not committed.

## Usage

### Single article (recommended)

Fit the extractor once on the corpus, then reuse it — this is what gives every
article the **same vocabulary and IDF weights** as the corpus model.

```python
from src.keyword_extraction import KeywordExtractor, load_dataset

df = load_dataset()                      # 11,844 cleaned articles
extractor = KeywordExtractor().fit_dataframe(df)   # title ×3 + content

keywords = extractor.extract(
    title="সিন নদীতে ব্যাপক দূষণ, স্থগিত করা হলো ট্রায়াথলন ইভেন্ট",
    content="...article body...",
    top_k=10,
)
for word, score in keywords:
    print(f"{word}\t{score:.4f}")
```

Convenience function `extract_keywords(title, content, top_k=10, extractor=None)`
wraps the same call. **Pass the fitted `extractor`.** If you omit it, a
vectorizer is fitted on that one article only; with a single document every IDF
weight collapses to 1, so the ranking degenerates to term frequency and is
**not** equivalent to the corpus model (this is documented, not accidental).

### Batch / CSV processing

```python
results = extractor.extract_batch_dataframe(df, top_k=10)
from src.keyword_extraction import save_results
save_results(results)                    # -> output/keywords.csv (UTF-8 + BOM)
```

`extract_batch()` and `extract_batch_dataframe()` use the same fitted model and
the same ranking routine as `extract()`, so single-article and batch results are
identical.

## Input / output format

Input (one article): `title: str`, `content: str`.

Output (one article): `list[tuple[str, float]]` — `(keyword, tfidf_score)` pairs,
descending by score, at most `top_k` items, surface forms preserved.

Output (batch, `output/keywords.csv`):

| column | meaning |
| --- | --- |
| `title` | article headline |
| `category` | news category |
| `url` | source URL |
| `keywords` | comma-separated top-10 keywords |
| `keyword_scores` | `keyword:score` pairs separated by `; ` |

## Methodology

```
title (repeated ×3) + content
    → clean_text          # Unicode NFC, drop zero-width joiners, strip URLs/HTML
    → tokenize            # bnlp BasicTokenizer; drop punctuation-only/numeric tokens
    → remove_stopwords    # bnlp BengaliCorpus.stopwords (398 words)
    → TfidfVectorizer(tokenizer=str.split, lowercase=False, max_features=100000)
    → top-k terms by TF-IDF weight
```

The headline is repeated three times because headline terms are the most salient
— this was the single largest measured improvement (+11.8 points over a
content-only model). **No stemming, POS filtering, or graph ranking is applied.**

## Evaluation and limitations

- **Reference set:** 50 sampled articles (`output/manual_evaluation.csv`, 10
  reference keywords each), scored against the model's top-10.
- **Metric:** macro-average precision/recall/F1 @10. Because both the reference
  and the prediction list contain exactly 10 terms, `P@10 = R@10 = F1@10`.
- **Results:** **44.40 %** exact-match F1; **49.00 %** with Unicode-normalized
  matching (NFC + zero-width removal + casefold), reported separately.

Limitations to keep in mind:

- **No lemmatisation.** No verified Bengali lemmatiser was available for this
  environment (bnlp has none; `indic-nlp-library` ships only Morfessor-based
  unsupervised segmentation; Stanza has no Bengali lemma model and needs
  PyTorch). Inflected surface forms (`নদী` / `নদীর` / `নদীতে`) therefore do not
  match, which is a major reason the exact-match score is not higher. Rule-based
  suffix stripping was rejected because it corrupts proper names
  (`কোহলি → কোহল`, `হাসিনা → হাসিন`).
- **Small, single-annotator reference.** 50 articles, no inter-annotator
  agreement; treat the score as indicative, not definitive.
- **Not comparable to the reference paper.** Showrov & Sobhan (ICAEE 2019)
  report precision at k = 20/40/60 over 8 single-title datasets (50–85 %). Our
  task uses 50 articles, k = 10, macro F1, and a different reference set — the
  figures are not directly comparable.

## Running the tests

```bash
pytest
```

34 tests cover the preprocessing helpers (cleaning, tokenisation, stop-word
removal, document building) and the public extraction API (single-article
ranking, empty/invalid inputs, unseen terms, and single-vs-batch consistency).

## Dependencies

`pandas`, `scikit-learn` (TF-IDF), `bnlp-toolkit` (Bengali tokenizer + stop
words), plus `matplotlib`/`ipykernel` for the notebooks and `pytest` for tests.
See `requirements.txt`.

## Out of scope

Web apps, GUIs, APIs, databases, ML classifiers and neural networks.
