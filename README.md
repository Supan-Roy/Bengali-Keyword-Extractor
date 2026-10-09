# Developing a Keyword Extraction System for Bangla Text Using TF-IDF and NLP Techniques

Course project scaffold for extracting keywords from Bengali news articles with
TF-IDF and classical NLP preprocessing. This repository currently contains the
**initial structure and function stubs only** — no processing logic is
implemented yet.

## Purpose

Build a lightweight, reproducible pipeline that turns raw Bengali news text into
ranked keywords, using:

- **pandas** for dataset handling;
- **scikit-learn** (`TfidfVectorizer`) for the TF-IDF scoring step;
- Bengali-aware text cleaning, tokenisation and stop-word removal;
- a verified Bengali NLP library for stemming (to be integrated).

Deliberately out of scope: web apps, GUIs, APIs, databases, ML classifiers and
neural networks.

## Project structure

```text
Bengali Keyword Extractor/
├── data/
│   ├── README.md              # Dataset notes (dataset itself is not committed)
│   └── bangla_news.csv        # Dataset (add it yourself)
├── src/
│   ├── __init__.py
│   └── keyword_extraction.py  # Pipeline interface (stubs only)
├── output/
│   └── .gitkeep               # Generated results land here
├── tests/
│   └── test_preprocessing.py
├── pytest.ini                 # Points pytest at tests/ and the project root
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset as data/bangla_news.csv (project root also works)
#    (see data/README.md for the expected columns)
```

## Running the tests

```bash
pytest
```

## Planned pipeline

Every stage below is a stub — none of it is implemented yet.

```
bangla_news.csv
    → load_dataset            # read CSV into a DataFrame           [stub]
    → clean_text              # strip URLs/HTML, normalise Unicode  [stub]
    → tokenize                # Bengali word tokens                 [stub]
    → remove_stopwords        # drop function words                 [stub]
    → stem_tokens             # Bengali stemming                    [stub]
    → extract_keywords_tfidf  # TF-IDF keyword ranking              [stub]
    → save_results            # write keywords to output/           [stub]
```

## Project status

All functions in `src/keyword_extraction.py` are placeholders that raise
`NotImplementedError`:

- `load_dataset` — dataset loading
- `clean_text` — Bengali text cleaning
- `tokenize` — tokenisation
- `remove_stopwords` — stop-word removal
- `stem_tokens` — Bengali stemming
- `preprocess` — composes the steps above
- `extract_keywords_tfidf` — TF-IDF keyword extraction
- `save_results` — result output

The tests currently only assert that these placeholders are unimplemented, so
the scaffold makes no claim that any behaviour works.

## Example (once implemented)

```python
from src.keyword_extraction import load_dataset, preprocess

df = load_dataset()  # reads data/bangla_news.csv, uses the "content" column
tokens = preprocess(df.loc[0, "content"])
print(tokens[:20])
```

This will raise `NotImplementedError` until the functions are filled in.

## Next steps

- Implement loading, cleaning, tokenisation and stop-word removal.
- Integrate a verified Bengali stemmer (for example `bnlp-toolkit` or
  `indic-nlp-library`) and wire it into `stem_tokens`.
- Implement `extract_keywords_tfidf` with `sklearn` and evaluate the extracted
  keywords.
