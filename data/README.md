# Data

This folder is for the project dataset and any notes about it. The dataset is
**not** generated or committed by the scaffold — add the real file yourself.

## Expected dataset

- **Filename:** `bangla_news.csv`
- **Intended location:** this folder (`data/bangla_news.csv`), with the project
  root as a fallback. The loader is planned to check both; you will also be able
  to point it anywhere with `load_dataset(path=...)`.
- **Encoding:** UTF-8
- **Column holding the article body:** `content`.

The supplied file also contains these columns, which are not used yet:

| Column           | Description                       |
| ---------------- | --------------------------------- |
| `title`          | Headline of the article           |
| `published_date` | Publication timestamp (string)    |
| `reporter`       | Reporter name (may be empty)      |
| `category`       | News category (e.g. `sports`)     |
| `url`            | Source URL                        |
| `content`        | Full article text (main input)    |

## Notes

- Do not fabricate or commit a placeholder dataset.
- Put the CSV here (`data/bangla_news.csv`) or at the project root, or pass an
  explicit path: `load_dataset(path="data/bangla_news.csv")`.
- Raw text is left untouched; cleaning will be done in
  `src/keyword_extraction.py`.
