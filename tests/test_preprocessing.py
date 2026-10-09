"""Initial tests for the keyword-extraction scaffold.

The pipeline is not implemented yet, so these tests only verify that the public
functions exist and report themselves as unimplemented. Replace them with real
preprocessing tests as the functions are filled in.
"""

import pytest

from src import keyword_extraction as ke

PLACEHOLDERS = [
    ke.load_dataset,
    ke.clean_text,
    ke.tokenize,
    ke.remove_stopwords,
    ke.stem_tokens,
    ke.preprocess,
    ke.extract_keywords_tfidf,
    ke.save_results,
]


@pytest.mark.parametrize("func", PLACEHOLDERS, ids=lambda func: func.__name__)
def test_placeholder_is_not_implemented(func):
    with pytest.raises(NotImplementedError):
        func("বাংলা")
