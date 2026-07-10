"""Feature extraction: turn raw code text into TF-IDF vectors.

We combine two views of the code:
- word n-grams (1-2): captures identifiers/API names like ``os.system``, ``eval``
- character n-grams (3-5, word-bounded): captures symbol/operator patterns like
  string concatenation in SQL, ``verify=False``, format strings — signals that a
  plain word tokenizer drops.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_vectorizer() -> FeatureUnion:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_features=20000,
        sublinear_tf=True,
        token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=30000,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])
