"""mlscan — a standalone, offline ML classifier for common code vulnerabilities.

No LLM, no API keys, no network at inference time. A scikit-learn pipeline
(TF-IDF + a linear/boosted classifier) trained on a real public dataset
classifies a code snippet into one of the top vulnerability categories (by CWE)
or "safe".
"""

__version__ = "1.0.0"
