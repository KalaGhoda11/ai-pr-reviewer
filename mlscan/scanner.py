"""Inference: load the trained model and classify code. Offline, no API keys.

>>> from mlscan.scanner import scan
>>> scan("query = 'SELECT * FROM u WHERE id=' + user; db.execute(query)")
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

from mlscan.labels import SAFE, describe

# Benign: LightGBM was fit on a named sparse matrix, predicts on an unnamed one.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

MODEL_PATH = Path(__file__).resolve().parent / "model" / "vuln_clf.joblib"

# A finding is reported when a non-"safe" class scores at/above this probability.
# Set for precision (few false positives) over recall — see docs/DESIGN.md.
DEFAULT_THRESHOLD = 0.50


class ModelNotTrained(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _model():
    if not MODEL_PATH.exists():
        raise ModelNotTrained(
            f"No trained model at {MODEL_PATH}. Run: python -m mlscan.train"
        )
    import joblib

    return joblib.load(MODEL_PATH)


def classify(code: str) -> list[tuple[str, float]]:
    """Return [(label, probability), ...] sorted by probability, descending."""
    model = _model()
    probs = model.predict_proba([code])[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda t: t[1], reverse=True)
    return [(label, float(p)) for label, p in ranked]


def scan(code: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Scan a code snippet.

    Returns a dict with the top prediction and a list of vulnerability
    ``findings`` (non-"safe" classes scoring >= threshold).
    """
    ranked = classify(code)
    top_label, top_conf = ranked[0]

    findings = []
    for label, prob in ranked:
        if label == SAFE or prob < threshold:
            continue
        findings.append({**describe(label), "confidence": round(prob, 3)})

    return {
        "top_prediction": top_label,
        "top_confidence": round(top_conf, 3),
        "is_vulnerable": bool(findings),
        "findings": findings,
    }
