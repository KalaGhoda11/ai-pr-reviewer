"""Inference: load the trained model and classify code. Offline, no API keys.

>>> from mlscan.scanner import scan
>>> scan("query = 'SELECT * FROM u WHERE id=' + user; db.execute(query)")

Scoring is delegated to :mod:`mlscan.inference`, which holds the single decision
rule this project reports metrics under — ``argmax_c(log P(c) - offset[c])``.
Nothing here reimplements it, so the number in ``model/metrics_v2.json`` and the
label a user gets from ``scan()`` cannot drift apart.
"""

from __future__ import annotations

import logging
import warnings
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

from mlscan import inference
from mlscan.labels import SAFE, describe

# Benign: LightGBM was fit on a named sparse matrix, predicts on an unnamed one.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

_MODEL_DIR = Path(__file__).resolve().parent / "model"

# The v1 artifact: 11-class LightGBM, plain argmax, no offsets. Kept as the
# public MODEL_PATH because it is the fallback that must exist for the scanner
# to work at all, and the test suite uses it as its "is anything trained?" guard.
MODEL_PATH = _MODEL_DIR / "vuln_clf.joblib"

# The v2 artifact from ``python -m mlscan.tune``: 9-class, served with the
# per-class offsets in its paired thresholds JSON. Preferred when present AND
# servable (see :func:`_artifact`).
MODEL_PATH_V2 = _MODEL_DIR / "vuln_clf_v2.joblib"

# A finding is reported when a non-"safe" class scores at/above this probability.
# Set for precision (few false positives) over recall — see docs/DESIGN.md.
DEFAULT_THRESHOLD = 0.50


class ModelNotTrained(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _artifact():
    """Return ``(pipeline, offsets, path)`` for the artifact actually served.

    v2 wins when it is present and can produce calibrated probabilities. A v2
    built on a margin-only estimator (LinearSVC, SGD+hinge) is *not* servable:
    the reported macro-F1 for such a model is measured through
    ``softmax(decision_function)``, which is not a probability and cannot be
    compared against ``DEFAULT_THRESHOLD``. Rather than crash every scan, we warn
    loudly and fall back to v1 — a stale-but-honest model beats a broken one, and
    the warning names the fix.
    """
    if MODEL_PATH_V2.exists():
        pipeline, offsets = inference.load_artifact(MODEL_PATH_V2)
        if inference.supports_proba(pipeline):
            return pipeline, offsets, MODEL_PATH_V2
        reason = (
            f"{MODEL_PATH_V2.name} cannot be served: "
            f"{type(inference.final_estimator(pipeline)).__name__} has no "
            f"predict_proba. Re-run the tuning harness so it selects (or wraps in "
            f"CalibratedClassifierCV) an estimator that does."
        )
        if not MODEL_PATH.exists():
            raise ModelNotTrained(reason)
        warnings.warn(f"{reason} Falling back to {MODEL_PATH.name}.",
                      RuntimeWarning, stacklevel=2)

    if not MODEL_PATH.exists():
        raise ModelNotTrained(
            f"No trained model at {MODEL_PATH}. Run: python -m mlscan.train"
        )
    pipeline, offsets = inference.load_artifact(MODEL_PATH)
    return pipeline, offsets, MODEL_PATH


def model_info() -> dict:
    """Which artifact and which decision rule are live (for reports/debugging)."""
    pipeline, offsets, path = _artifact()
    return {
        "model_path": str(path),
        "estimator": type(inference.final_estimator(pipeline)).__name__,
        "classes": inference.classes_of(pipeline),
        "rule": "argmax_c(log P(c) - offset[c])" if offsets else "argmax_c(P(c))",
        "offsets": offsets,
    }


def classify(code: str) -> list[tuple[str, float]]:
    """Return [(label, probability), ...] sorted by probability, descending."""
    pipeline, offsets, _ = _artifact()
    return inference.scan_scores(pipeline, code, offsets)


def scan(code: str, threshold: float = DEFAULT_THRESHOLD,
         use_rules: bool = True) -> dict:
    """Scan a code snippet (hybrid: ML classifier + deterministic rules).

    Returns a dict with the top prediction and a list of vulnerability
    ``findings`` (non-"safe" classes scoring >= threshold).

    Two independent detectors are combined, the way real SAST tooling works:

    * the **ML classifier** generalises — it flags patterns no rule was written
      for — but is probabilistic and sensitive to how code is phrased;
    * the **rule engine** (:mod:`mlscan.rules`) is deterministic and
      high-precision — it never misses ``eval(user_input)`` because a variable
      was renamed, which is exactly where the classifier is brittle.

    A finding carries ``source``: ``"ml"``, ``"rule"``, or ``"ml+rule"`` when
    both agreed. NOTE: the macro-F1 figures in ``model/metrics_v2.json`` measure
    the ML component alone; the rule layer only ever adds high-confidence hits.
    """
    ranked = classify(code)
    top_label, top_conf = ranked[0]

    by_cwe: dict[str, dict] = {}
    for label, prob in ranked:
        if label == SAFE or prob < threshold:
            continue
        by_cwe[label] = {**describe(label), "confidence": round(prob, 3),
                         "source": "ml"}

    if use_rules:
        try:
            from mlscan.rules import scan_rules
            for r in scan_rules(code):
                cwe = r["cwe"]
                extra = {"line": r.get("line"), "rule_id": r.get("rule_id"),
                         "evidence": r.get("evidence")}
                if cwe in by_cwe:  # both detectors agree - keep the stronger score
                    hit = by_cwe[cwe]
                    hit["source"] = "ml+rule"
                    hit["confidence"] = round(max(hit["confidence"],
                                                  r["confidence"]), 3)
                    hit.update(extra)
                else:
                    by_cwe[cwe] = {**describe(cwe),
                                   "confidence": round(r["confidence"], 3),
                                   "source": "rule", **extra}
        except Exception:  # noqa: BLE001 - rules must never break a scan
            logger.exception("rule engine failed; returning ML findings only")

    findings = sorted(by_cwe.values(), key=lambda f: f["confidence"], reverse=True)

    return {
        "top_prediction": top_label,
        "top_confidence": round(top_conf, 3),
        "is_vulnerable": bool(findings),
        "findings": findings,
    }
