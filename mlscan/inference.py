"""The one decision rule, shared by the tuning harness and the shipped scanner.

This module exists because those two used to disagree. ``mlscan.tune`` attached
per-class offsets to the saved pipeline and reported a TEST macro-F1 measured
*with* them, while ``mlscan.scanner`` called ``predict_proba`` and took a plain
argmax — so the headline number described a rule no code path implemented. Worse,
the bake-off grid was free to pick an estimator with no ``predict_proba`` at all
(it picked ``SGDClassifier(loss='hinge')``), which the scanner cannot serve.

Everything that turns a code string into a label now goes through here:

    load_artifact(path)                  -> (pipeline, offsets | None)
    class_scores(pipeline, codes)        -> (classes, probabilities)
    apply_offsets(classes, proba, offs)  -> probabilities, re-normalised
    predict_with_offsets(pipe, codes, o) -> list[label]
    scan_scores(pipeline, code, offsets) -> [(label, probability), ...] desc

The rule
--------
``predict = argmax_c( log P(c | x) - offset[c] )``

An additive offset in log-probability space is a multiplicative correction to
the class prior; a positive offset suppresses a class, a negative one promotes
it. :func:`apply_offsets` returns the *re-normalised* posterior rather than raw
adjusted log-scores, which is deliberate: softmax is monotone, so the argmax is
bit-for-bit the argmax of ``log P - offset`` that ``mlscan.tune`` scores, while
the numbers a user sees still form a proper probability distribution that can be
compared against a single confidence threshold.

Calibrated probabilities are **required**. Reading a margin-only estimator
through ``softmax(decision_function)`` — as the bake-off does to keep every
candidate comparable — produces a number that looks like a probability, is not
one, and cannot be thresholded meaningfully. An estimator without
``predict_proba`` is therefore rejected at load time with an actionable error
rather than crashing later inside a scan.

Serving also truncates to ``mlscan.data.MAX_CODE_CHARS``, because that is the
exact string the model was fitted and evaluated on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mlscan.data import MAX_CODE_CHARS

# Guards log(0). Well below any probability a fitted model reports, so it only
# ever bites on exact zeros (ComplementNB and tree ensembles do emit them).
PROBA_EPS = 1e-12

# Filename convention written by ``mlscan.tune``: vuln_clf_v2.joblib is paired
# with thresholds_v2.json. Pairing by name keeps a set of offsets bound to the
# artifact it was tuned for — applying v2's offsets to the v1 model would be
# silently, badly wrong.
_MODEL_STEM = "vuln_clf"
_OFFSETS_STEM = "thresholds"


class NoProbabilityEstimator(TypeError):
    """Raised when an artifact cannot produce calibrated probabilities."""


class OffsetsMismatch(ValueError):
    """Raised when an offset vector does not line up with the model's classes."""


# ---------------------------------------------------------------------------
# estimator introspection
# ---------------------------------------------------------------------------

def final_estimator(pipeline):
    """The estimator that actually predicts — unwraps a sklearn ``Pipeline``."""
    steps = getattr(pipeline, "steps", None)
    return steps[-1][1] if steps else pipeline


def supports_proba(pipeline) -> bool:
    """True when this artifact can be served by the canonical rule."""
    return hasattr(final_estimator(pipeline), "predict_proba")


def require_proba(pipeline) -> None:
    """Raise a clear, actionable error if the artifact has no ``predict_proba``."""
    if supports_proba(pipeline):
        return
    est = final_estimator(pipeline)
    name = type(est).__name__
    detail = f" (loss={est.loss!r})" if getattr(est, "loss", None) else ""
    raise NoProbabilityEstimator(
        f"{name}{detail} has no predict_proba, so it cannot be scored by the "
        f"decision rule the metrics are reported under (argmax of log P(c) minus "
        f"a per-class offset). softmax(decision_function) is NOT a substitute: it "
        f"is not calibrated and cannot be thresholded. Fix this where the model is "
        f"built — either restrict the search to estimators that expose "
        f"predict_proba, or wrap this one: "
        f"CalibratedClassifierCV({name}(...), method='sigmoid', cv=3), fitted on "
        f"TRAIN only."
    )


def classes_of(pipeline) -> list[str]:
    """The model's class labels as plain ``str`` (they arrive as ``np.str_``)."""
    return [str(c) for c in pipeline.classes_]


# ---------------------------------------------------------------------------
# input preparation
# ---------------------------------------------------------------------------

def prepare_codes(codes):
    """Truncate raw code strings to what the model was trained on.

    Passed anything that is not a sequence of ``str`` (e.g. an already-vectorised
    matrix, which is how ``mlscan.tune`` calls in), this returns it untouched —
    that caller truncated at load time via :mod:`mlscan.data`.
    """
    if isinstance(codes, str):
        raise TypeError("pass a sequence of code strings, not a single string")
    try:
        is_text = all(isinstance(c, str) for c in codes)
    except TypeError:
        return codes
    return [c[:MAX_CODE_CHARS] for c in codes] if is_text else codes


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------

def class_scores(pipeline, codes) -> tuple[list[str], np.ndarray]:
    """Return ``(classes, probabilities)`` — one calibrated row per input.

    Raises :class:`NoProbabilityEstimator` if the artifact cannot produce them.
    """
    require_proba(pipeline)
    proba = np.asarray(pipeline.predict_proba(prepare_codes(codes)),
                       dtype=np.float64)
    if proba.ndim == 1:  # single-row estimators occasionally squeeze
        proba = proba.reshape(1, -1)
    return classes_of(pipeline), proba


def offsets_vector(classes, offsets) -> np.ndarray:
    """Align ``offsets`` (dict or sequence) to ``classes``; zeros for ``None``.

    A dict may omit classes (they get 0.0) but may not name classes the model
    does not have — that means the offsets belong to a different artifact.
    """
    classes = list(classes)
    if offsets is None:
        return np.zeros(len(classes), dtype=np.float64)
    if isinstance(offsets, dict):
        unknown = sorted({str(k) for k in offsets} - set(classes))
        if unknown:
            raise OffsetsMismatch(
                f"offsets name classes the model does not have: {unknown}; "
                f"model classes are {classes}. These offsets were tuned for a "
                f"different artifact."
            )
        return np.array([float(offsets.get(c, 0.0)) for c in classes],
                        dtype=np.float64)
    arr = np.asarray(offsets, dtype=np.float64).ravel()
    if arr.shape[0] != len(classes):
        raise OffsetsMismatch(
            f"offsets have length {arr.shape[0]} but the model has "
            f"{len(classes)} classes"
        )
    return arr


def apply_offsets(classes, proba, offsets) -> np.ndarray:
    """Apply additive log-space offsets and re-normalise back to probabilities.

    Argmax-identical to ``argmax(log(proba) - offsets)``, which is what
    ``mlscan.tune`` scores, so the reported metric and the shipped scanner agree
    by construction.
    """
    off = offsets_vector(classes, offsets)
    proba = np.asarray(proba, dtype=np.float64)
    if not off.any():
        return proba
    logp = np.log(np.clip(proba, PROBA_EPS, None)) - off
    logp -= logp.max(axis=1, keepdims=True)  # overflow-safe softmax
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_with_offsets(pipeline, codes, offsets=None) -> list[str]:
    """Predict a label per input under the canonical rule."""
    classes, proba = class_scores(pipeline, codes)
    adjusted = apply_offsets(classes, proba, offsets)
    return [classes[i] for i in np.argmax(adjusted, axis=1)]


def scan_scores(pipeline, code: str, offsets=None) -> list[tuple[str, float]]:
    """Score ONE snippet: ``[(label, probability), ...]`` sorted descending.

    ``scan_scores(...)[0][0] == predict_with_offsets(..., [code])[0]`` always.
    """
    classes, proba = class_scores(pipeline, [code])
    adjusted = apply_offsets(classes, proba, offsets)[0]
    ranked = zip(classes, (float(p) for p in adjusted))
    return sorted(ranked, key=lambda t: t[1], reverse=True)


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

def offsets_path_for(model_path) -> Path:
    """The offsets sidecar paired with a model file, by naming convention."""
    p = Path(model_path)
    return p.with_name(p.stem.replace(_MODEL_STEM, _OFFSETS_STEM) + ".json")


def load_offsets(path) -> dict[str, float] | None:
    """Read ``{"offsets": {...}}`` from a thresholds JSON, or ``None``."""
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("offsets")
    if not raw:
        return None
    return {str(k): float(v) for k, v in raw.items()}


def load_artifact(path) -> tuple[object, dict[str, float] | None]:
    """Load a saved pipeline and the offsets it should be served with.

    Offsets come from the paired ``thresholds*.json`` when present, else from a
    ``class_offsets_`` attribute on the pipeline, else ``None`` (plain argmax).
    They are validated against the model's classes here, so a mismatched pair
    fails at load time rather than quietly changing predictions.
    """
    import joblib

    path = Path(path)
    pipeline = joblib.load(path)
    offsets = load_offsets(offsets_path_for(path))
    if offsets is None:
        attr = getattr(pipeline, "class_offsets_", None)
        offsets = {str(k): float(v) for k, v in attr.items()} if attr else None
    if offsets is not None:
        offsets_vector(classes_of(pipeline), offsets)  # validate now, not mid-scan
    return pipeline, offsets
