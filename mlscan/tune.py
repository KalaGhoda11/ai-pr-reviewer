"""Bake-off harness: pick the best classifier, calibrate it, save v2 artifacts.

Two modes, and the difference matters:

    python -m mlscan.tune                     # DEV: validation only, writes metrics_dev.json
    python -m mlscan.tune --final             # FINAL: the one look at TEST, writes the v2 artifacts

Smoke test (a couple of minutes, never touches TEST):
    python -m mlscan.tune --sample 2000 --max-features 3000 --skip-lgbm

Why the split exists
--------------------
This harness used to score the TEST split on *every* invocation — smoke runs
included — print the delta against the published baseline, and overwrite the
canonical artifacts each time. That is multiple testing: after N runs the
selected configuration is partly fitted to the test sample and the reported
number is optimistic by an unknown amount. So:

* Without ``--final`` the test split is **dropped from memory immediately after
  loading** and never vectorized, scored or reported. Selection, calibration
  and reporting all happen on VALIDATION, and the run writes only
  ``model/metrics_dev.json``. The committed ``vuln_clf_v2.joblib`` /
  ``thresholds_v2.json`` / ``metrics_v2.json`` are left untouched.
* ``--final`` is the deliberate, ideally once-per-project evaluation. It is
  rejected together with ``--sample``, because a headline number produced from
  a subsampled training set is not the number the artifact deserves.

What it does, in order
----------------------
1. Loads the 9-class splits via :func:`mlscan.data.load_splits`, which now also
   de-duplicates TRAIN against validation/test and flags every eval row with
   ``dup_of_train``. The merged ``MEMORY-OOB`` taxonomy is applied there.
   ``--drop-weak-classes`` (OFF by default) removes a named class from *every*
   split before anything else runs; see :func:`partition_classes` and the
   "Reduced taxonomy" section below.
2. Builds ONE feature matrix — a :class:`~sklearn.pipeline.FeatureUnion` of
   word TF-IDF (1-2), char_wb TF-IDF (3-5) and the dense
   :class:`~mlscan.security_features.SecurityFeatures` block (followed by
   ``MaxAbsScaler``, which preserves sparsity where ``StandardScaler`` would
   densify the whole matrix). It is fitted on TRAIN ONLY and reused by every
   candidate — vectorizing is the expensive step, so it happens exactly once.
3. Bakes off LinearSVC / LogisticRegression / SGDClassifier / ComplementNB and
   one deliberately cheap LightGBM config over small explicit parameter grids.
   **Every candidate exposes ``predict_proba``**: the margin-only estimators
   (LinearSVC, SGD+hinge) are wrapped in ``CalibratedClassifierCV(cv=3)`` fitted
   on TRAIN only. A bare LinearSVC used to be selectable, and would have made
   :mod:`mlscan.scanner` raise ``AttributeError`` on the first scan.
4. Calibrates a per-class decision offset for the winner on VALIDATION (see
   :func:`tune_offsets`), which replaces plain ``argmax`` at inference, then
   estimates how much of that gain actually transfers off the tuning sample
   (see :func:`offset_transfer_estimate`) and reverts to plain argmax if the
   held-out estimate is negative.
5. Verifies that :func:`mlscan.inference.predict_with_offsets` — the function
   the scanner actually calls — reproduces the decisions the metrics are
   computed from, row for row. Every reported number goes through
   :mod:`mlscan.inference`, so "the metric" and "the deployed rule" cannot
   drift apart.
6. DEV: writes ``model/metrics_dev.json``.
   ``--final``: evaluates TEST once, reports both ``test_macro_f1`` (all rows)
   and ``test_macro_f1_unseen_only`` (rows whose code was NOT byte-identical to
   a training row — the honest generalization number), then writes
   ``model/vuln_clf_v2.joblib``, ``model/thresholds_v2.json`` and
   ``model/metrics_v2.json``.

The offset search, and what was actually measured
-------------------------------------------------
The grid bounds are now ``--offset-max`` (default 6.0) and the resolution is
``--offset-step`` (default 0.25), so the claim "the grid saturates" is testable
instead of asserted — every run reports ``saturated_classes`` and refuses to be
quiet about it. Three things were measured on the shipped v2 artifact and are
recorded here so they are not re-litigated:

* **Range is not the binding constraint.** Re-tuning from cached validation
  log-probabilities at ``--offset-max`` 2.0 / 3.0 / 6.0 / 12.0 returns the
  *bit-identical* offset vector (VAL 0.6317, TEST 0.5908). The shipped offsets
  have max magnitude 0.667, i.e. an order of magnitude inside the boundary. The
  default stays at 6.0 (wide enough that a genuinely mis-set prior could move
  there, and the saturation check will say so if one ever does), but widening it
  further is a measured no-op that only costs search time.
* **Resolution IS the binding constraint, in the wrong direction.** Refining
  ``--offset-step`` from 0.25 to 0.05 raises VALIDATION macro-F1 by +0.005 and
  *lowers* TEST macro-F1 by -0.010. A finer grid does not find a better
  correction, it fits validation noise. 0.25 is deliberate; do not lower it
  without re-measuring on test.
* **Half the tuned gain is luck.** Split-half resampling of validation (tune on
  one half, score the other) puts the expected transferred gain at roughly
  +0.005 macro-F1 with an optimism gap of ~0.027, while the shipped artifact
  realises +0.0118 on test. That is why :func:`offset_transfer_estimate` runs on
  every invocation and its numbers are written into the metrics: rare-class
  validation support here is only ~18-119 rows, so an in-sample offset gain is
  not evidence of anything on its own.

Reduced taxonomy (``--drop-weak-classes``, default OFF)
-------------------------------------------------------
CWE-200 cannot be recovered from a single snippet in this corpus: the model
scores only ~0.14 recall on CWE-200 test rows it saw *verbatim* in training
(versus 0.87-0.95 for CWE-89/CWE-502), because the defect is typically a missing
statement that the snippet does not contain. Dropping it is defensible on that
evidence — but the resulting macro-F1 is an average over fewer, easier classes,
and essentially the *whole* apparent gain is that denominator change (measured:
+0.051 from excluding CWE-200 from the average, while every surviving class moves
by at most +0.014).

So the flag defaults to OFF — the honest full-coverage 9-class number stays the
default — and when it is on, the run:

* prints a disclosure banner before and after the numbers,
* records ``weak_class_exclusion`` in the metrics with the excluded classes, the
  rows removed per split and ``comparable_to_9class: false``,
* measures the **coverage** that was given up: how many held-out rows of the
  dropped class the reduced model still flags as *some* vulnerability, and
* refuses to print the v1 baseline comparison at all, because a 7- or 8-class
  macro-F1 next to a 9-class one is not a comparison.

Cost note: the linear candidates fit in seconds on this 4-CPU box; the single
LightGBM config is the only expensive one, hence ``--skip-lgbm``. Calibration
wrapping multiplies a wrapped candidate's fit cost by roughly two (3 folds on
2/3 of the rows each). The offset stability estimate re-runs the coordinate
ascent ``--offset-stability-seeds`` times on half of validation; it is pure numpy
over a cached score matrix and costs seconds, not minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mlscan.data import SAFE_RATIO as DATA_SAFE_RATIO
from mlscan.data import SEED as DATA_SEED
from mlscan.data import Split
from mlscan.labels import CLASSES
from mlscan.labels import SAFE as SAFE_CLASS

MODEL_DIR = Path(__file__).resolve().parent / "model"
# Canonical artifacts. Written by --final ONLY.
MODEL_PATH = MODEL_DIR / "vuln_clf_v2.joblib"
THRESHOLDS_PATH = MODEL_DIR / "thresholds_v2.json"
METRICS_PATH = MODEL_DIR / "metrics_v2.json"
# Development scratch metrics. Written by every non-final run; safe to clobber.
METRICS_DEV_PATH = MODEL_DIR / "metrics_dev.json"

SEED = DATA_SEED

# The committed 11-class model's headline TEST macro-F1, which is the number
# this project has historically quoted.
BASELINE_11CLASS_MACRO_F1 = 0.581
# The same model's predictions re-scored under the 9-class taxonomy (CWE-119 /
# 787 / 125 collapsed into MEMORY-OOB). This is the *fair* bar for anything
# trained on the merged labels: the merge alone is worth ~+0.07 and is not a
# modelling improvement.
BASELINE_9CLASS_MACRO_F1 = 0.6543
# ...but that 0.6543 was earned by a model TRAINED ON THE LEAKY SPLIT: 9.1% of
# test rows are byte-identical to a row it was fitted on. Scored only on rows it
# never saw, the same model gets 0.4474 — a memorization premium of ~0.207.
# A de-duplicated model is structurally denied that premium, so all-rows-vs-
# all-rows would charge it ~0.2 macro-F1 for having FIXED the leak. The fair,
# apples-to-apples bar is unseen-only vs unseen-only.
BASELINE_9CLASS_MACRO_F1_UNSEEN_ONLY = 0.4474
# Both baselines are TEST numbers, so they are only ever printed in --final
# mode. Comparing a validation score against them is what made every smoke run
# feel like a result.


def _honest_baseline() -> dict:
    """Read the re-scored baseline from disk so the bar can't go stale."""
    path = MODEL_DIR / "baseline_honest.json"
    fallback = {"all": BASELINE_9CLASS_MACRO_F1,
                "unseen": BASELINE_9CLASS_MACRO_F1_UNSEEN_ONLY, "source": "constant"}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return {"all": d["baseline_9class_macro_f1"],
                "unseen": d["baseline_9class_macro_f1_unseen_only"],
                "source": path.name}
    except Exception:  # noqa: BLE001 - fall back to the pinned constants
        return fallback

# Per-class decision offsets are searched on this grid, in log-probability
# space (an additive log offset == a multiplicative prior correction).
# The range is deliberately wide — +/-6.0 is a ~400x prior correction either way
# — so that a badly mis-set class prior CAN move, and every run reports which
# classes (if any) actually ran into the boundary instead of assuming none did.
# Measured on the v2 artifact: bounds of 2.0/3.0/6.0/12.0 all return the same
# offset vector, so the width is currently free headroom, not a live constraint.
OFFSET_MAX = 6.0
# Resolution, NOT range, is the variable that matters — and finer is worse:
# step 0.05 buys +0.005 on validation and loses -0.010 on test (measured on the
# v2 artifact). 0.25 is a deliberate anti-overfitting choice.
OFFSET_STEP = 0.25
OFFSET_ROUNDS = 3  # 1 init pass on per-class F1 + 2 macro-F1 refinement passes
# Split-half repeats used to estimate how much of the tuned gain transfers off
# the tuning sample. Cheap (numpy over a cached score matrix) and the only
# defence against reading an in-sample offset gain as a result.
OFFSET_STABILITY_SEEDS = 5

# Classes ``--drop-weak-classes`` removes when no explicit list is given.
# CWE-200 only: the model cannot recover it even from rows it memorized (0.14
# recall on verbatim-seen test rows), so its absence is an absence of signal in
# the snippet, not a fixable modelling problem. CWE-20 is deliberately NOT here
# — it is a mislabelled residual bucket rather than an unlearnable one, its rows
# are useful negatives, and folding or dropping it was measured to cost more on
# CWE-89 than it buys.
DEFAULT_WEAK_CLASSES = ("CWE-200",)

# Folds for the CalibratedClassifierCV wrapper around margin-only estimators.
CALIBRATION_CV = 3

# Offsets are rounded to this many decimals the moment they are fitted, so the
# vector that is scored, the vector that is cross-checked against inference and
# the vector written to thresholds_v2.json are the same numbers. Rounding after
# measuring would let the shipped model differ from the reported metric on
# knife-edge rows.
OFFSET_DECIMALS = 4

_MISSING = -1e9  # log-score for a class the fitted model never saw


# ---------------------------------------------------------------------------
# the deployed decision rule (mlscan.inference) — single source of truth
# ---------------------------------------------------------------------------

def load_inference():
    """Import :mod:`mlscan.inference`, the module the scanner scores through.

    Every number this harness reports is computed with these functions, so the
    reported macro-F1 describes a decision rule that production code actually
    implements. The previous version scored with a private copy of the logic
    while :meth:`Pipeline.predict` did plain argmax — the metric described a
    classifier nothing could run.
    """
    try:
        from mlscan import inference
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SystemExit(
            "mlscan.inference is required: it defines the decision rule "
            "(class_scores + predict_with_offsets) that both this harness and "
            f"mlscan.scanner must share.\n  import failed: {exc}"
        ) from exc
    missing = [n for n in ("class_scores", "predict_with_offsets",
                           "classes_of", "PROBA_EPS")
               if not hasattr(inference, n)]
    if missing:
        raise SystemExit(
            f"mlscan.inference is missing {', '.join(missing)}; this harness "
            "computes every metric through them so that the reported score and "
            "the deployed score cannot diverge.")
    return inference


def aligned_log_scores(inference, model, X, all_classes: list[str]) -> np.ndarray:
    """Log-probabilities aligned to ``all_classes``, obtained via inference.

    ``inference.class_scores`` returns ``(classes, proba)`` in the estimator's
    own class order and rejects estimators without ``predict_proba`` outright.
    Offsets are additive in log space, so the log is taken here using the same
    epsilon ``inference.apply_offsets`` uses — the two therefore agree bit for
    bit. Classes the fitted model never saw get ``_MISSING`` and can never win
    an argmax.
    """
    classes, proba = inference.class_scores(model, X)
    logp = np.log(np.clip(np.asarray(proba, dtype=np.float64),
                          inference.PROBA_EPS, None))
    out = np.full((logp.shape[0], len(all_classes)), _MISSING, dtype=np.float64)
    idx = {c: i for i, c in enumerate(all_classes)}
    for j, c in enumerate(classes):
        out[:, idx[str(c)]] = logp[:, j]
    return out


def offsets_for_model(inference, model, all_classes, offsets) -> dict[str, float]:
    """``{class: offset}`` restricted to the estimator's own classes.

    ``inference.offsets_vector`` rejects a mapping naming classes the model does
    not have — a mismatched model/offsets pair must fail loudly rather than
    quietly change predictions. ``all_classes`` is the union over train,
    validation and test, so a class that appears only in an eval split has to be
    dropped here; it is unpredictable anyway.
    """
    model_classes = set(inference.classes_of(model))
    return {c: float(o) for c, o in zip(all_classes, offsets) if c in model_classes}


def deployed_labels(inference, model, X, all_classes, offset_map) -> np.ndarray:
    """Predictions from the function the scanner runs, as ``all_classes`` indices.

    Every per-class report in this module is built from these, so the numbers
    describe literally what the shipped artifact outputs.
    """
    idx = {c: i for i, c in enumerate(all_classes)}
    return np.asarray([idx[str(lbl)] for lbl in
                       inference.predict_with_offsets(model, X, offset_map)])


def verify_deployed_rule(scores: np.ndarray, offsets: np.ndarray,
                         deployed_idx: np.ndarray) -> dict:
    """Assert the deployed predictor reproduces the tuning objective exactly.

    :func:`tune_offsets` optimises ``argmax(log P - offset)``;
    ``inference.predict_with_offsets`` re-normalises through a softmax, which is
    monotone and therefore argmax-identical. Asserting it rather than trusting
    it costs one comparison and is the whole point of this refactor — the two
    used to differ silently. Runs on VALIDATION, before any test data is
    touched, so a mismatch fails fast instead of after a full training run.
    """
    expected = np.argmax(scores - offsets, axis=1)
    disagree = int((deployed_idx != expected).sum())
    if disagree:
        raise SystemExit(
            f"mlscan.inference.predict_with_offsets disagrees with the tuning "
            f"objective argmax(log P - offset) on {disagree}/{len(expected)} "
            f"validation rows. Every metric below would describe a rule the "
            f"scanner does not run; refusing to continue.")
    return {"checked_rows": int(len(expected)), "disagreements": 0,
            "checked_on": "validation"}


def score_source(model) -> str:
    """How scores are obtained. Always proba now — inference requires it."""
    return ("predict_proba" if hasattr(model, "predict_proba")
            else "softmax(decision_function)")


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def build_features(max_features: int = 20000, sec_weight: float = 1.0):
    """Return the unfitted FeatureUnion used for every candidate.

    ``max_features`` caps EACH TF-IDF block independently — it is the main
    CPU/RAM dial (the char block dominates both). ``sec_weight`` scales the
    dense security block relative to the two L2-normalised TF-IDF blocks.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import MaxAbsScaler

    from mlscan.security_features import SecurityFeatures

    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_features=max_features,
        sublinear_tf=True,
        token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    sec = Pipeline([
        ("extract", SecurityFeatures()),
        ("scale", MaxAbsScaler()),  # never StandardScaler: centring densifies
    ])
    return FeatureUnion(
        [("word", word), ("char", char), ("sec", sec)],
        transformer_weights={"word": 1.0, "char": 1.0, "sec": sec_weight},
    )


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

def calibration_folds(y, requested: int = CALIBRATION_CV) -> int | None:
    """Largest usable stratified fold count for the calibration wrapper.

    ``CalibratedClassifierCV`` stratifies, so it cannot use more folds than the
    rarest class has members. On the real corpus this is always ``requested``;
    it only bites on a tiny ``--sample`` smoke run, where silently dropping
    every margin-only candidate would be worse than calibrating on 2 folds.
    Returns ``None`` when even 2 folds are impossible.
    """
    counts = np.unique(np.asarray(y), return_counts=True)[1]
    usable = min(requested, int(counts.min()))
    return usable if usable >= 2 else None


def _calibrated(estimator, cv: int):
    """Wrap a margin-only estimator so the saved artifact has predict_proba.

    ``mlscan.scanner`` calls ``predict_proba``; a bare ``LinearSVC`` or
    ``SGDClassifier(loss='hinge')`` has none and would raise ``AttributeError``
    on the first scan — and the grid did previously select SGD+hinge. Platt
    scaling (``method='sigmoid'``) is used rather than isotonic because the rare
    classes have far too few rows for a non-parametric fit. ``cv`` folds are
    taken inside TRAIN only; validation and test are never seen.
    """
    from sklearn.calibration import CalibratedClassifierCV

    # sklearn >= 1.2 names this ``estimator`` (was ``base_estimator``).
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)


def candidate_grid(skip_lgbm: bool = False, class_weight: str = "none",
                   seed: int = SEED, calibration_cv: int | None = CALIBRATION_CV):
    """Yield ``(family, params, estimator)`` for every point of the small grid.

    Grids are deliberately tiny: validation support for the rare classes is
    only ~50-120 rows, so differences below roughly 0.02 macro-F1 are noise and
    a large sweep would just fit that noise.

    Two hard constraints:

    * Every yielded estimator exposes ``predict_proba`` — margin-only families
      come back wrapped by :func:`_calibrated`. If ``calibration_cv`` is None
      the data cannot support the wrapper and those families are skipped
      entirely rather than yielded unwrapped.
    * ``class_weight`` defaults to ``"none"``. :mod:`mlscan.data` already
      corrects the imbalance by down-sampling "safe" to 4:1; stacking
      ``'balanced'` on top corrects it twice and measurably over-shoots
      (validation macro-F1: 1:1+balanced 0.4987, 1:1 alone 0.5661, 4:1 alone
      0.5980). It stays selectable for an explicit A/B, but never by default.
    """
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import ComplementNB
    from sklearn.svm import LinearSVC

    weights = {"balanced": ["balanced"], "none": [None],
               "both": [None, "balanced"]}[class_weight]

    if calibration_cv is None:
        print("  (too few rows in the rarest class to calibrate; skipping "
              "LinearSVC and SGD+hinge rather than shipping a model without "
              "predict_proba)", flush=True)
    else:
        for cw in weights:
            for C in (0.5, 1.0, 2.0):
                yield ("linear_svc_cal", {"C": C, "class_weight": cw,
                                          "calibration_cv": calibration_cv},
                       _calibrated(LinearSVC(C=C, class_weight=cw,
                                             random_state=seed), calibration_cv))

    for cw in weights:
        for C in (1.0, 4.0):
            # No n_jobs: it is a no-op for the lbfgs multinomial solver and
            # oversubscribes against BLAS threads on 4 logical CPUs.
            yield ("logreg", {"C": C, "class_weight": cw},
                   LogisticRegression(C=C, class_weight=cw, max_iter=2000))

    for cw in weights:
        # modified_huber already has predict_proba; hinge does not, so it is
        # only offered wrapped.
        for loss, alpha in (("modified_huber", 1e-5), ("modified_huber", 1e-4)):
            yield ("sgd", {"loss": loss, "alpha": alpha, "class_weight": cw},
                   SGDClassifier(loss=loss, alpha=alpha, class_weight=cw,
                                 max_iter=100, tol=1e-3, random_state=seed))
        if calibration_cv is not None:
            yield ("sgd_cal", {"loss": "hinge", "alpha": 1e-5, "class_weight": cw,
                               "calibration_cv": calibration_cv},
                   _calibrated(SGDClassifier(loss="hinge", alpha=1e-5,
                                             class_weight=cw, max_iter=100,
                                             tol=1e-3, random_state=seed),
                               calibration_cv))

    # ComplementNB has no class_weight (it is designed for imbalanced text) and
    # needs non-negative input — TF-IDF and the MaxAbsScaler'd security block
    # both satisfy that.
    for alpha in (0.1, 0.5, 1.0):
        yield ("complement_nb", {"alpha": alpha}, ComplementNB(alpha=alpha))

    if skip_lgbm:
        return
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:  # noqa: BLE001 - optional dependency
        print(f"  (lightgbm unavailable, skipping: {exc})", flush=True)
        return
    # ONE cheap config. The committed 400-tree/63-leaf model took ~88 min for a
    # single fit on this box; max_bin=63 + feature_fraction=0.3 + force_col_wise
    # is what keeps histogram building on a wide sparse matrix affordable.
    params = {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.15,
              "max_bin": 63, "feature_fraction": 0.3, "min_child_samples": 10}
    yield ("lightgbm", params, LGBMClassifier(
        objective="multiclass", class_weight=weights[0], force_col_wise=True,
        n_jobs=4, random_state=seed, verbose=-1, **params))


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------

def _f1_per_class(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """Vectorised per-class F1 over integer-encoded labels (fast inner loop)."""
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)


def make_offset_grid(max_abs: float = OFFSET_MAX,
                     step: float = OFFSET_STEP) -> np.ndarray:
    """Symmetric search grid ``[-max_abs, +max_abs]`` in increments of ``step``.

    Built from an integer count rather than ``linspace`` so that 0.0 is always a
    grid point exactly (the fallback "no correction" must be reachable) and so
    that the boundary is exactly ``+/-max_abs`` — :func:`tune_offsets` compares
    against it to decide whether a coordinate saturated.
    """
    if max_abs <= 0:
        raise ValueError(f"--offset-max must be positive, got {max_abs}")
    if step <= 0 or step > max_abs:
        raise ValueError(
            f"--offset-step must be in (0, {max_abs}], got {step}")
    n = int(round(max_abs / step))
    return np.round(np.arange(-n, n + 1, dtype=np.float64) * step, 6)


def tune_offsets(scores: np.ndarray, y_true_idx: np.ndarray, k: int,
                 grid: np.ndarray | None = None,
                 rounds: int = OFFSET_ROUNDS) -> tuple[np.ndarray, dict]:
    """Fit one additive decision offset per class by coordinate ascent.

    The decision rule becomes ``argmax_c(log P(c) - offset[c])`` instead of
    plain argmax; a positive offset suppresses a class, a negative one promotes
    it. This is the multiclass generalisation of a per-class threshold and it
    can fix classes whose errors point in OPPOSITE directions (one
    over-predicted, another under-predicted), which no single global
    ``class_weight`` can do.

    The first pass sets each offset to maximise THAT class's F1; later passes
    refine on macro-F1, because a per-class offset also moves the other classes
    through the shared argmax. Zeros are always a fallback, so the result can
    never score below plain argmax on validation.

    Adding a constant to every offset leaves the argmax unchanged, so the
    solution is only identifiable up to a shift: ties are broken toward the
    smallest magnitude and the result is mean-centred, which keeps the saved
    numbers readable ("this class is being suppressed / promoted relative to
    the rest") without altering a single prediction.

    A coordinate that comes to rest on the first or last grid point is
    **saturated**: the search wanted to go further and the grid stopped it, so
    the reported gain is capped by an arbitrary constant. Those classes are
    returned in ``info['saturated_classes']`` and the caller prints a warning —
    the alternative (assuming saturation never happens, or assuming it always
    does) is how ``OFFSET_MAX`` acquired a folklore justification in the first
    place.

    Must be called with VALIDATION scores only — this is a fitting procedure,
    and fitting it on test is exactly the leak this module now prevents.
    """
    if grid is None:
        grid = make_offset_grid()
    grid = np.asarray(grid, dtype=np.float64)
    grid_max = float(np.abs(grid).max())
    # Sweep small offsets first so a plateau resolves to the least aggressive
    # correction rather than wherever the grid happened to start.
    grid = grid[np.argsort(np.abs(grid), kind="stable")]
    offsets = np.zeros(k, dtype=np.float64)
    base = _f1_per_class(y_true_idx, np.argmax(scores, axis=1), k)
    best_macro = float(base.mean())
    history = [{"round": 0, "objective": "argmax", "val_macro_f1": round(best_macro, 4)}]

    for r in range(rounds):
        per_class_objective = (r == 0)
        for c in range(k):
            trial = offsets.copy()
            best_val, best_off = -1.0, offsets[c]
            for g in grid:
                trial[c] = g
                f1 = _f1_per_class(y_true_idx,
                                   np.argmax(scores - trial, axis=1), k)
                val = float(f1[c]) if per_class_objective else float(f1.mean())
                if val > best_val:
                    best_val, best_off = val, float(g)
            offsets[c] = best_off
        macro = float(_f1_per_class(
            y_true_idx, np.argmax(scores - offsets, axis=1), k).mean())
        history.append({
            "round": r + 1,
            "objective": "per_class_f1" if per_class_objective else "macro_f1",
            "val_macro_f1": round(macro, 4),
        })
        if macro <= best_macro + 1e-6 and r > 0:
            break  # converged; extra rounds buy nothing
        best_macro = max(best_macro, macro)

    # Saturation is measured on the RAW offsets, before mean-centring shifts
    # them off the grid: hitting the boundary is a property of the search, not
    # of the readable vector that comes out of it.
    saturated = [int(c) for c in np.flatnonzero(
        np.abs(offsets) >= grid_max - 1e-9)]

    # Mean-centring is argmax-invariant and makes the vector readable ("this
    # class is suppressed relative to the rest"); rounding here — before the
    # score below is measured — is what keeps reported and shipped identical.
    offsets = np.round(offsets - offsets.mean(), OFFSET_DECIMALS)
    tuned = float(_f1_per_class(
        y_true_idx, np.argmax(scores - offsets, axis=1), k).mean())
    if tuned < base.mean():  # never ship a calibration that hurts validation
        offsets = np.zeros(k, dtype=np.float64)
        tuned = float(base.mean())
        saturated = []
    return offsets, {"history": history,
                     "val_macro_f1_argmax": round(float(base.mean()), 4),
                     "val_macro_f1_tuned": round(tuned, 4),
                     "grid": {"max_abs": round(grid_max, 6),
                              "step": (round(float(np.diff(np.sort(grid))[0]), 6)
                                       if grid.size > 1 else 0.0),
                              "points": int(grid.size)},
                     "saturated_class_indices": saturated}


def offset_transfer_estimate(scores: np.ndarray, y_true_idx: np.ndarray, k: int,
                             grid: np.ndarray | None = None,
                             rounds: int = OFFSET_ROUNDS,
                             seeds: int = OFFSET_STABILITY_SEEDS,
                             rng_seed: int = SEED) -> dict | None:
    """Split-half estimate of how much of the offset gain survives transfer.

    The offsets are fitted on validation, and the rare classes have only ~18-119
    validation rows each. An in-sample gain over that support is not evidence:
    coordinate ascent over 49 grid points x 9 classes x 3 rounds has plenty of
    freedom to fit the sample. So this repeatedly tunes on a random HALF of
    validation and scores the untouched other half, which is the same
    fit-then-transfer motion the real val -> test hand-off performs, but
    measurable without spending the one look at test.

    Reports the optimism gap (in-sample minus held-out gain) and the sd of the
    held-out gain across repeats. A held-out gain that is negative on average
    means the calibration is fitting validation noise, and
    :func:`main` then reverts to plain argmax unless ``--no-offset-guard``.

    Returns ``None`` when the estimate would be meaningless (no repeats
    requested, or too few rows to halve). Pure numpy over an already-computed
    score matrix — no refitting, so it costs seconds.
    """
    n = len(y_true_idx)
    if seeds <= 0 or n < 200:
        return None
    if grid is None:
        grid = make_offset_grid()
    rng = np.random.default_rng(rng_seed)
    in_gains: list[float] = []
    out_gains: list[float] = []
    for _ in range(int(seeds)):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        offs, _info = tune_offsets(scores[a], y_true_idx[a], k, grid, rounds)
        for idx, bucket in ((a, in_gains), (b, out_gains)):
            plain = float(_f1_per_class(
                y_true_idx[idx], np.argmax(scores[idx], axis=1), k).mean())
            tuned = float(_f1_per_class(
                y_true_idx[idx], np.argmax(scores[idx] - offs, axis=1), k).mean())
            bucket.append(tuned - plain)
    in_arr, out_arr = np.asarray(in_gains), np.asarray(out_gains)
    return {
        "method": "split-half on VALIDATION: tune on one half, score the other",
        "seeds": int(seeds),
        "n_per_half": int(n // 2),
        "in_sample_gain_mean": round(float(in_arr.mean()), 4),
        "held_out_gain_mean": round(float(out_arr.mean()), 4),
        "held_out_gain_sd": round(float(out_arr.std(ddof=0)), 4),
        "optimism": round(float(in_arr.mean() - out_arr.mean()), 4),
        "note": "held_out_gain_mean is the honest expected effect size of the "
                "offset calibration; the gain measured on the full validation "
                "split is in-sample and is optimistic by roughly 'optimism'.",
    }


# ---------------------------------------------------------------------------
# reduced taxonomy (--drop-weak-classes)
# ---------------------------------------------------------------------------

def partition_classes(split: Split, drop: set[str]) -> tuple[Split, Split]:
    """Split into ``(kept, removed)`` by label, preserving ``dup_of_train``.

    The removed rows are not thrown away: they are the only way to measure what
    dropping a class COSTS. A reduced-taxonomy run scores them to report how many
    are still flagged as some vulnerability by the surviving classes, which is
    the coverage number that has to sit next to the improved macro-F1.
    """
    keep_idx = [i for i, lbl in enumerate(split.labels) if lbl not in drop]
    drop_idx = [i for i, lbl in enumerate(split.labels) if lbl in drop]

    def _take(idx: list[int]) -> Split:
        dup = (None if split.dup_of_train is None
               else [split.dup_of_train[i] for i in idx])
        return Split(codes=[split.codes[i] for i in idx],
                     labels=[split.labels[i] for i in idx], dup_of_train=dup)

    return _take(keep_idx), _take(drop_idx)


def coverage_loss(inference, model, features, removed: Split, split_name: str,
                  offset_map: dict[str, float]) -> dict | None:
    """What the reduced model does with the rows whose class was removed.

    Those rows are real vulnerabilities that the scanner can no longer NAME. The
    only remaining question is whether it still flags them at all, so this
    reports the fraction predicted as something other than ``safe`` — a number
    that must be published in the same breath as the higher macro-F1, because
    the macro-F1 gain is almost entirely the smaller denominator.
    """
    if not removed.codes:
        return None
    preds = inference.predict_with_offsets(
        model, features.transform(removed.codes), offset_map)
    flagged = sum(1 for p in preds if str(p) != "safe")
    counts: dict[str, int] = {}
    for p in preds:
        counts[str(p)] = counts.get(str(p), 0) + 1
    return {
        "split": split_name,
        "n_rows": len(preds),
        "still_flagged_as_some_vulnerability": int(flagged),
        "fraction_still_flagged": round(flagged / len(preds), 4),
        "predicted_as": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "note": "these rows carry an excluded class, so the scanner can no "
                "longer name them; 'fraction_still_flagged' is the share it at "
                "least still calls vulnerable.",
    }


def evaluate(y_idx: np.ndarray, deployed_idx: np.ndarray, scores: np.ndarray,
             all_classes: list[str], mask: np.ndarray | None = None) -> dict:
    """Macro-F1 under the deployed rule, plus the plain-argmax reference.

    ``deployed_idx`` comes from :func:`deployed_labels`, i.e. from
    ``mlscan.inference.predict_with_offsets`` — the headline ``macro_f1`` and
    the per-class report therefore describe the shipped artifact's own output.
    ``macro_f1_argmax`` is only kept as the uncalibrated reference the offset
    gain is measured against.

    ``mask`` selects a row subset, so the same code path produces the all-rows
    number and the unseen-only one.
    """
    from sklearn.metrics import classification_report

    k = len(all_classes)
    if mask is not None:
        y_idx, deployed_idx, scores = y_idx[mask], deployed_idx[mask], scores[mask]
    if len(y_idx) == 0:
        return {"n": 0, "macro_f1_argmax": None, "macro_f1": None,
                "classification_report": "(no rows)"}
    report = classification_report(
        [all_classes[i] for i in y_idx], [all_classes[i] for i in deployed_idx],
        digits=3, zero_division=0, labels=all_classes)
    return {
        "n": int(len(y_idx)),
        "macro_f1_argmax": round(float(_f1_per_class(
            y_idx, np.argmax(scores, axis=1), k).mean()), 4),
        "macro_f1": round(float(_f1_per_class(y_idx, deployed_idx, k).mean()), 4),
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

def stratified_subsample(split: Split, n: int, seed: int = SEED) -> Split:
    """Down-sample a split to ~``n`` rows, keeping every class represented.

    ``dup_of_train`` is carried through with the same index selection, so a
    subsampled eval split can still report unseen-only metrics.
    """
    if n <= 0 or n >= len(split.codes):
        return split
    labels = np.asarray(split.labels)
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        take = max(1, int(round(len(idx) * n / len(labels))))
        take = min(take, len(idx))
        keep.extend(rng.choice(idx, size=take, replace=False).tolist())
    keep_arr = rng.permutation(np.asarray(keep, dtype=int))
    dup = (None if split.dup_of_train is None
           else [split.dup_of_train[i] for i in keep_arr])
    return Split(codes=[split.codes[i] for i in keep_arr],
                 labels=[split.labels[i] for i in keep_arr],
                 dup_of_train=dup)


def _label_counts(labels) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


def _dup_mask(split: Split, name: str) -> np.ndarray | None:
    """Boolean mask of rows NOT byte-identical to a training row."""
    if split.dup_of_train is None:
        print(f"  warning: {name} split carries no dup_of_train flags; "
              "unseen-only metrics unavailable", flush=True)
        return None
    return ~np.asarray(split.dup_of_train, dtype=bool)


def _print_reduced_taxonomy_reminder(dropped: list[str], k: int) -> None:
    """Repeat the disclosure after the numbers, where it will actually be read.

    A banner printed 200 lines above a macro-F1 does not travel with the macro-F1
    when someone copies it into a report; one printed immediately after it has a
    chance to.
    """
    if not dropped:
        return
    print("\n" + "!" * 72, flush=True)
    print(f"!! The macro-F1 above averages {k} classes, NOT 9: {dropped} were "
          f"excluded.\n!! It is NOT comparable to any 9-class figure, and the "
          f"gain from excluding a\n!! class is almost entirely the smaller "
          f"denominator. Quote it only together\n!! with the coverage_loss block "
          f"in the metrics JSON.", flush=True)
    print("!" * 72, flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m mlscan.tune",
        description="Bake off classifiers on shared features, calibrate on "
                    "validation, and (only with --final) score TEST once.")
    p.add_argument("--final", action="store_true",
                   help="THE final evaluation: score TEST once and overwrite the "
                        "canonical v2 artifacts. Without it the test split is "
                        "discarded on load and only metrics_dev.json is written.")
    p.add_argument("--max-features", type=int, default=20000,
                   help="max_features for EACH TF-IDF block (default: 20000)")
    p.add_argument("--skip-lgbm", action="store_true",
                   help="skip the LightGBM candidate (it is the only slow fit)")
    p.add_argument("--sample", type=int, default=0,
                   help="limit TRAIN to ~N stratified rows for a quick smoke run "
                        "(rejected with --final)")
    p.add_argument("--safe-ratio", type=float, default=None,
                   help="safe:vuln down-sampling ratio for TRAIN (default: mlscan.data.SAFE_RATIO = 4.0)")
    p.add_argument("--sec-weight", type=float, default=1.0,
                   help="weight of the dense SecurityFeatures block (default: 1.0)")
    p.add_argument("--class-weight", choices=("balanced", "none", "both"),
                   default="none",
                   help="class_weight to bake off, where supported (default: none). "
                        "mlscan.data already down-samples safe to 4:1; stacking "
                        "'balanced' on top corrects the imbalance twice and "
                        "measurably over-shoots (0.4987 vs 0.5980 val macro-F1).")
    p.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=True,
                   help="drop TRAIN rows byte-identical to a val/test row "
                        "(default: on; --no-dedup reproduces the leaky splits)")
    p.add_argument("--offset-max", type=float, default=OFFSET_MAX,
                   help=f"half-width of the per-class offset search grid, in "
                        f"log-probability space (default: {OFFSET_MAX}). Every "
                        f"run reports which classes, if any, saturated it. "
                        f"Measured on the v2 artifact: 2.0/3.0/6.0/12.0 all "
                        f"return the identical offset vector, so widening is "
                        f"free headroom, not a gain.")
    p.add_argument("--offset-step", type=float, default=OFFSET_STEP,
                   help=f"offset grid resolution (default: {OFFSET_STEP}). Do "
                        f"NOT lower this casually: 0.05 measured +0.005 on "
                        f"validation and -0.010 on TEST -- it fits the tuning "
                        f"split, it does not find a better correction.")
    p.add_argument("--offset-stability-seeds", type=int,
                   default=OFFSET_STABILITY_SEEDS,
                   help=f"split-half repeats used to estimate how much of the "
                        f"offset gain transfers off the tuning sample "
                        f"(default: {OFFSET_STABILITY_SEEDS}; 0 disables)")
    p.add_argument("--offset-guard", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="revert to plain argmax when the split-half estimate "
                        "says the offset gain does not transfer (default: on)")
    p.add_argument("--drop-weak-classes", action="store_true",
                   help="EXCLUDE weak classes (default: "
                        f"{','.join(DEFAULT_WEAK_CLASSES)}) from training AND "
                        "from the reported macro average. OFF by default: the "
                        "resulting macro-F1 is over fewer, easier classes and "
                        "is NOT comparable to the 9-class figure, so it is only "
                        "ever a deliberate, disclosed choice.")
    p.add_argument("--weak-classes", type=str,
                   default=",".join(DEFAULT_WEAK_CLASSES),
                   help="comma-separated classes --drop-weak-classes removes "
                        f"(default: {','.join(DEFAULT_WEAK_CLASSES)})")
    p.add_argument("--seed", type=int, default=SEED, help="random seed")
    p.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True,
                   help="force HuggingFace offline mode (the dataset is cached locally)")
    args = p.parse_args(argv)
    if args.final and args.sample:
        p.error("--final may not be combined with --sample: the one look at the "
                "test set must use the full training data, not a smoke-sized "
                "subsample. Run the sampled configuration without --final first.")
    try:
        make_offset_grid(args.offset_max, args.offset_step)
    except ValueError as exc:
        p.error(str(exc))
    args.weak_classes = [c.strip() for c in args.weak_classes.split(",") if c.strip()]
    unknown = [c for c in args.weak_classes if c not in CLASSES]
    if unknown:
        p.error(f"--weak-classes names {unknown}, which are not in the taxonomy "
                f"({CLASSES}).")
    if SAFE_CLASS in args.weak_classes:
        p.error(f"--weak-classes may not contain {SAFE_CLASS!r}: dropping the "
                "negative class would leave a model that can only ever answer "
                "'vulnerable'.")
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    MODEL_DIR.mkdir(exist_ok=True)

    # Benign, and it would otherwise interleave with the bake-off table:
    # LightGBM names its columns at fit time, we score on an unnamed matrix.
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    if args.offline:  # load_dataset otherwise stalls on a network check
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import joblib
    from sklearn.pipeline import Pipeline

    from mlscan.data import load_splits

    inference = load_inference()

    t_start = time.time()
    if args.final:
        print("=== FINAL RUN: the test split will be scored ONCE and the v2 "
              "artifacts overwritten ===", flush=True)
    else:
        print("=== DEV RUN: validation only. The test split is discarded on "
              "load; v2 artifacts are not touched. Use --final for the real "
              "evaluation. ===", flush=True)

    print("Loading + preparing data (9-class merged taxonomy)...", flush=True)
    safe_ratio = DATA_SAFE_RATIO if args.safe_ratio is None else args.safe_ratio
    splits = load_splits(safe_ratio=safe_ratio, dedup=args.dedup)

    if not args.final:
        # Structural, not merely a promise: nothing downstream can reach the
        # test rows because they are no longer in the dict.
        splits.pop("test", None)

    # --- optional reduced taxonomy, OFF by default ------------------------
    dropped_classes = list(args.weak_classes) if args.drop_weak_classes else []
    removed_rows: dict[str, Split] = {}
    if dropped_classes:
        drop_set = set(dropped_classes)
        print("\n" + "!" * 72, flush=True)
        print(f"!! REDUCED TAXONOMY: {dropped_classes} excluded from training "
              f"and from the\n!! reported macro average. The macro-F1 below is "
              f"an average over FEWER,\n!! EASIER classes and is NOT comparable "
              f"to the 9-class figure.", flush=True)
        print("!" * 72 + "\n", flush=True)
        for name in list(splits):
            kept, removed = partition_classes(splits[name], drop_set)
            splits[name] = kept
            removed_rows[name] = removed
            print(f"  {name}: removed {len(removed.codes)} rows "
                  f"({len(removed.codes) / max(len(removed.codes) + len(kept.codes), 1):.2%} "
                  f"of the split)", flush=True)

    if args.sample:
        splits["train"] = stratified_subsample(splits["train"], args.sample,
                                               args.seed)
        print(f"  --sample {args.sample}: train reduced to "
              f"{len(splits['train'].codes)} rows", flush=True)
    for name in ("train", "validation", "test"):
        if name in splits:
            s = splits[name]
            print(f"  {name}: {len(s.codes)} rows, {len(set(s.labels))} classes",
                  flush=True)
    if not args.final:
        print("  test: NOT LOADED (dev run)", flush=True)

    ytr, yval = splits["train"].labels, splits["validation"].labels
    yte = splits["test"].labels if args.final else []
    all_classes = sorted(set(ytr) | set(yval) | set(yte))
    k = len(all_classes)
    index = {c: i for i, c in enumerate(all_classes)}
    yval_idx = np.asarray([index[c] for c in yval])

    # --- vectorize ONCE, fit on TRAIN ONLY (no leakage) -------------------
    print(f"Vectorizing (fit on TRAIN only, max_features={args.max_features} "
          f"per TF-IDF block)...", flush=True)
    features = build_features(args.max_features, args.sec_weight)
    t0 = time.time()
    Xtr = features.fit_transform(splits["train"].codes)
    Xval = features.transform(splits["validation"].codes)
    vec_secs = time.time() - t0
    print(f"  {Xtr.shape[1]} features, {Xtr.nnz / max(Xtr.shape[0], 1):.0f} nnz/doc "
          f"({vec_secs:.1f}s)", flush=True)

    # --- bake off (selection on VALIDATION, never on TEST) ----------------
    cal_cv = calibration_folds(ytr)
    if cal_cv is not None and cal_cv != CALIBRATION_CV:
        print(f"  (rarest train class has {cal_cv} rows; calibrating "
              f"margin-only candidates with cv={cal_cv})", flush=True)
    print("Baking off candidates (selection on VALIDATION macro-F1)...", flush=True)
    results: list[dict] = []
    best = {"val_macro_f1": -1.0}
    best_model = None
    best_val_scores = None
    for family, params, model in candidate_grid(args.skip_lgbm,
                                                args.class_weight, args.seed,
                                                cal_cv):
        t0 = time.time()
        try:
            model.fit(Xtr, ytr)
            if not hasattr(model, "predict_proba"):
                # Belt and braces: mlscan.scanner calls predict_proba, so a
                # candidate without it can never be shipped and must not be
                # allowed to win a bake-off it could not honour.
                raise TypeError("no predict_proba; would break mlscan.scanner")
            # Score through the SAME path the deployed model uses, so selection
            # and deployment agree. (They can differ: SGD+modified_huber's
            # predict() argmaxes the raw margins while predict_proba() clips
            # them, which can flip a prediction.)
            scores = aligned_log_scores(inference, model, Xval, all_classes)
            val_f1 = float(_f1_per_class(
                yval_idx, np.argmax(scores, axis=1), k).mean())
        except Exception as exc:  # noqa: BLE001 - one bad point must not kill the run
            # NoProbabilityEstimator lands here too, which is the point: a
            # candidate the scanner could not serve is recorded as failed
            # rather than allowed to win.
            print(f"  {family} {params}: FAILED ({exc})", flush=True)
            results.append({"family": family, "params": params, "error": str(exc)})
            continue
        secs = time.time() - t0
        results.append({"family": family, "params": params,
                        "val_macro_f1": round(val_f1, 4), "fit_secs": round(secs, 1)})
        print(f"  {family:<15} {params}  val macro-F1 = {val_f1:.4f}  ({secs:.1f}s)",
              flush=True)
        if val_f1 > best["val_macro_f1"]:
            best = results[-1]
            best_model = model
            best_val_scores = scores

    if best_model is None:
        print("error: every candidate failed to fit", flush=True)
        return 2
    print(f"\nWinner: {best['family']} {best['params']} "
          f"(validation macro-F1 = {best['val_macro_f1']:.4f})", flush=True)

    # --- per-class threshold calibration, on VALIDATION only --------------
    grid = make_offset_grid(args.offset_max, args.offset_step)
    print(f"Calibrating per-class decision offsets on VALIDATION "
          f"(grid +/-{args.offset_max} step {args.offset_step}, "
          f"{grid.size} points)...", flush=True)
    t0 = time.time()
    offsets, offset_info = tune_offsets(best_val_scores, yval_idx, k, grid)
    print(f"  argmax {offset_info['val_macro_f1_argmax']:.4f} -> tuned "
          f"{offset_info['val_macro_f1_tuned']:.4f} ({time.time() - t0:.1f}s)",
          flush=True)
    for c, off in zip(all_classes, offsets):
        print(f"    {c:<12} offset {off:+.2f}", flush=True)

    # Saturation is now measured rather than assumed in a comment. If nothing
    # reaches the boundary, the grid width is not what limits the gain and
    # widening it further is wasted search time.
    saturated = [all_classes[i] for i in offset_info["saturated_class_indices"]]
    offset_info["saturated_classes"] = saturated
    if saturated:
        print(f"  WARNING: {saturated} saturated the +/-{args.offset_max} grid: "
              f"the search wanted to go further and the bound stopped it. "
              f"Re-run with a larger --offset-max before trusting these offsets.",
              flush=True)
    else:
        print(f"  no class reached the +/-{args.offset_max} boundary "
              f"(largest |offset| = {float(np.abs(offsets).max()):.3f}), so the "
              f"grid width is not the binding constraint here", flush=True)

    # How much of that gain is real? Fitted on ~18-119 rows for the rare
    # classes, an in-sample gain proves nothing, so measure the transfer.
    stability = offset_transfer_estimate(best_val_scores, yval_idx, k, grid,
                                         seeds=args.offset_stability_seeds,
                                         rng_seed=args.seed)
    if stability:
        print(f"  offset gain, split-half over validation ({stability['seeds']} "
              f"repeats, n={stability['n_per_half']}/half):", flush=True)
        print(f"    in-sample {stability['in_sample_gain_mean']:+.4f} -> "
              f"held-out {stability['held_out_gain_mean']:+.4f} "
              f"(sd {stability['held_out_gain_sd']:.4f}, optimism "
              f"{stability['optimism']:+.4f})", flush=True)
        print("    ^ held-out is the honest expected effect size; the full-"
              "validation gain above is in-sample.", flush=True)
        if stability["held_out_gain_mean"] < 0:
            if args.offset_guard:
                print("  GUARD: the offset calibration does not transfer "
                      "(held-out gain is negative). Reverting to plain argmax "
                      "(pass --no-offset-guard to ship it anyway).", flush=True)
                offsets = np.zeros(k, dtype=np.float64)
                offset_info["reverted_to_argmax"] = True
                offset_info["val_macro_f1_tuned"] = \
                    offset_info["val_macro_f1_argmax"]
                offset_info["saturated_classes"] = []
            else:
                print("  WARNING: the offset calibration does not transfer "
                      "(held-out gain is negative) and --no-offset-guard is "
                      "set, so it is being shipped anyway.", flush=True)
                offset_info["reverted_to_argmax"] = False
    offset_info["stability"] = stability
    offset_info["grid"]["configured_max_abs"] = args.offset_max
    offset_info["grid"]["configured_step"] = args.offset_step

    # --- the reported rule IS the deployed rule ---------------------------
    print("Checking mlscan.inference.predict_with_offsets reproduces these "
          "decisions...", flush=True)
    threshold_map = offsets_for_model(inference, best_model, all_classes, offsets)
    val_pred = deployed_labels(inference, best_model, Xval, all_classes,
                               threshold_map)
    rule_check = verify_deployed_rule(best_val_scores, offsets, val_pred)
    print(f"  OK: {rule_check['checked_rows']} validation rows, 0 disagreements",
          flush=True)
    eval_only = [c for c in all_classes if c not in threshold_map]
    if eval_only:
        print(f"  note: {eval_only} appear only in eval splits; the model cannot "
              "predict them and they carry no offset", flush=True)

    val_unseen = _dup_mask(splits["validation"], "validation")
    val_all = evaluate(yval_idx, val_pred, best_val_scores, all_classes)
    val_unseen_eval = (evaluate(yval_idx, val_pred, best_val_scores, all_classes,
                                val_unseen) if val_unseen is not None else None)
    print(f"VALIDATION macro-F1 (tuned) = {val_all['macro_f1']:.4f}"
          f"  [macro over {k} classes]", flush=True)
    if val_unseen_eval:
        print(f"VALIDATION macro-F1 (tuned, unseen-only, n={val_unseen_eval['n']}) "
              f"= {val_unseen_eval['macro_f1']:.4f}", flush=True)

    # --- what the reduced taxonomy cost, measured ---------------------------
    weak_exclusion = {
        "enabled": bool(dropped_classes),
        "excluded_classes": dropped_classes,
        "n_classes_in_macro": k,
        "comparable_to_9class": not dropped_classes,
    }
    if dropped_classes:
        weak_exclusion["rows_removed"] = {
            name: len(s.codes) for name, s in removed_rows.items()}
        weak_exclusion["coverage_loss"] = {}
        cov = coverage_loss(inference, best_model, features,
                            removed_rows.get("validation", Split([], [])),
                            "validation", threshold_map)
        if cov:
            weak_exclusion["coverage_loss"]["validation"] = cov
            print(f"  coverage given up on VALIDATION: {cov['n_rows']} rows of "
                  f"{dropped_classes} can no longer be named; "
                  f"{cov['fraction_still_flagged']:.1%} are still flagged as "
                  f"some vulnerability", flush=True)
        weak_exclusion["note"] = (
            f"macro-F1 here is averaged over {k} classes, not 9. Excluding a "
            f"class raises the macro average almost entirely through the "
            f"DENOMINATOR — measured on the v2 artifact, dropping CWE-200 moved "
            f"macro-F1 +0.0513 while no surviving class gained more than "
            f"+0.0135. This number must NOT be reported next to, or as an "
            f"improvement on, the 9-class figure, and the coverage_loss block "
            f"must be quoted with it.")
    else:
        weak_exclusion["note"] = (
            "full 9-class coverage (default). No class was excluded from "
            "training or from the macro average.")

    common = {
        "dataset": "ayshajavd/code-security-vulnerability-dataset",
        "taxonomy": CLASSES,
        "classes_seen": all_classes,
        "config": {
            "final": args.final,
            "max_features_per_block": args.max_features,
            "sec_weight": args.sec_weight,
            "class_weight": args.class_weight,
            "sample": args.sample or None,
            "safe_ratio": safe_ratio,
            "dedup": args.dedup,
            "skip_lgbm": args.skip_lgbm,
            "calibration_cv": cal_cv,
            "seed": args.seed,
            "offset_max": args.offset_max,
            "offset_step": args.offset_step,
            "offset_stability_seeds": args.offset_stability_seeds,
            "offset_guard": args.offset_guard,
            "drop_weak_classes": bool(dropped_classes),
        },
        "n_train": len(ytr), "n_val": len(yval),
        "n_features": int(Xtr.shape[1]),
        "train_label_counts": _label_counts(ytr),
        "vectorize_secs": round(vec_secs, 1),
        "candidates": results,
        "best": best,
        "thresholds": threshold_map,
        "classes_eval_only": eval_only,
        "threshold_tuning": offset_info,
        "weak_class_exclusion": weak_exclusion,
        "decision_rule": {
            "rule": "predict = argmax_c(log P(c) - offset[c])",
            "implemented_by": "mlscan.inference.predict_with_offsets",
            "score_source": score_source(best_model),
            "verified_against_inference": rule_check,
        },
        "validation": val_all,
        "validation_unseen_only": val_unseen_eval,
    }

    # --- DEV: stop here. TEST was never loaded. ---------------------------
    if not args.final:
        METRICS_DEV_PATH.write_text(json.dumps({
            **common,
            "test": "NOT EVALUATED (dev run). Re-run with --final for the "
                    "single sanctioned test evaluation.",
            "total_secs": round(time.time() - t_start, 1),
        }, indent=2), encoding="utf-8")
        print(f"\nSaved dev metrics -> {METRICS_DEV_PATH}", flush=True)
        print(f"v2 artifacts untouched ({MODEL_PATH.name}, "
              f"{THRESHOLDS_PATH.name}, {METRICS_PATH.name})", flush=True)
        _print_reduced_taxonomy_reminder(dropped_classes, k)
        print(f"Done in {time.time() - t_start:.1f}s", flush=True)
        return 0

    # --- TEST, exactly once ------------------------------------------------
    print("\nEvaluating on TEST (first and only look)...", flush=True)
    Xte = features.transform(splits["test"].codes)
    yte_idx = np.asarray([index[c] for c in yte])
    test_scores = aligned_log_scores(inference, best_model, Xte, all_classes)
    test_pred = deployed_labels(inference, best_model, Xte, all_classes,
                                threshold_map)

    unseen = _dup_mask(splits["test"], "test")
    test_all = evaluate(yte_idx, test_pred, test_scores, all_classes)
    test_unseen = (evaluate(yte_idx, test_pred, test_scores, all_classes, unseen)
                   if unseen is not None else None)

    print("\n--- TEST, all rows ---", flush=True)
    print(test_all["classification_report"], flush=True)
    print(f"test_macro_f1             = {test_all['macro_f1']:.4f} "
          f"(argmax {test_all['macro_f1_argmax']:.4f}, n={test_all['n']}, "
          f"macro over {k} classes)", flush=True)

    if test_unseen:
        n_dup = test_all["n"] - test_unseen["n"]
        print(f"\n--- TEST, unseen only ({test_unseen['n']} rows; {n_dup} "
              f"duplicated a TRAIN row and are excluded) ---", flush=True)
        print(test_unseen["classification_report"], flush=True)
        print(f"test_macro_f1_unseen_only = {test_unseen['macro_f1']:.4f} "
              f"(argmax {test_unseen['macro_f1_argmax']:.4f})", flush=True)
        if getattr(args, "dedup", True):
            print("  ^ this model was trained WITHOUT those duplicates, so its "
                  "all-rows figure is already honest. This slice exists so it "
                  "and the leaky-trained baseline can be compared on rows "
                  "NEITHER of them saw.", flush=True)
        else:
            print("  ^ this is the generalization number. The all-rows figure "
                  "is inflated by memorized duplicates.", flush=True)

    # Offsets were fitted on validation; how much of the gain survives the
    # transfer is the only honest read on whether the calibration generalises.
    val_gain = val_all["macro_f1"] - val_all["macro_f1_argmax"]
    test_gain = test_all["macro_f1"] - test_all["macro_f1_argmax"]
    print(f"\nOffset transfer (tuned - argmax): validation {val_gain:+.4f} -> "
          f"test {test_gain:+.4f} (delta {test_gain - val_gain:+.4f})", flush=True)
    if test_unseen:
        unseen_gain = test_unseen["macro_f1"] - test_unseen["macro_f1_argmax"]
        print(f"                                  test unseen-only {unseen_gain:+.4f}",
              flush=True)

    # What the reduced taxonomy cost on the held-out split, measured on the
    # rows it can no longer name. This has to be printed BEFORE the macro-F1 is
    # allowed anywhere near a comparison.
    if dropped_classes:
        cov_test = coverage_loss(inference, best_model, features,
                                 removed_rows.get("test", Split([], [])),
                                 "test", threshold_map)
        if cov_test:
            weak_exclusion.setdefault("coverage_loss", {})["test"] = cov_test
            total_test = cov_test["n_rows"] + test_all["n"]
            print(f"\n--- COVERAGE GIVEN UP by excluding {dropped_classes} ---",
                  flush=True)
            print(f"  {cov_test['n_rows']} TEST rows "
                  f"({cov_test['n_rows'] / max(total_test, 1):.1%} of the split) "
                  f"are real vulnerabilities the scanner can no longer name.",
                  flush=True)
            print(f"  Of those, {cov_test['still_flagged_as_some_vulnerability']} "
                  f"({cov_test['fraction_still_flagged']:.1%}) are still flagged "
                  f"as some vulnerability; the rest are called safe.", flush=True)

    base = _honest_baseline()
    if dropped_classes:
        print(f"\n=== NO BASELINE COMPARISON (reduced taxonomy) ===", flush=True)
        print(f"  {dropped_classes} were excluded, so this macro-F1 is an "
              f"average over {k} classes.\n  The committed baselines "
              f"({base['all']:.4f} all-rows, {base['unseen']:.4f} unseen-only) "
              f"are 9-class\n  numbers. Subtracting them would report a "
              f"denominator change as a modelling gain,\n  which is the single "
              f"most misleading thing this harness could print. Re-run without\n"
              f"  --drop-weak-classes for a comparable figure.", flush=True)
    else:
        print(f"\n=== COMPARISON vs committed v1 baseline (source: "
              f"{base['source']}) ===", flush=True)
        if test_unseen:
            headline = test_unseen["macro_f1"] - base["unseen"]
            print(f"  HEADLINE (fair, unseen-only vs unseen-only):", flush=True)
            print(f"    new {test_unseen['macro_f1']:.4f}  vs  baseline "
                  f"{base['unseen']:.4f}   ->  {headline:+.4f}", flush=True)
            print("    ^ the only apples-to-apples number: rows NEITHER model was "
                  "fitted on.", flush=True)
        print(f"  legacy/all-rows (NOT a fair bar - the baseline's 'all rows' "
              f"score is inflated by\n    memorized duplicates it was trained "
              f"on): new {test_all['macro_f1']:.4f} vs {base['all']:.4f}"
              f" -> {test_all['macro_f1'] - base['all']:+.4f}", flush=True)
        print(f"  (11-class legacy figure {BASELINE_11CLASS_MACRO_F1:.4f} is a "
              f"different taxonomy; not comparable.)", flush=True)

    # --- artifacts ---------------------------------------------------------
    if not hasattr(best_model, "predict_proba"):  # unreachable; cheap to assert
        print("error: winner has no predict_proba; refusing to save an artifact "
              "mlscan.scanner cannot use", flush=True)
        return 3

    pipeline = Pipeline([("features", features), ("clf", best_model)])
    # Carried on the artifact too, so an inference layer can apply the rule
    # without also loading thresholds_v2.json.
    pipeline.class_offsets_ = threshold_map
    pipeline.score_source_ = score_source(best_model)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nSaved model      -> {MODEL_PATH} ({size_mb:.1f} MB)", flush=True)

    THRESHOLDS_PATH.write_text(json.dumps({
        "rule": "predict = argmax_c(log(P[c]) - offset[c]); offsets are additive "
                "in log-probability space (equivalently multiplicative priors)",
        "applied_by": "mlscan.inference.predict_with_offsets",
        "score_source": score_source(best_model),
        "classes": all_classes,
        "offsets": threshold_map,
        "tuned_on": "validation",
        "search_grid": offset_info["grid"],
        "saturated_classes": offset_info["saturated_classes"],
        "transfer_estimate": offset_info["stability"],
        "excluded_classes": dropped_classes,
        **{key: offset_info[key] for key in
           ("val_macro_f1_argmax", "val_macro_f1_tuned")},
    }, indent=2), encoding="utf-8")
    print(f"Saved thresholds -> {THRESHOLDS_PATH}", flush=True)

    METRICS_PATH.write_text(json.dumps({
        **common,
        "final_run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_test": len(yte),
        "test_label_counts": _label_counts(yte),
        "test": test_all,
        "test_unseen_only": test_unseen,
        # Headline keys, hoisted so nothing has to dig for them.
        "test_macro_f1": test_all["macro_f1"],
        "test_macro_f1_argmax": test_all["macro_f1_argmax"],
        "test_macro_f1_unseen_only": test_unseen["macro_f1"] if test_unseen else None,
        "macro_f1_over_n_classes": k,
        "offset_transfer": {
            "val_gain": round(val_gain, 4),
            "test_gain": round(test_gain, 4),
            "delta": round(test_gain - val_gain, 4),
            "split_half_expected_gain": (
                offset_info["stability"]["held_out_gain_mean"]
                if offset_info.get("stability") else None),
            "note": "val_gain is in-sample (the offsets were fitted on that "
                    "split). split_half_expected_gain is the honest expected "
                    "effect size, estimated on validation alone; report that, "
                    "not val_gain, as 'what the calibration buys'.",
        },
        "baselines": ({
            "comparable": False,
            "excluded_classes": dropped_classes,
            "committed_11class_legacy": BASELINE_11CLASS_MACRO_F1,
            "committed_rescored_9class_all_rows": _honest_baseline()["all"],
            "committed_rescored_9class_unseen_only": _honest_baseline()["unseen"],
            "HEADLINE_delta_unseen_vs_unseen": None,
            "legacy_delta_all_rows": None,
            "note": f"NO DELTA IS REPORTED. {dropped_classes} were excluded, so "
                    f"this run's macro-F1 averages {k} classes while every "
                    f"committed baseline averages 9. Subtracting them would "
                    f"present a denominator change as a modelling gain — "
                    f"measured on the v2 artifact, excluding CWE-200 alone moves "
                    f"macro-F1 +0.0513 while no surviving class gains more than "
                    f"+0.0135. Quote weak_class_exclusion.coverage_loss "
                    f"alongside any figure from this run.",
        } if dropped_classes else {
            "comparable": True,
            "committed_11class_legacy": BASELINE_11CLASS_MACRO_F1,
            "committed_rescored_9class_all_rows": _honest_baseline()["all"],
            "committed_rescored_9class_unseen_only": _honest_baseline()["unseen"],
            "HEADLINE_delta_unseen_vs_unseen": (
                round(test_unseen["macro_f1"] - _honest_baseline()["unseen"], 4)
                if test_unseen else None),
            "legacy_delta_all_rows": round(
                test_all["macro_f1"] - _honest_baseline()["all"], 4),
            "note": "The ONLY fair comparison is unseen-only vs unseen-only "
                    "(HEADLINE_delta_unseen_vs_unseen): the v1 baseline was "
                    "trained on a split containing 9.1% of the test rows "
                    "verbatim, so its all-rows score carries a ~0.207 "
                    "memorization premium this de-duplicated model cannot and "
                    "should not earn. legacy_delta_all_rows is kept only for "
                    "continuity with previously published figures.",
        }),
        "total_secs": round(time.time() - t_start, 1),
    }, indent=2), encoding="utf-8")
    print(f"Saved metrics    -> {METRICS_PATH}", flush=True)
    _print_reduced_taxonomy_reminder(dropped_classes, k)
    print(f"Done in {time.time() - t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
