"""Measured head-to-head benchmark of the three detectors on held-out data.

    python -m mlscan.benchmark --sample 400      # fast smoke run
    python -m mlscan.benchmark                   # full TEST split -> model/benchmark.json

Why this module exists
----------------------
The project's headline claim for the hybrid scanner used to be "7/7
vulnerabilities caught, 0 false positives". That number came from seven
hand-picked snippets, so it measured nothing: the rule engine had never been run
against a corpus, and its real precision/recall were unknown. A benchmark you
choose the inputs for is an anecdote. This module replaces it with a
reproducible measurement on data neither detector was fitted on.

Three detectors, scored on identical rows
-----------------------------------------
* ``ml``     - :func:`mlscan.scanner.scan` with ``use_rules=False``
* ``rules``  - :func:`mlscan.rules.scan_rules`
* ``hybrid`` - :func:`mlscan.scanner.scan` (both layers)

Each is reduced to the same primitive: **the set of CWE ids it names for a
row**. "Flagged as vulnerable" is then exactly "that set is non-empty", which is
the definition :func:`mlscan.scanner.scan` already uses for ``is_vulnerable``,
so the binary metric here and the shipped scanner's behaviour cannot diverge.

The ML pass is batched through :mod:`mlscan.inference` rather than looping over
``scan()``: on full-length (4000-char) rows, vectorizing one at a time measured
61 ms/row against 37 ms/row batched — a ~7-minute difference over the full test
split. Batching is an optimisation, not a second decision rule, so
:func:`verify_against_scanner` re-scores the first rows through the real
``scanner.scan`` and aborts on any disagreement. Same discipline as
``mlscan.tune`` step 5.

The code_fixed precision set
----------------------------
The corpus carries a ``code_fixed`` column: the patched version of a vulnerable
sample. Firing on the patch is *mostly* a false positive, which gives a negative
pool that needs no hand labelling — and it is the only negative pool that covers
the non-C languages the rule engine targets at all (``source=cybernative_dpo``
contributes zero "safe" rows).

**Treat the resulting rate as an upper bound on the false-positive rate, not as
gospel.** Three reasons it overstates:

1. a patch fixes *one* defect; the function may still contain another;
2. many "fixes" in this corpus are LLM-authored and keep the dangerous
   construct (an ``eval`` fix that replaces ``eval(user_input)`` with
   ``eval(compile(validated_tree))`` still contains ``eval``);
3. a rule may legitimately fire on a *different* CWE than the one patched.

So the report pairs the raw fire rate with two sharper numbers:
``fires_only_on_fixed`` (silent on the vulnerable side, fires on the patch —
nothing was removed, so this is an unambiguous false positive) and
``patch_sensitivity`` (of the rows where the detector named the true CWE, the
fraction where that CWE goes silent once the defect is patched).

Reporting caveats that are baked into the output
------------------------------------------------
* Metrics are reported for **all rows** and for **unseen-only** rows
  (``dup_of_train == False``) — the same split ``mlscan.data`` documents. Only
  the unseen-only number describes generalization.
* ``fire_rate_on_safe`` is broken out **per source**, because a chunk of the
  corpus's "safe" labels are wrong: ``eval(user_input)`` and
  ``subprocess.run(user_input, shell=True)`` are both labelled safe in
  ``source=labeled_dataset``. Pooling those into one precision denominator
  measures label noise, not the detector.
* Metrics are broken out for the C family versus everything else, because the
  ML model trains on a ~92% C corpus while the rule engine's coverage is almost
  entirely non-C. A single pooled number hides that they cover disjoint
  languages.

Cost: measured 42 ms/row for all three detectors on this 4-CPU box, so the full
17,121-row test split is ~13 minutes (a ``--sample 400`` smoke run is ~35 s
including the dataset load). Use ``--sample N`` while iterating; sampled runs
write ``model/benchmark_smoke.json`` so they can never overwrite the headline
artifact.

Multiple-testing note: iterate rule changes with ``--split validation`` and look
at ``--split test`` once, mirroring the ``mlscan.tune --final`` discipline.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mlscan.labels import SAFE, TAXONOMY

MODEL_DIR = Path(__file__).resolve().parent / "model"

# Full runs own the headline artifact; sampled runs are quarantined into their
# own file so a smoke test can never be mistaken for (or overwrite) the number
# the write-up quotes.
BENCHMARK_PATH = MODEL_DIR / "benchmark.json"
SMOKE_PATH = MODEL_DIR / "benchmark_smoke.json"

DETECTORS = ("ml", "rules", "hybrid")

# Rows per predict_proba call. Big enough to amortise the vectorizer, small
# enough that the intermediate sparse matrix stays well inside the RAM budget.
BATCH_ROWS = 512

# How many rows to re-score through the real ``scanner.scan`` as a cross-check
# on the batched path. 25 rows is ~3 s and has caught every drift in practice.
VERIFY_ROWS = 25

SEED = 42

# Languages the ML model was overwhelmingly trained on. Grouped separately
# because rule coverage and model coverage are near-disjoint across this line.
C_FAMILY = frozenset({"c", "cpp", "c++", "cc", "cxx", "h", "hpp"})

CAVEATS = [
    "code_fixed fire rates are an UPPER BOUND on the false-positive rate: a "
    "patch fixes one defect, the function may contain another, and many "
    "'fixes' in this corpus are LLM-authored and keep the dangerous construct.",
    "fire_rate_on_safe pooled across sources measures label noise as much as "
    "detector precision - source=labeled_dataset labels eval(user_input) and "
    "subprocess.run(user_input, shell=True) as safe. Read by_source.",
    "Per-CWE metrics are one-vs-rest over the SET of CWEs a detector names, so "
    "a detector gets credit when the right CWE is anywhere in its output. This "
    "is generous by construction; at the shipped 0.50 threshold the ML set "
    "holds at most one class, so it only loosens the rule and hybrid columns.",
    "Only the unseen_only slice measures generalization; all_rows includes "
    "eval rows byte-identical to a training row (see mlscan.data).",
    "The rule engine's confidences are hardcoded constants, not calibrated "
    "probabilities, and are deliberately not reported here.",
]


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

@dataclass
class Row:
    """One evaluation row: the code, its gold label, and its patched twin."""

    code: str
    label: str                 # SAFE or a folded taxonomy class
    code_fixed: str = ""       # patched version, "" when the corpus has none
    language: str = ""
    source: str = ""
    dup_of_train: bool = False

    @property
    def is_vulnerable(self) -> bool:
        return self.label != SAFE

    @property
    def has_patch(self) -> bool:
        """A usable negative sample: a patch that actually changed something."""
        return (self.is_vulnerable and bool(self.code_fixed.strip())
                and self.code_fixed != self.code)


def language_group(language: str) -> str:
    """``"c_family"`` or ``"non_c"`` — the line the two detectors split on."""
    return "c_family" if str(language).strip().lower() in C_FAMILY else "non_c"


def stratified_sample(rows: list[Row], n: int, seed: int = SEED) -> list[Row]:
    """Take ~``n`` rows, proportionally per class, keeping every class present.

    Proportional (not balanced) so the sampled slice keeps the corpus's ~91%
    "safe" prior — a benchmark run on a re-balanced sample would report a
    precision that no real repository would ever see.
    """
    if n <= 0 or n >= len(rows):
        return rows
    rng = random.Random(seed)
    by_label: dict[str, list[Row]] = {}
    for row in rows:
        by_label.setdefault(row.label, []).append(row)
    kept: list[Row] = []
    for label in sorted(by_label):
        group = by_label[label]
        take = min(len(group), max(1, round(len(group) * n / len(rows))))
        kept.extend(rng.sample(group, take))
    rng.shuffle(kept)
    return kept


# ---------------------------------------------------------------------------
# metrics (pure stdlib - importable and testable without the ML extras)
# ---------------------------------------------------------------------------

def _ratio(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict:
    """Precision / recall / F1, with the 0/0 cases pinned to 0.0."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


def binary_metrics(golds, detections) -> dict:
    """"Is this code vulnerable?" scored as a binary decision.

    A detector answers yes exactly when it names at least one CWE, which is how
    ``scanner.scan`` sets ``is_vulnerable``.
    """
    tp = fp = fn = tn = 0
    for gold, named in zip(golds, detections):
        flagged = bool(named)
        vulnerable = gold != SAFE
        if vulnerable and flagged:
            tp += 1
        elif vulnerable:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    return {
        "n": n,
        "n_vulnerable": tp + fn,
        "n_safe": fp + tn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        **_prf(tp, fp, fn),
        "accuracy": _ratio(tp + tn, n),
        "flag_rate": _ratio(tp + fp, n),
        # The number a reviewer feels: how often the detector fires on a row the
        # corpus calls safe. Read it per source, not pooled (see CAVEATS).
        "fire_rate_on_safe": _ratio(fp, fp + tn),
    }


def per_cwe_metrics(golds, detections) -> dict:
    """One-vs-rest precision/recall per CWE, over the named-CWE set.

    A row counts as a positive for class C when the detector names C anywhere in
    its output — not only when C is its top choice. Detectors that emit several
    findings (rules, hybrid) are otherwise unscoreable against a single-label
    gold column.

    ``macro_f1`` averages only over classes with non-zero support, so a detector
    is not punished for a class the sampled slice does not contain.
    """
    classes = sorted(set(TAXONOMY) | {c for named in detections for c in named})
    out: dict[str, dict] = {}
    for cls in classes:
        tp = fp = fn = 0
        for gold, named in zip(golds, detections):
            hit = cls in named
            if gold == cls:
                tp += hit
                fn += not hit
            elif hit:
                fp += 1
        out[cls] = {"support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                    **_prf(tp, fp, fn)}
    scored = [m["f1"] for m in out.values() if m["support"]]
    return {
        "classes": out,
        "macro_f1": round(sum(scored) / len(scored), 4) if scored else 0.0,
        "macro_over": [c for c, m in out.items() if m["support"]],
    }


def by_group(golds, detections, keys) -> dict[str, dict]:
    """``binary_metrics`` split by an arbitrary per-row key (source, language)."""
    buckets: dict[str, tuple[list, list]] = {}
    for gold, named, key in zip(golds, detections, keys):
        g, d = buckets.setdefault(str(key), ([], []))
        g.append(gold)
        d.append(named)
    return {k: binary_metrics(*v) for k, v in sorted(buckets.items())}


def patched_metrics(golds, on_vuln, on_fixed) -> dict:
    """False-positive evidence from the ``code_fixed`` column.

    ``fire_rate_on_fixed`` is the headline upper bound. The two numbers under it
    are the ones that survive scrutiny:

    * ``fires_only_on_fixed`` — silent on the vulnerable side, fires on the
      patch. Nothing was removed, so this is an unambiguous false positive.
    * ``patch_sensitivity`` — of the rows where the detector named the true CWE,
      the fraction where that CWE disappears once the defect is patched. A
      detector that keeps firing after the fix is matching a construct, not a
      defect.
    """
    n = fired_fixed = only_fixed = named_gold = went_silent = 0
    for gold, vuln_named, fixed_named in zip(golds, on_vuln, on_fixed):
        n += 1
        fired_fixed += bool(fixed_named)
        only_fixed += bool(fixed_named) and not vuln_named
        if gold in vuln_named:
            named_gold += 1
            went_silent += gold not in fixed_named
    return {
        "n_pairs": n,
        "fired_on_fixed": fired_fixed,
        "fire_rate_on_fixed": _ratio(fired_fixed, n),
        "fires_only_on_fixed": only_fixed,
        "fires_only_on_fixed_rate": _ratio(only_fixed, n),
        "named_true_cwe_on_vulnerable": named_gold,
        "patch_sensitive": went_silent,
        "patch_sensitivity": _ratio(went_silent, named_gold),
        "note": "upper bound on FP rate - see module docstring",
    }


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------

def _to_cwe(label: str) -> str:
    """Fold a model class onto the id ``scanner.scan`` reports for it."""
    from mlscan.labels import describe

    try:
        return describe(label)["cwe"] or label
    except KeyError:  # a legacy artifact class outside the taxonomy
        return label


def rule_detect(code: str) -> frozenset[str]:
    """CWE ids named by the deterministic rule engine (pure stdlib path)."""
    from mlscan.rules import scan_rules

    return frozenset(f["cwe"] for f in scan_rules(code))


def ml_detect(codes, threshold: float, progress=None) -> list[frozenset[str]]:
    """CWE ids the ML classifier scores at/above ``threshold``, batched.

    Uses the artifact :mod:`mlscan.scanner` actually serves (v2 when servable,
    else the v1 fallback), so the benchmark cannot describe a model the user
    never gets.
    """
    from mlscan import inference
    from mlscan.scanner import _artifact

    pipeline, offsets, _ = _artifact()
    out: list[frozenset[str]] = []
    for start in range(0, len(codes), BATCH_ROWS):
        chunk = codes[start:start + BATCH_ROWS]
        classes, proba = inference.class_scores(pipeline, chunk)
        adjusted = inference.apply_offsets(classes, proba, offsets)
        cwes = [_to_cwe(c) for c in classes]
        for row in adjusted:
            out.append(frozenset(
                cwe for cwe, p, cls in zip(cwes, row, classes)
                if cls != SAFE and p >= threshold))
        if progress:
            progress(min(start + BATCH_ROWS, len(codes)), len(codes))
    return out


def verify_against_scanner(codes, ml_sets, hybrid_sets, threshold: float,
                           n: int = VERIFY_ROWS) -> int:
    """Re-score a prefix through the real ``scanner.scan`` and assert agreement.

    The batched ML path and the set-union hybrid are optimisations of what
    ``scanner.scan`` does per row. If they ever stop being *exactly* that, every
    number in this report describes a detector nobody ships — so this fails
    loudly rather than reporting a plausible-looking table.
    """
    from mlscan.scanner import scan

    checked = 0
    for i, code in enumerate(codes[:n]):
        if ml_sets is not None:
            got = frozenset(f["cwe"] for f in
                            scan(code, threshold, use_rules=False)["findings"])
            if got != ml_sets[i]:
                raise RuntimeError(
                    f"batched ML detection disagrees with scanner.scan on row "
                    f"{i}: scanner={sorted(got)} batched={sorted(ml_sets[i])}")
        if hybrid_sets is not None:
            got = frozenset(f["cwe"] for f in scan(code, threshold)["findings"])
            if got != hybrid_sets[i]:
                raise RuntimeError(
                    f"derived hybrid detection disagrees with scanner.scan on "
                    f"row {i}: scanner={sorted(got)} "
                    f"derived={sorted(hybrid_sets[i])}")
        checked += 1
    return checked


def run_detectors(codes, detectors, threshold: float,
                  progress=None) -> dict[str, list[frozenset[str]]]:
    """Named-CWE sets per detector, computed with one pass of each layer.

    ``hybrid`` is the union of the other two, which is precisely what
    ``scanner.scan`` produces: rule findings are merged into the ML findings
    keyed on the CWE id, and rule findings are never threshold-filtered. Derived
    rather than re-run so the expensive ML pass happens once;
    :func:`verify_against_scanner` proves the identity on real rows.
    """
    need_ml = bool({"ml", "hybrid"} & set(detectors))
    need_rules = bool({"rules", "hybrid"} & set(detectors))

    out: dict[str, list[frozenset[str]]] = {}
    if need_ml:
        out["ml"] = ml_detect(codes, threshold, progress=progress)
    if need_rules:
        rules: list[frozenset[str]] = []
        for i, code in enumerate(codes, 1):
            rules.append(rule_detect(code))
            if progress and i % BATCH_ROWS == 0:
                progress(i, len(codes))
        out["rules"] = rules
    if "hybrid" in detectors:
        out["hybrid"] = [m | r for m, r in zip(out["ml"], out["rules"])]
    return {d: out[d] for d in detectors}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def _select(seq, idx):
    return [seq[i] for i in idx]


def _headline(result: dict) -> dict:
    """The numbers a write-up should quote for one detector, in one place."""
    fixed = result["code_fixed"]
    return {
        "binary_precision": result["all_rows"]["binary"]["precision"],
        "binary_recall": result["all_rows"]["binary"]["recall"],
        "binary_f1": result["all_rows"]["binary"]["f1"],
        "binary_f1_unseen_only": result["unseen_only"]["binary"]["f1"],
        "cwe_macro_f1": result["all_rows"]["per_cwe"]["macro_f1"],
        "fire_rate_on_safe_rows": result["all_rows"]["binary"]["fire_rate_on_safe"],
        "fire_rate_on_patched_code_UPPER_BOUND": (
            fixed["fire_rate_on_fixed"] if fixed else None),
        "patch_sensitivity": fixed["patch_sensitivity"] if fixed else None,
    }


def benchmark_rows(rows: list[Row], detectors=DETECTORS, threshold: float = 0.50,
                   verify: int = VERIFY_ROWS, quiet: bool = False) -> dict:
    """Score every detector on ``rows`` and return the full result payload.

    Pure with respect to the dataset: the caller supplies :class:`Row` objects,
    so this is testable on synthetic input with no HuggingFace download and no
    model artifact (as long as ``detectors`` stays inside what is installed).
    """
    started = time.time()
    codes = [r.code for r in rows]
    golds = [r.label for r in rows]
    sources = [r.source for r in rows]
    languages = [r.language for r in rows]

    def progress(done, total):
        if not quiet:
            print(f"    {done}/{total}", end="\r", flush=True)

    if not quiet:
        print(f"  scanning {len(rows)} rows with {', '.join(detectors)} ...",
              flush=True)
    named = run_detectors(codes, detectors, threshold, progress=progress)

    verified = 0
    if verify and "ml" in named:
        verified = verify_against_scanner(
            codes, named.get("ml"), named.get("hybrid"), threshold, verify)
        if not quiet:
            print(f"  verified {verified} rows against scanner.scan", flush=True)

    # The patched (code_fixed) negative pool.
    patch_idx = [i for i, r in enumerate(rows) if r.has_patch]
    patched_named: dict[str, list[frozenset[str]]] = {}
    if patch_idx:
        if not quiet:
            print(f"  scanning {len(patch_idx)} code_fixed patches ...",
                  flush=True)
        patched_named = run_detectors(
            [rows[i].code_fixed for i in patch_idx], detectors, threshold,
            progress=progress)

    unseen_idx = [i for i, r in enumerate(rows) if not r.dup_of_train]

    results: dict[str, dict] = {}
    for det in detectors:
        sets = named[det]
        results[det] = {
            "all_rows": {
                "binary": binary_metrics(golds, sets),
                "per_cwe": per_cwe_metrics(golds, sets),
                "by_source": by_group(golds, sets, sources),
                "by_language": by_group(golds, sets, languages),
                "by_language_group": by_group(
                    golds, sets, [language_group(x) for x in languages]),
            },
            "unseen_only": {
                "binary": binary_metrics(_select(golds, unseen_idx),
                                         _select(sets, unseen_idx)),
                "per_cwe": per_cwe_metrics(_select(golds, unseen_idx),
                                           _select(sets, unseen_idx)),
            },
            "code_fixed": patched_metrics(
                _select(golds, patch_idx), _select(sets, patch_idx),
                patched_named[det]) if patch_idx else None,
        }

    return {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Hoisted so a report can quote the benchmark without digging, and so
        # nobody is tempted to go back to hand-picked snippets for a headline.
        "headline": {det: _headline(res) for det, res in results.items()},
        "threshold": threshold,
        "n_rows": len(rows),
        "n_unseen": len(unseen_idx),
        "n_code_fixed_pairs": len(patch_idx),
        "label_counts": {k: golds.count(k) for k in sorted(set(golds))},
        "rows_verified_against_scanner": verified,
        "detectors": results,
        "caveats": CAVEATS,
        "elapsed_secs": round(time.time() - started, 1),
    }


# ---------------------------------------------------------------------------
# dataset loading
# ---------------------------------------------------------------------------

def load_rows(split: str = "test", offline: bool = True) -> list[Row]:
    """Load one evaluation split as :class:`Row` objects, with ``code_fixed``.

    ``mlscan.data.load_splits`` deliberately returns only codes/labels, so this
    rebuilds the same frame through the *same* private helper
    (``_labelled_frame``) and the same ``dup_of_train`` derivation as
    ``load_splits`` — identical taxonomy folding, identical truncation,
    identical row order — while keeping the ``code_fixed``, ``language`` and
    ``source`` columns the benchmark needs. Evaluation splits are never
    filtered or re-balanced here, exactly as ``load_splits`` documents.
    """
    import os

    if offline:  # load_dataset otherwise stalls on a network check
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    from datasets import load_dataset

    from mlscan.data import DATASET_ID, MAX_CODE_CHARS, _labelled_frame

    ds = load_dataset(DATASET_ID)
    # Hashes of the ORIGINAL, pre-dedup train split: the same reference
    # load_splits flags eval rows against.
    train_hashes = set(_labelled_frame(ds["train"].to_pandas())["code_hash"])

    frame = _labelled_frame(ds[split].to_pandas())
    n = len(frame)

    def column(name, truncate=False):
        if name not in frame.columns:
            return [""] * n
        col = frame[name].fillna("").astype(str)
        return (col.str.slice(0, MAX_CODE_CHARS) if truncate else col).tolist()

    # code_fixed is truncated to the same budget as code, so both sides of a
    # patch pair are judged on the same amount of text.
    return [
        Row(code=code, label=label, code_fixed=patch, language=lang,
            source=src, dup_of_train=h in train_hashes)
        for code, label, patch, lang, src, h in zip(
            frame["code"], frame["label"], column("code_fixed", truncate=True),
            column("language"), column("source"), frame["code_hash"])
    ]


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _table(header: list[str], rows: list[list]) -> str:
    widths = [max(len(str(h)), *(len(_fmt(r[i])) for r in rows)) if rows
              else len(str(h)) for i, h in enumerate(header)]
    lines = ["  ".join(str(h).ljust(w) for h, w in zip(header, widths)).rstrip(),
             "  ".join("-" * w for w in widths)]
    for row in rows:
        lines.append("  ".join(_fmt(c).ljust(w)
                               for c, w in zip(row, widths)).rstrip())
    return "\n".join(lines)


def format_report(payload: dict, split: str = "test") -> str:
    """The printed comparison table. Pure string building, no I/O."""
    dets = payload["detectors"]
    out = [
        f"=== mlscan detector benchmark - {split.upper()} split ===",
        f"rows {payload['n_rows']}  (unseen {payload['n_unseen']}, "
        f"code_fixed pairs {payload['n_code_fixed_pairs']})  "
        f"threshold {payload['threshold']}",
        "",
        "Binary decision: is this code vulnerable?",
    ]
    for slice_name in ("all_rows", "unseen_only"):
        rows = []
        for det, res in dets.items():
            b = res[slice_name]["binary"]
            rows.append([det, b["n"], b["tp"], b["fp"], b["fn"],
                         b["precision"], b["recall"], b["f1"],
                         b["fire_rate_on_safe"],
                         res[slice_name]["per_cwe"]["macro_f1"]])
        out += ["", f"  [{slice_name}]",
                _table(["detector", "n", "TP", "FP", "FN", "prec", "recall",
                        "F1", "FPR_safe", "cwe_macroF1"], rows)]

    if payload["n_code_fixed_pairs"]:
        rows = []
        for det, res in dets.items():
            p = res["code_fixed"]
            rows.append([det, p["n_pairs"], p["fired_on_fixed"],
                         p["fire_rate_on_fixed"], p["fires_only_on_fixed"],
                         p["named_true_cwe_on_vulnerable"],
                         p["patch_sensitivity"]])
        out += ["", "False positives on patched code (code_fixed) - UPPER BOUND",
                _table(["detector", "pairs", "fired", "fire_rate",
                        "only_on_fix", "named_cwe", "patch_sens"], rows)]

    out += ["", "Per-CWE F1 (named-CWE set, all rows)"]
    # Support depends only on the gold column, so any detector's copy will do.
    first = next(iter(dets.values()))["all_rows"]["per_cwe"]["classes"]
    rows = []
    for cls in sorted(first):
        rows.append([cls, first[cls]["support"]] +
                    [dets[d]["all_rows"]["per_cwe"]["classes"][cls]["f1"]
                     for d in dets])
    out.append(_table(["cwe", "support", *dets], rows))

    out += ["", "Binary precision / recall by language group (all rows)"]
    groups = sorted({g for res in dets.values()
                     for g in res["all_rows"]["by_language_group"]})
    rows = []
    for g in groups:
        row = [g]
        for det in dets:
            m = dets[det]["all_rows"]["by_language_group"].get(g)
            row += [m["n"] if m else 0, m["precision"] if m else 0.0,
                    m["recall"] if m else 0.0]
        rows.append(row)
    out.append(_table(["group", *[f"{d}_{c}" for d in dets
                                  for c in ("n", "prec", "rec")]], rows))

    out += ["", "Caveats:"] + [f"  - {c}" for c in payload["caveats"]]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m mlscan.benchmark",
        description="Measure ML / rules / hybrid on held-out data and on the "
                    "code_fixed patches. Replaces the hand-picked 7/7 anecdote.")
    p.add_argument("--split", choices=("test", "validation"), default="test",
                   help="split to score (default: test). Iterate rule changes "
                        "on validation and look at test once.")
    p.add_argument("--sample", type=int, default=0,
                   help="score ~N stratified rows for a fast smoke run; writes "
                        "benchmark_smoke.json instead of benchmark.json")
    p.add_argument("--threshold", type=float, default=None,
                   help="ML confidence threshold (default: "
                        "mlscan.scanner.DEFAULT_THRESHOLD)")
    p.add_argument("--no-ml", action="store_true",
                   help="rules only - no scikit-learn, no model artifact needed")
    p.add_argument("--seed", type=int, default=SEED, help="sampling seed")
    p.add_argument("--verify", type=int, default=VERIFY_ROWS,
                   help="rows to re-score through scanner.scan as a cross-check "
                        "on the batched ML path (0 disables)")
    p.add_argument("--out", type=Path, default=None,
                   help="output JSON path (default: model/benchmark.json, or "
                        "model/benchmark_smoke.json for a --sample run)")
    p.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True,
                   help="force HuggingFace offline mode (dataset is cached)")
    return p.parse_args(argv)


def _dataset_id() -> str:
    from mlscan.data import DATASET_ID

    return DATASET_ID


def _model_info() -> dict:
    """Which artifact was actually benchmarked (v2, or the v1 fallback)."""
    from mlscan.scanner import model_info

    info = model_info()
    # Filename only: the JSON is committed, and an absolute path from whoever
    # happened to run it is noise.
    return {"model": Path(info["model_path"]).name,
            "estimator": info["estimator"], "rule": info["rule"]}


def main(argv=None) -> int:
    args = _parse_args(argv)
    MODEL_DIR.mkdir(exist_ok=True)

    threshold = args.threshold
    detectors = ("rules",) if args.no_ml else DETECTORS
    if threshold is None:
        if args.no_ml:
            threshold = 0.50
        else:
            from mlscan.scanner import DEFAULT_THRESHOLD
            threshold = DEFAULT_THRESHOLD

    started = time.time()
    print(f"Loading {args.split} split ...", flush=True)
    rows = load_rows(args.split, offline=args.offline)
    print(f"  {len(rows)} rows "
          f"({sum(r.is_vulnerable for r in rows)} vulnerable, "
          f"{sum(r.has_patch for r in rows)} with a usable code_fixed)",
          flush=True)
    if args.sample:
        rows = stratified_sample(rows, args.sample, args.seed)
        print(f"  --sample {args.sample}: reduced to {len(rows)} rows "
              f"(stratified, prior preserved)", flush=True)

    payload = benchmark_rows(rows, detectors=detectors, threshold=threshold,
                             verify=args.verify)
    payload["split"] = args.split
    payload["sample"] = args.sample or None
    payload["seed"] = args.seed
    payload["dataset"] = _dataset_id()
    payload["model"] = _model_info() if not args.no_ml else None

    out_path = args.out or (SMOKE_PATH if args.sample else BENCHMARK_PATH)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(format_report(payload, args.split))
    print(f"\nSaved -> {out_path}")
    print(f"Done in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
