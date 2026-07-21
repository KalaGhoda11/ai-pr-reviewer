"""Re-score the committed v1 artifact to produce the HONEST baseline.

Run:  .venv/Scripts/python.exe scripts/rescore_baseline.py

Why this exists
---------------
``mlscan/model/vuln_clf.joblib`` is the currently committed model. It was trained
under the OLD taxonomy (11 classes: safe + 10 CWEs, with CWE-119 / CWE-787 /
CWE-125 kept apart) and its published metrics were computed on the OLD, leaky
splits. Neither number is a legitimate bar for the new 9-class model, because:

* the taxonomies differ, so the macro-F1 averages are over different class sets;
* 9.1% of the test rows are byte-identical (on ``code[:MAX_CODE_CHARS]`` — the
  exact string the model sees) to a training row, and the duplication is
  concentrated in exactly the rare classes that dominate a macro average.

This script fixes both by *inference only* — it does not retrain anything:

1. loads the committed 11-class pipeline as-is;
2. predicts over the TEST split returned by the CURRENT ``mlscan.data.load_splits()``
   (9-class labels; de-duplication touches TRAIN only, so TEST is the untouched
   held-out sample and carries ``dup_of_train`` flags);
3. folds each 11-class prediction into the 9-class taxonomy through
   ``labels.CWE_MERGE_MAP`` (CWE-119 / CWE-787 / CWE-125 -> MEMORY-OOB). A
   predicted class that is not in the 9-class taxonomy after folding is mapped
   to "safe" and counted explicitly in the report, so the substitution can never
   hide inside the aggregate;
4. reports macro-F1 over ALL test rows and over the UNSEEN-ONLY subset
   (``dup_of_train == False``).

The unseen-only number is the honest bar. The all-rows number is kept beside it
so the size of the memorization premium is visible.

Nothing here is tuned, thresholded, or offset: this is plain ``predict()``
(argmax), which is exactly what ``Pipeline.predict`` does for the v1 artifact in
production. The baseline therefore describes a decision rule that real code runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

# Benign: the v1 LightGBM was fit on a named sparse matrix and predicts on an
# unnamed one. Same suppression mlscan.scanner applies for the same reason.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlscan.labels import CLASSES, CWE_MERGE_MAP, SAFE, TAXONOMY  # noqa: E402

MODEL_PATH = REPO_ROOT / "mlscan" / "model" / "vuln_clf.joblib"
OUT_PATH = REPO_ROOT / "mlscan" / "model" / "baseline_honest.json"

# Macro-F1 is averaged over this fixed label set, not over "whatever appeared in
# y_true | y_pred". Pinning it means the baseline and the tuned model average
# over identical classes even if one of them never predicts some rare class.
MACRO_LABELS = sorted(CLASSES)


def fold_prediction(pred: str) -> tuple[str, bool]:
    """Fold one v1 (11-class) prediction into the 9-class taxonomy.

    Returns ``(folded_label, was_out_of_taxonomy)``. CWE-119 / CWE-787 / CWE-125
    become MEMORY-OOB. A label that is still outside the taxonomy afterwards —
    a class the v1 model can emit but the new taxonomy no longer has — is mapped
    to "safe" (the conservative choice: it becomes a missed detection, never a
    free correct answer) and flagged so the caller can report the count.
    """
    label = str(pred)
    folded = CWE_MERGE_MAP.get(label, label)
    if folded == SAFE or folded in TAXONOMY:
        return folded, False
    return SAFE, True


def _scores(y_true, y_pred):
    """macro-F1 + per-class classification report over the pinned label set."""
    from sklearn.metrics import classification_report, f1_score

    return {
        "macro_f1": round(
            float(f1_score(y_true, y_pred, average="macro",
                           labels=MACRO_LABELS, zero_division=0)), 4),
        "report": classification_report(
            y_true, y_pred, labels=MACRO_LABELS, digits=3,
            zero_division=0, output_dict=True),
        "report_text": classification_report(
            y_true, y_pred, labels=MACRO_LABELS, digits=3, zero_division=0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke mode: score only the first N test rows. Does "
                         "NOT write the baseline JSON unless --out is given.")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"where to write the JSON (default {OUT_PATH})")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        print(f"ERROR: no committed artifact at {MODEL_PATH}", file=sys.stderr)
        return 1

    print("=" * 78)
    print("HONEST BASELINE -- committed v1 artifact re-scored on the 9-class test split")
    print("=" * 78)

    # ---- data ---------------------------------------------------------------
    from mlscan.data import load_splits

    t0 = time.time()
    print(f"\n[1/4] Loading splits via mlscan.data.load_splits() ...", flush=True)
    splits = load_splits()
    test = splits["test"]
    del splits  # release train/validation before the model is loaded
    if test.dup_of_train is None:
        print("ERROR: test split carries no dup_of_train flags; cannot compute "
              "the unseen-only baseline.", file=sys.stderr)
        return 1

    codes, y_true, is_dup = test.codes, test.labels, test.dup_of_train
    if args.limit:
        codes, y_true, is_dup = (codes[:args.limit], y_true[:args.limit],
                                 is_dup[:args.limit])
        print(f"      !! SMOKE MODE: limited to {len(codes)} test rows")
    n_test = len(codes)
    n_unseen = sum(1 for d in is_dup if not d)
    print(f"      test rows: {n_test}   unseen (dup_of_train == False): {n_unseen}"
          f"   duplicated: {n_test - n_unseen} "
          f"({100.0 * (n_test - n_unseen) / max(n_test, 1):.1f}%)"
          f"   [{time.time() - t0:.1f}s]", flush=True)

    # ---- model --------------------------------------------------------------
    import joblib

    t0 = time.time()
    print(f"\n[2/4] Loading committed artifact {MODEL_PATH.name} ...", flush=True)
    pipeline = joblib.load(MODEL_PATH)
    clf = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
    v1_classes = [str(c) for c in clf.classes_]
    print(f"      {type(clf).__name__} with {len(v1_classes)} classes: "
          f"{', '.join(sorted(v1_classes))}   [{time.time() - t0:.1f}s]", flush=True)

    # ---- predict ------------------------------------------------------------
    t0 = time.time()
    print(f"\n[3/4] Predicting over {n_test} test rows (argmax, no offsets) ...",
          flush=True)
    raw_pred = [str(p) for p in pipeline.predict(codes)]
    print(f"      done in {time.time() - t0:.1f}s", flush=True)

    # ---- fold + score -------------------------------------------------------
    print("\n[4/4] Folding 11-class predictions into the 9-class taxonomy ...")
    folded, out_of_taxonomy = [], Counter()
    for p in raw_pred:
        lab, dropped = fold_prediction(p)
        folded.append(lab)
        if dropped:
            out_of_taxonomy[p] += 1

    merged = {c: n for c, n in Counter(raw_pred).items() if c in CWE_MERGE_MAP}
    if merged:
        print("      merged into MEMORY-OOB: "
              + ", ".join(f"{c}={n}" for c, n in sorted(merged.items()))
              + f"  (total {sum(merged.values())})")
    else:
        print("      merged into MEMORY-OOB: none predicted")
    if out_of_taxonomy:
        total = sum(out_of_taxonomy.values())
        print(f"      !! {total} prediction(s) fell OUTSIDE the 9-class taxonomy "
              f"and were mapped to '{SAFE}': "
              + ", ".join(f"{c}={n}" for c, n in sorted(out_of_taxonomy.items())))
    else:
        print("      out-of-taxonomy predictions mapped to 'safe': 0 "
              "(every v1 class folds into the taxonomy)")

    all_rows = _scores(y_true, folded)
    unseen_true = [t for t, d in zip(y_true, is_dup) if not d]
    unseen_pred = [p for p, d in zip(folded, is_dup) if not d]
    unseen = _scores(unseen_true, unseen_pred)

    # ---- report -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ALL TEST ROWS  (n = %d)" % n_test)
    print("=" * 78)
    print(all_rows["report_text"])
    print(f"baseline_9class_macro_f1             = {all_rows['macro_f1']:.4f}")

    print("\n" + "=" * 78)
    print("UNSEEN ONLY -- dup_of_train == False  (n = %d)" % n_unseen)
    print("=" * 78)
    print(unseen["report_text"])
    print(f"baseline_9class_macro_f1_unseen_only = {unseen['macro_f1']:.4f}")

    delta = all_rows["macro_f1"] - unseen["macro_f1"]
    print("\n" + "=" * 78)
    print("SUMMARY -- the bar the tuned 9-class model must clear")
    print("=" * 78)
    print(f"  artifact                             : {MODEL_PATH.name} "
          f"({len(v1_classes)}-class, argmax)")
    print(f"  n_test                               : {n_test}")
    print(f"  n_unseen                             : {n_unseen} "
          f"({100.0 * n_unseen / max(n_test, 1):.1f}% of test)")
    print(f"  baseline_9class_macro_f1 (all rows)  : {all_rows['macro_f1']:.4f}")
    print(f"  baseline_9class_macro_f1_unseen_only : {unseen['macro_f1']:.4f}"
          f"   <-- THE HONEST BAR")
    print(f"  memorization premium (all - unseen)  : {delta:+.4f}")
    print("\n  Compare like with like: the tuned model must beat the UNSEEN-ONLY")
    print("  number, measured with the same argmax-or-declared decision rule it")
    print("  actually ships. Rare-class support here is small, so differences")
    print("  under ~0.02 macro-F1 are noise, not progress.")

    # ---- persist ------------------------------------------------------------
    payload = {
        "what": "Honest baseline: the committed v1 artifact scored on the "
                "current 9-class TEST split. Inference only, no retraining.",
        "artifact": str(MODEL_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "artifact_estimator": type(clf).__name__,
        "artifact_classes_raw": sorted(v1_classes),
        "taxonomy_classes": MACRO_LABELS,
        "decision_rule": "argmax (Pipeline.predict), no per-class offsets, "
                         "no threshold",
        "fold": {
            "map": dict(CWE_MERGE_MAP),
            "merged_prediction_counts": dict(sorted(merged.items())),
            "out_of_taxonomy_mapped_to_safe": dict(sorted(out_of_taxonomy.items())),
            "n_out_of_taxonomy_mapped_to_safe": int(sum(out_of_taxonomy.values())),
        },
        "n_test": n_test,
        "n_unseen": n_unseen,
        "n_duplicated_from_train": n_test - n_unseen,
        "baseline_9class_macro_f1": all_rows["macro_f1"],
        "baseline_9class_macro_f1_unseen_only": unseen["macro_f1"],
        "memorization_premium": round(delta, 4),
        "classification_report_all": all_rows["report"],
        "classification_report_unseen_only": unseen["report"],
        "classification_report_all_text": all_rows["report_text"],
        "classification_report_unseen_only_text": unseen["report_text"],
    }
    if args.limit:
        payload["PARTIAL_SMOKE_RUN_limit"] = args.limit

    out = args.out or (None if args.limit else OUT_PATH)
    if out is None:
        print(f"\n(smoke run -- {OUT_PATH.name} not written)")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
