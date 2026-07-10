"""Train the vulnerability classifier: compare models, keep the best, save it.

Run:  python -m mlscan.train

Trains two candidates on the same TF-IDF features and keeps whichever scores
the higher macro-F1 on the validation split:
  1. Logistic Regression (one-vs-rest, balanced) — strong linear baseline for
     sparse text.
  2. LightGBM (gradient-boosted trees, multiclass).
The winner is evaluated on the held-out test split and saved as a single
scikit-learn Pipeline (vectorizer + classifier) to model/vuln_clf.joblib.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

from mlscan.data import load_splits
from mlscan.features import build_vectorizer

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "vuln_clf.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
SEED = 42


def _candidates():
    yield "logreg", LogisticRegression(
        max_iter=2000, C=4.0, class_weight="balanced", n_jobs=-1,
    )
    try:
        from lightgbm import LGBMClassifier

        yield "lightgbm", LGBMClassifier(
            objective="multiclass", n_estimators=400, learning_rate=0.1,
            num_leaves=63, class_weight="balanced", n_jobs=-1,
            random_state=SEED, verbose=-1,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"(lightgbm unavailable: {exc})")


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    print("Loading + preparing data...", flush=True)
    splits = load_splits()
    for name, s in splits.items():
        print(f"  {name}: {len(s.codes)} samples, {len(set(s.labels))} classes", flush=True)

    print("Vectorizing (fit on train)...", flush=True)
    vec = build_vectorizer()
    t0 = time.time()
    Xtr = vec.fit_transform(splits["train"].codes)
    Xval = vec.transform(splits["validation"].codes)
    Xte = vec.transform(splits["test"].codes)
    ytr, yval, yte = (splits[k].labels for k in ("train", "validation", "test"))
    print(f"  features: {Xtr.shape[1]}  (vectorized in {time.time()-t0:.1f}s)", flush=True)

    results = {}
    best = (None, None, -1.0)  # name, model, val_macro_f1
    for name, model in _candidates():
        t0 = time.time()
        print(f"Training {name}...", flush=True)
        model.fit(Xtr, ytr)
        val_f1 = f1_score(yval, model.predict(Xval), average="macro")
        results[name] = {"val_macro_f1": round(val_f1, 4), "train_secs": round(time.time()-t0, 1)}
        print(f"  {name}: validation macro-F1 = {val_f1:.4f} ({results[name]['train_secs']}s)", flush=True)
        if val_f1 > best[2]:
            best = (name, model, val_f1)

    best_name, best_model, _ = best
    print(f"\nBest model: {best_name}", flush=True)

    print("Evaluating best model on TEST split...", flush=True)
    yte_pred = best_model.predict(Xte)
    report = classification_report(yte, yte_pred, digits=3, zero_division=0)
    test_macro_f1 = f1_score(yte, yte_pred, average="macro")
    print(report, flush=True)
    print(f"TEST macro-F1 = {test_macro_f1:.4f}", flush=True)

    # Save the winner as one Pipeline (vectorizer already fitted + classifier).
    pipeline = Pipeline([("features", vec), ("clf", best_model)])
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"Saved model -> {MODEL_PATH} ({size_mb:.1f} MB)", flush=True)

    METRICS_PATH.write_text(json.dumps({
        "dataset": "ayshajavd/code-security-vulnerability-dataset",
        "best_model": best_name,
        "candidates": results,
        "test_macro_f1": round(test_macro_f1, 4),
        "classes": sorted(set(ytr)),
        "test_classification_report": report,
        "n_train": len(ytr), "n_val": len(yval), "n_test": len(yte),
        "n_features": int(Xtr.shape[1]),
    }, indent=2), encoding="utf-8")
    print(f"Saved metrics -> {METRICS_PATH}", flush=True)


if __name__ == "__main__":
    main()
