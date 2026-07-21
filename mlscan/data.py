"""Load and prepare the training data from the real public dataset.

Dataset: ayshajavd/code-security-vulnerability-dataset (HuggingFace) — 175k real
code samples labelled with CWE / OWASP / is_vulnerable.

We reduce it to a 9-class problem: the 8 vulnerability classes in
``labels.TAXONOMY`` plus "safe". Raw CWE ids are folded through
``labels.CWE_MERGE_MAP`` first, so CWE-119 / CWE-787 / CWE-125 all collapse into
the single ``MEMORY-OOB`` class; vulnerable rows outside the taxonomy are
dropped.

Two data-hygiene decisions are baked in here, because both materially change
what the reported metrics mean:

1. **TRAIN-ONLY de-duplication (``dedup=True``).** The published splits leak:
   many validation/test rows are byte-identical to a train row once the code is
   truncated to ``MAX_CODE_CHARS`` — i.e. identical in the exact string the
   model sees. Measured on the folded 9-class data, 1557 of the 17121 kept test
   rows (9.1%) duplicated a train row, concentrated in the rare classes
   (CWE-89 82.7%, CWE-200 66.3%, MEMORY-OOB 62.6%) versus only 4.3% of "safe"
   rows. Left alone, macro-F1 is largely a memorization score. We therefore drop
   every TRAIN row whose truncated-code hash appears in validation or test.
   Validation and test are **never** modified or filtered — they stay the honest
   held-out sample. Each eval Split also carries ``dup_of_train``, flagging the
   rows that duplicated the *original* (pre-dedup) train set, so downstream code
   can additionally report "unseen-only" metrics. Pass ``dedup=False`` to
   reproduce the leaky behaviour for an A/B measurement.

2. **A single imbalance correction (``SAFE_RATIO = 4.0``).** The raw data is
   ~91% "safe". Down-sampling TRAIN to 1:1 *and* stacking
   ``class_weight='balanced'`` on top corrects the same imbalance twice and
   over-shoots: measured validation macro-F1 was 0.4987 for 1:1 + balanced,
   0.5661 for 1:1 with no weights, and 0.5980 for 4:1 with no weights. 4:1 keeps
   the training prior near the ~91%-safe evaluation splits, so the model is not
   asked to learn a decision boundary calibrated for a prior it will never see.
   Do not re-apply ``class_weight='balanced'`` on top of this ratio.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from mlscan.labels import SAFE, fold_cwe

DATASET_ID = "ayshajavd/code-security-vulnerability-dataset"
MAX_CODE_CHARS = 4000  # truncate very long samples; vuln signals are local
SEED = 42

# Down-sampling target for the TRAIN split, as a multiple of the vulnerable row
# count (1.0 = one "safe" row per vulnerable row). 4.0 keeps the training prior
# close to the ~91%-safe evaluation splits and is the measured best setting when
# no class weighting is applied on top (see module docstring).
SAFE_RATIO = 4.0

# Floor on the number of "safe" rows kept, so a small/filtered corpus still has
# enough negatives to fit a vectorizer on.
MIN_SAFE_ROWS = 3000


@dataclass
class Split:
    codes: list[str]
    labels: list[str]
    # For evaluation splits: parallel to ``codes``, True where the row was
    # byte-identical (on the truncated code) to a row of the ORIGINAL, pre-dedup
    # TRAIN split. ``None`` for the train split itself, and whenever the
    # information was not computed. Declared last with a default so existing
    # positional callers — ``Split(codes, labels)`` — keep working.
    dup_of_train: list[bool] | None = None


def _row_label(is_vuln, cwe_id) -> str | None:
    """Map a dataset row to our label, or None to drop it.

    Vulnerable rows are folded through ``CWE_MERGE_MAP`` (the three memory CWEs
    become ``MEMORY-OOB``); anything still outside the taxonomy returns None.
    """
    if not is_vuln:
        return SAFE
    return fold_cwe(cwe_id)


def code_hash(code: str) -> str:
    """Hash of the exact string the model sees (already truncated).

    Used for the train/eval overlap check. md5 is fine here: this is a
    duplicate-detection key, not a security primitive.
    """
    return hashlib.md5(code.encode("utf-8", "replace")).hexdigest()


def _labelled_frame(df):
    """Apply the taxonomy, drop out-of-taxonomy rows, truncate, hash."""
    df = df.copy()
    df["label"] = [
        _row_label(v, c) for v, c in zip(df["is_vulnerable"], df["cwe_id"])
    ]
    df = df[df["label"].notna()]
    df["code"] = df["code"].astype(str).str.slice(0, MAX_CODE_CHARS)
    df["code_hash"] = [code_hash(c) for c in df["code"]]
    return df


def _downsample_safe(df, safe_ratio: float):
    """Down-sample "safe" rows to ``safe_ratio`` x the vulnerable row count."""
    import pandas as pd  # local import so importing this module is cheap

    vuln = df[df["label"] != SAFE]
    safe = df[df["label"] == SAFE]
    target = max(int(round(len(vuln) * safe_ratio)), MIN_SAFE_ROWS)
    n = min(len(safe), target)
    safe = safe.sample(n=n, random_state=SEED)
    return pd.concat([vuln, safe]).sample(frac=1, random_state=SEED)


def _prepare_frame(df, balance_safe: bool, safe_ratio: float = SAFE_RATIO):
    """Label + optionally balance a single frame (no cross-split de-dup)."""
    df = _labelled_frame(df)
    if balance_safe:
        df = _downsample_safe(df, safe_ratio)
    return Split(codes=df["code"].tolist(), labels=df["label"].tolist())


def load_splits(safe_ratio: float = SAFE_RATIO, dedup: bool = True) -> dict[str, Split]:
    """Return prepared {'train','validation','test'} splits.

    ``safe_ratio`` controls the TRAIN down-sampling only; validation and test
    keep their natural class distribution.

    ``dedup`` (default True) drops TRAIN rows whose truncated code is
    byte-identical to a validation or test row, before the "safe" down-sampling
    so the ratio is computed on the rows actually kept. Validation and test are
    never filtered; they instead carry ``dup_of_train``, marking rows that were
    duplicates of the original pre-dedup train set. Set ``dedup=False`` to
    reproduce the leaky splits for comparison (``dup_of_train`` is still
    populated, so the leakage remains measurable).
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID)
    frames = {name: _labelled_frame(ds[name].to_pandas())
              for name in ("train", "validation", "test")}

    # Hashes of the ORIGINAL train split — what the eval rows are flagged
    # against, regardless of whether we go on to de-duplicate.
    train_hashes = set(frames["train"]["code_hash"])

    out: dict[str, Split] = {}
    for name in ("validation", "test"):
        f = frames[name]
        out[name] = Split(
            codes=f["code"].tolist(),
            labels=f["label"].tolist(),
            dup_of_train=[h in train_hashes for h in f["code_hash"]],
        )

    train = frames["train"]
    if dedup:
        eval_hashes = set(frames["validation"]["code_hash"]) | set(
            frames["test"]["code_hash"])
        train = train[~train["code_hash"].isin(eval_hashes)]
    train = _downsample_safe(train, safe_ratio)
    out["train"] = Split(codes=train["code"].tolist(),
                         labels=train["label"].tolist())
    return out
