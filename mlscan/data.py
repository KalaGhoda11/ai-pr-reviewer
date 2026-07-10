"""Load and prepare the training data from the real public dataset.

Dataset: ayshajavd/code-security-vulnerability-dataset (HuggingFace) — 175k real
code samples labelled with CWE / OWASP / is_vulnerable.

We reduce it to an 11-class problem: the 10 vulnerability classes in
``labels.TAXONOMY`` plus "safe". Because the raw data is ~90% "safe", the TRAIN
split is balanced by down-sampling "safe"; validation/test are left at their
natural distribution for an honest evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlscan.labels import SAFE, TAXONOMY

DATASET_ID = "ayshajavd/code-security-vulnerability-dataset"
MAX_CODE_CHARS = 4000  # truncate very long samples; vuln signals are local
SEED = 42


@dataclass
class Split:
    codes: list[str]
    labels: list[str]


def _row_label(is_vuln, cwe_id) -> str | None:
    """Map a dataset row to our label, or None to drop it."""
    if not is_vuln:
        return SAFE
    if cwe_id in TAXONOMY:
        return cwe_id
    return None  # vulnerable but not one of our 10 classes -> drop


def _prepare_frame(df, balance_safe: bool):
    import pandas as pd  # local import so importing this module is cheap

    df = df.copy()
    df["label"] = [
        _row_label(v, c) for v, c in zip(df["is_vulnerable"], df["cwe_id"])
    ]
    df = df[df["label"].notna()]
    df["code"] = df["code"].astype(str).str.slice(0, MAX_CODE_CHARS)

    if balance_safe:
        vuln = df[df["label"] != SAFE]
        safe = df[df["label"] == SAFE]
        n = min(len(safe), max(len(vuln), 3000))
        safe = safe.sample(n=n, random_state=SEED)
        df = pd.concat([vuln, safe]).sample(frac=1, random_state=SEED)

    return Split(codes=df["code"].tolist(), labels=df["label"].tolist())


def load_splits() -> dict[str, Split]:
    """Return prepared {'train','validation','test'} splits."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID)
    return {
        "train": _prepare_frame(ds["train"].to_pandas(), balance_safe=True),
        "validation": _prepare_frame(ds["validation"].to_pandas(), balance_safe=False),
        "test": _prepare_frame(ds["test"].to_pandas(), balance_safe=False),
    }
