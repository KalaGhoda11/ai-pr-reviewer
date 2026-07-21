"""Tests for the train-only de-duplication in :mod:`mlscan.data`.

The real corpus is a 175k-row HuggingFace download, so these tests substitute a
tiny in-memory dataset for ``datasets.load_dataset`` and assert the invariants:
TRAIN loses the rows that leak into validation/test, the evaluation splits are
never touched, and the leaked rows stay flagged via ``dup_of_train``.
"""

import pytest

pd = pytest.importorskip("pandas")
datasets = pytest.importorskip("datasets")

from mlscan.data import (  # noqa: E402
    MAX_CODE_CHARS,
    MIN_SAFE_ROWS,
    SAFE_RATIO,
    Split,
    code_hash,
    load_splits,
)


class _FakeSplit:
    def __init__(self, frame):
        self._frame = frame

    def to_pandas(self):
        return self._frame.copy()


def _frame(rows):
    return pd.DataFrame(
        [{"code": c, "is_vulnerable": v, "cwe_id": w} for c, v, w in rows]
    )


def _vuln(tag, n, cwe="CWE-89"):
    return [(f"{tag}-vuln-{i}", True, cwe) for i in range(n)]


def _safe(tag, n):
    return [(f"{tag}-safe-{i}", False, None) for i in range(n)]


@pytest.fixture
def fake_dataset(monkeypatch):
    """Install a tiny fake corpus and return the raw row lists."""

    def install(train_rows, val_rows, test_rows):
        fake = {
            "train": _FakeSplit(_frame(train_rows)),
            "validation": _FakeSplit(_frame(val_rows)),
            "test": _FakeSplit(_frame(test_rows)),
        }
        monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake)
        return fake

    return install


# ---- Split backwards compatibility ----

def test_split_still_constructs_positionally():
    s = Split(["a"], ["safe"])
    assert s.codes == ["a"] and s.labels == ["safe"]
    assert s.dup_of_train is None


def test_split_accepts_keywords():
    s = Split(codes=["a"], labels=["safe"])
    assert s.dup_of_train is None


# ---- de-duplication ----

def test_dedup_drops_leaked_train_rows_and_leaves_eval_alone(fake_dataset):
    shared = _vuln("shared", 5)
    fake_dataset(
        train_rows=shared + _vuln("trainonly", 5) + _safe("train", 200),
        val_rows=_vuln("valonly", 3) + _safe("val", 30),
        test_rows=shared + _vuln("testonly", 3) + _safe("test", 30),
    )

    leaky = load_splits(dedup=False)
    clean = load_splits(dedup=True)

    leaked = {code_hash(c) for c, _, _ in shared}
    assert leaked <= {code_hash(c) for c in leaky["train"].codes}
    assert not leaked & {code_hash(c) for c in clean["train"].codes}
    assert len(clean["train"].codes) == len(leaky["train"].codes) - len(shared)

    # evaluation splits are byte-for-byte identical either way
    for name in ("validation", "test"):
        assert clean[name].codes == leaky[name].codes
        assert clean[name].labels == leaky[name].labels


def test_no_hash_overlap_remains_between_train_and_eval(fake_dataset):
    shared = _vuln("shared", 4)
    fake_dataset(
        train_rows=shared + _vuln("trainonly", 6) + _safe("train", 100),
        val_rows=shared[:2] + _safe("val", 20),
        test_rows=shared[2:] + _safe("test", 20),
    )
    splits = load_splits()
    train = {code_hash(c) for c in splits["train"].codes}
    for name in ("validation", "test"):
        assert not train & {code_hash(c) for c in splits[name].codes}


def test_dedup_uses_the_truncated_string_the_model_sees(fake_dataset):
    """Rows differing only past MAX_CODE_CHARS are the same to the model."""
    prefix = "x" * MAX_CODE_CHARS
    fake_dataset(
        train_rows=[(prefix + "TRAIN-TAIL", True, "CWE-89")]
        + _vuln("trainonly", 3) + _safe("train", 50),
        val_rows=_safe("val", 10),
        test_rows=[(prefix + "TEST-TAIL", True, "CWE-89")] + _safe("test", 10),
    )
    splits = load_splits()
    assert all(not c.startswith(prefix) for c in splits["train"].codes)
    assert splits["test"].codes[0] == prefix  # eval row kept, just truncated
    assert splits["test"].dup_of_train[0] is True


# ---- dup_of_train bookkeeping ----

def test_dup_of_train_flags_rows_against_the_original_train_split(fake_dataset):
    shared = _vuln("shared", 3)
    fake_dataset(
        train_rows=shared + _vuln("trainonly", 4) + _safe("train", 60),
        val_rows=_safe("val", 10),
        test_rows=shared + _vuln("testonly", 4) + _safe("test", 10),
    )
    for dedup in (True, False):
        splits = load_splits(dedup=dedup)
        test = splits["test"]
        assert len(test.dup_of_train) == len(test.codes)
        flagged = {c for c, f in zip(test.codes, test.dup_of_train) if f}
        assert flagged == {c for c, _, _ in shared}


def test_train_split_has_no_dup_flags(fake_dataset):
    fake_dataset(
        train_rows=_vuln("t", 5) + _safe("train", 40),
        val_rows=_safe("val", 5),
        test_rows=_safe("test", 5),
    )
    assert load_splits()["train"].dup_of_train is None


# ---- imbalance correction ----

def test_default_safe_ratio_is_four_to_one():
    assert SAFE_RATIO == 4.0


def test_safe_ratio_controls_train_only(fake_dataset):
    n_vuln = MIN_SAFE_ROWS  # large enough that the MIN_SAFE_ROWS floor is inert
    fake_dataset(
        train_rows=_vuln("t", n_vuln) + _safe("train", 10 * n_vuln),
        val_rows=_vuln("v", 5) + _safe("val", 50),
        test_rows=_vuln("e", 5) + _safe("test", 50),
    )
    splits = load_splits(safe_ratio=4.0)
    labels = splits["train"].labels
    assert labels.count("safe") == 4 * n_vuln
    assert len(labels) - labels.count("safe") == n_vuln
    # eval splits keep their natural distribution
    assert splits["test"].labels.count("safe") == 50
