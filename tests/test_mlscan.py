"""Tests for the standalone ML vulnerability scanner.

These require the ML extras (scikit-learn/lightgbm) and the trained model
artifact. They skip cleanly where those aren't present (e.g. the web-app CI job
that only installs requirements.txt), so they never break the main pipeline.
"""

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from mlscan.labels import CLASSES, SAFE, TAXONOMY, describe  # noqa: E402
from mlscan.scanner import MODEL_PATH, classify, scan  # noqa: E402

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="trained model artifact not present"
)


# ---- taxonomy (no model needed) ----

def test_taxonomy_is_ten_plus_safe():
    assert len(TAXONOMY) == 10
    assert len(CLASSES) == 11
    assert SAFE in CLASSES


def test_describe_safe_and_cwe():
    assert describe(SAFE)["cwe"] is None
    d = describe("CWE-89")
    assert d["cwe"] == "CWE-89"
    assert "SQL" in d["name"]


# ---- model behaviour (verified against the trained artifact) ----

def test_classify_returns_probability_distribution():
    ranked = classify("def add(a, b):\n    return a + b")
    assert len(ranked) == len(CLASSES)
    total = sum(p for _, p in ranked)
    assert 0.98 <= total <= 1.02          # a proper distribution
    assert ranked == sorted(ranked, key=lambda t: t[1], reverse=True)


def test_detects_sql_injection():
    r = scan("def f(user):\n    q = \"SELECT * FROM accounts WHERE id='\" + user + \"'\"\n    db.execute(q)")
    assert r["is_vulnerable"] is True
    assert any(f["cwe"] == "CWE-89" for f in r["findings"])


def test_detects_code_injection():
    r = scan("def run(user_input):\n    return eval(user_input)")
    assert r["is_vulnerable"] is True
    assert any(f["cwe"] == "CWE-94" for f in r["findings"])


def test_safe_code_not_flagged():
    assert scan("def add(a, b):\n    return a + b")["is_vulnerable"] is False
    assert scan("import hashlib\ndef h(x):\n    return hashlib.sha256(x).hexdigest()")["is_vulnerable"] is False


def test_findings_carry_metadata():
    r = scan("def run(user_input):\n    return eval(user_input)")
    f = r["findings"][0]
    assert set(f) >= {"cwe", "name", "owasp", "description", "confidence"}
    assert 0.0 <= f["confidence"] <= 1.0
