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

def test_taxonomy_is_eight_plus_safe():
    # CWE-119 / 787 / 125 were merged into MEMORY-OOB, so 8 vuln classes + safe.
    assert len(TAXONOMY) == 8
    assert len(CLASSES) == 9
    assert SAFE in CLASSES
    assert "MEMORY-OOB" in CLASSES


def test_describe_safe_and_cwe():
    assert describe(SAFE)["cwe"] is None
    d = describe("CWE-89")
    assert d["cwe"] == "CWE-89"
    assert "SQL" in d["name"]


# ---- model behaviour (verified against the trained artifact) ----

def test_classify_returns_probability_distribution():
    ranked = classify("def add(a, b):\n    return a + b")
    # The served artifact may be the 9-class v2 or the legacy 11-class v1, so
    # assert the distribution properties rather than an exact class count.
    assert len(ranked) >= len(CLASSES) - 1
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
    assert set(f) >= {"cwe", "name", "owasp", "description", "confidence", "source"}
    assert 0.0 <= f["confidence"] <= 1.0
    assert f["source"] in {"ml", "rule", "ml+rule"}


# ---- hybrid layer: rules cover the classifier's known brittleness ----

def test_rules_catch_what_the_classifier_alone_misses():
    """os.system concatenation scores <threshold for the ML model on its own."""
    code = 'import os\ndef ping(host):\n    os.system("ping " + host)'
    assert scan(code, use_rules=False)["is_vulnerable"] is False   # ML alone misses it
    hybrid = scan(code)                                            # hybrid catches it
    assert hybrid["is_vulnerable"] is True
    assert any(f["cwe"] == "CWE-94" for f in hybrid["findings"])


def test_detection_is_stable_across_identifier_renaming():
    """The ML model's confidence swings with variable names; rules must not."""
    long_names = ("def f(user):\n    q = \"SELECT * FROM accounts WHERE id='\" "
                  "+ user + \"'\"\n    db.execute(q)")
    short_names = ("def f(u):\n    q = \"SELECT * FROM t WHERE id='\" "
                   "+ u + \"'\"\n    db.execute(q)")
    for code in (long_names, short_names):
        assert any(f["cwe"] == "CWE-89" for f in scan(code)["findings"])


def test_hybrid_does_not_flag_secure_equivalents():
    safe_snippets = [
        'def f(u, db):\n    return db.execute("SELECT * FROM t WHERE id=?", (u,))',
        'import subprocess\ndef r(h):\n    return subprocess.run(["ping", h], shell=False)',
        'import hashlib\ndef h(x):\n    return hashlib.sha256(x).hexdigest()',
    ]
    for code in safe_snippets:
        assert scan(code)["is_vulnerable"] is False, code
