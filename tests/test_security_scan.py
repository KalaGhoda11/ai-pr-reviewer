"""Tests for the deterministic static-scan adapter.

Deliberately stdlib + requirements.txt only (no ``importorskip``): the whole
point of this feature is that it ships in the deployed app, so its tests must
run in the CI job that gates the deploy. ``mlscan.rules`` is pure stdlib and
lives in this repo, so importing it here costs nothing.
"""

import textwrap

from app import webhook
from app.review_engine import (
    SOURCE_BOTH,
    SOURCE_LLM,
    SOURCE_RULES,
    Category,
    Finding,
    ReviewResult,
    Severity,
    format_comment,
)
from app.security_scan import (
    _to_finding,
    is_scannable_path,
    known_issues_block,
    merge_findings,
    parse_unified_diff,
    scan_diff,
)


def _diff(text: str) -> str:
    """Dedent a diff literal without touching its leading +/-/space column."""
    return textwrap.dedent(text).lstrip("\n")


# Adds an SQL injection and an insecure yaml.load to an existing file.
VULN_DIFF = _diff("""
    diff --git a/svc/users.py b/svc/users.py
    index 1111111..2222222 100644
    --- a/svc/users.py
    +++ b/svc/users.py
    @@ -1,6 +1,10 @@
     import yaml
    \x20
    \x20
     def get_user(conn, uid):
    -    return conn.execute("SELECT * FROM users WHERE id = ?", (uid,))
    +    sql = "SELECT * FROM users WHERE id = " + uid
    +    return conn.execute(sql)
    +
    +def load_cfg(blob):
    +    return yaml.load(blob)
""")


# ==========================================================================
# diff parsing
# ==========================================================================

def test_parse_extracts_path_and_real_new_file_line_numbers():
    files = parse_unified_diff(VULN_DIFF)
    assert len(files) == 1
    assert files[0].path == "svc/users.py"
    # Hunk starts at new line 1; three context lines, one deleted, five added.
    assert sorted(files[0].added_lines) == [5, 6, 7, 8, 9]


def test_reconstruct_puts_every_line_at_its_file_line_number():
    text = parse_unified_diff(VULN_DIFF)[0].reconstruct()
    lines = text.split("\n")
    assert lines[0] == "import yaml"
    assert lines[4] == '    sql = "SELECT * FROM users WHERE id = " + uid'
    assert lines[8] == "    return yaml.load(blob)"


def test_reconstruct_pads_the_gap_before_a_mid_file_hunk():
    diff = _diff("""
        --- a/m.py
        +++ b/m.py
        @@ -40,2 +40,3 @@
         def handler(self):
        +    x = 1
         return None
    """)
    text = parse_unified_diff(diff)[0].reconstruct()
    lines = text.split("\n")
    assert lines[:39] == [""] * 39          # gap padded, numbering preserved
    assert lines[39] == "def handler(self):"
    assert lines[40] == "    x = 1"


def test_parse_handles_multiple_files():
    diff = _diff("""
        diff --git a/a.py b/a.py
        --- a/a.py
        +++ b/a.py
        @@ -1,1 +1,2 @@
         one
        +two
        diff --git a/b/nested/c.js b/b/nested/c.js
        --- a/b/nested/c.js
        +++ b/b/nested/c.js
        @@ -5,1 +5,2 @@
         five
        +six
    """)
    files = parse_unified_diff(diff)
    assert [f.path for f in files] == ["a.py", "b/nested/c.js"]
    assert sorted(files[0].added_lines) == [2]
    assert sorted(files[1].added_lines) == [6]


def test_parse_skips_deleted_files():
    diff = _diff("""
        diff --git a/old.py b/old.py
        deleted file mode 100644
        --- a/old.py
        +++ /dev/null
        @@ -1,2 +0,0 @@
        -import yaml
        -yaml.load(x)
    """)
    assert parse_unified_diff(diff) == []


def test_parse_skips_binary_patches():
    diff = _diff("""
        diff --git a/img.png b/img.png
        index 1111111..2222222 100644
        GIT binary patch
        literal 12
        zc$abcdef
    """)
    assert parse_unified_diff(diff) == []


def test_added_line_starting_with_plus_plus_is_not_read_as_a_header():
    # An added line whose own text starts with "++" renders as "+++...". Prefix
    # sniffing would treat it as a new file header; the hunk counts do not.
    diff = _diff("""
        --- a/notes.md
        +++ b/notes.md
        @@ -1,1 +1,3 @@
         a
        +++ b/evil.py
        +@@ -0,0 +1 @@
    """)
    files = parse_unified_diff(diff)
    assert [f.path for f in files] == ["notes.md"]
    assert sorted(files[0].added_lines) == [2, 3]


def test_hunk_header_without_explicit_counts():
    diff = _diff("""
        --- a/x.py
        +++ b/x.py
        @@ -3 +3 @@
        +changed = True
    """)
    assert sorted(parse_unified_diff(diff)[0].added_lines) == [3]


def test_files_with_no_added_lines_are_dropped():
    diff = _diff("""
        --- a/x.py
        +++ b/x.py
        @@ -1,2 +1,1 @@
         keep
        -gone
    """)
    assert parse_unified_diff(diff) == []


# ==========================================================================
# path filtering
# ==========================================================================

def test_is_scannable_path_accepts_source_and_rejects_noise():
    assert is_scannable_path("svc/users.py")
    assert is_scannable_path("b/web/render.js")
    assert is_scannable_path("src/Main.java")

    assert not is_scannable_path("README.md")            # not source
    assert not is_scannable_path("package.json")         # not source
    assert not is_scannable_path("Makefile")             # no extension
    assert not is_scannable_path("tests/test_rules.py")  # test dir AND name
    assert not is_scannable_path("app/conftest.py")
    assert not is_scannable_path("pkg/foo_test.go")
    assert not is_scannable_path("node_modules/lib/a.js")
    assert not is_scannable_path("static/app.min.js")
    assert not is_scannable_path("db/migrations/0001_init.py")


# ==========================================================================
# scanning
# ==========================================================================

def test_scan_reports_added_vulnerabilities_at_real_line_numbers():
    findings = scan_diff(VULN_DIFF)
    by_rule = {f.rule_id: f for f in findings}
    assert set(by_rule) == {"PY-SQL-CONCAT", "PY-YAML-LOAD"}

    sqli = by_rule["PY-SQL-CONCAT"]
    assert sqli.file == "svc/users.py"
    assert sqli.line == 6            # the execute(), as numbered in the file
    assert sqli.cwe == "CWE-89"
    assert sqli.category == Category.security
    assert sqli.severity == Severity.critical
    assert sqli.source == SOURCE_RULES
    assert sqli.suggestion                     # remediation is always attached
    assert "SELECT" in sqli.evidence

    assert by_rule["PY-YAML-LOAD"].line == 9


def test_scan_is_silent_on_a_diff_that_removes_a_vulnerability():
    # The single most important negative case: scanning the raw diff text would
    # fire on the deleted line and report the fix as the defect.
    diff = _diff("""
        --- a/svc/cfg.py
        +++ b/svc/cfg.py
        @@ -1,5 +1,5 @@
         import yaml
        \x20
         def load(blob):
        -    return yaml.load(blob)
        +    return yaml.safe_load(blob)
    """)
    assert scan_diff(diff) == []


def test_scan_ignores_pre_existing_debt_on_context_lines():
    diff = _diff("""
        --- a/svc/pre.py
        +++ b/svc/pre.py
        @@ -1,4 +1,5 @@
         import pickle
        \x20
         def load(b):
             return pickle.loads(b)
        +LIMIT = 10
    """)
    assert scan_diff(diff) == []


def test_context_lines_still_feed_the_cross_statement_flow_analysis():
    # The SQL string is built on a context line and only the execute() is added.
    # A hunk-only reconstruction is what lets the AST rule connect the two.
    diff = _diff("""
        --- a/svc/q.py
        +++ b/svc/q.py
        @@ -1,3 +1,4 @@
         def run(conn, uid):
             sql = "SELECT * FROM users WHERE id = " + uid
        +    return conn.execute(sql)
    """)
    findings = scan_diff(diff)
    assert [(f.rule_id, f.line) for f in findings] == [("PY-SQL-CONCAT", 3)]


def test_scan_skips_test_and_vendor_paths():
    diff = VULN_DIFF.replace("svc/users.py", "tests/test_users.py")
    assert scan_diff(diff) == []


def test_scan_respects_the_file_cap():
    files = []
    for i in range(5):
        files.append(_diff(f"""
            --- a/svc/m{i}.py
            +++ b/svc/m{i}.py
            @@ -1,1 +1,2 @@
             import yaml
            +yaml.load(blob)
        """))
    diff = "".join(files)
    assert len(scan_diff(diff)) == 5
    assert len(scan_diff(diff, max_files=2)) == 2


def test_scan_skips_files_over_the_size_cap():
    assert scan_diff(VULN_DIFF, max_chars=10) == []


def test_findings_are_sorted_most_severe_first():
    diff = _diff("""
        --- a/svc/mix.py
        +++ b/svc/mix.py
        @@ -1,1 +1,4 @@
         import hashlib, yaml
        +def a(p): return hashlib.md5(p.encode()).hexdigest()  # password hash
        +def b(blob): return yaml.load(blob)
    """)
    findings = scan_diff(diff)
    ranks = [f.severity for f in findings]
    assert ranks == sorted(ranks, key=lambda s: [Severity.critical, Severity.major,
                                                 Severity.minor, Severity.info].index(s))


# ==========================================================================
# CWE mapping
# ==========================================================================

def test_cwe_maps_to_a_severity_derived_from_the_defect_class():
    def sev(cwe):
        return _to_finding("f.py", {"cwe": cwe, "name": "n", "line": 1,
                                    "rule_id": "R", "evidence": "e"}).severity

    assert sev("CWE-89") == Severity.critical
    assert sev("CWE-94") == Severity.critical
    assert sev("CWE-502") == Severity.critical
    assert sev("CWE-79") == Severity.major
    assert sev("MEMORY-OOB") == Severity.major
    assert sev("CWE-476") == Severity.minor
    assert sev("CWE-200") == Severity.minor
    assert sev("CWE-9999") == Severity.major     # unknown -> conservative


def test_confidence_is_never_surfaced():
    # rules.py confidences are hardcoded literals, not calibrated probabilities.
    f = _to_finding("f.py", {"cwe": "CWE-89", "name": "SQL Injection", "line": 2,
                             "rule_id": "PY-SQL-CONCAT", "evidence": "e",
                             "confidence": 0.93})
    assert "0.93" not in f.model_dump_json()
    assert "confidence" not in f.model_dump()


# ==========================================================================
# graceful degradation
# ==========================================================================

class _Boom:
    def __call__(self, code):
        raise RuntimeError("rule engine exploded")


def test_a_raising_rule_engine_degrades_to_no_findings():
    assert scan_diff(VULN_DIFF, scanner=_Boom()) == []


def test_a_missing_mlscan_degrades_to_no_findings(monkeypatch):
    monkeypatch.setattr("app.security_scan._load_scan_rules", lambda: None)
    assert scan_diff(VULN_DIFF) == []


def test_malformed_diff_does_not_raise():
    assert scan_diff("not a diff at all") == []
    assert scan_diff("") == []
    assert scan_diff(None) == []


def test_a_rule_finding_without_a_line_is_dropped():
    # No line means we cannot prove the PR introduced it.
    assert scan_diff(VULN_DIFF, scanner=lambda code: [
        {"cwe": "CWE-89", "name": "SQL Injection", "line": None,
         "rule_id": "X", "evidence": ""}]) == []


def test_a_malformed_rule_payload_does_not_lose_the_good_ones():
    def scanner(code):
        return [{"line": 5},                                  # no cwe/name
                {"cwe": "CWE-502", "name": "Insecure Deserialization",
                 "line": 9, "rule_id": "PY-YAML-LOAD", "evidence": "e"}]

    findings = scan_diff(VULN_DIFF, scanner=scanner)
    # Sorted most-severe first: the CWE-502 hit outranks the unclassified one.
    assert [f.cwe for f in findings] == ["CWE-502", None]


# ==========================================================================
# merging with the model's findings
# ==========================================================================

def _rule(file="svc/users.py", line=10, cwe="CWE-89"):
    return _to_finding(file, {"cwe": cwe, "name": "SQL Injection", "line": line,
                              "rule_id": "PY-SQL-CONCAT", "evidence": "sql + uid"})


def _llm(**kw):
    kw.setdefault("file", "svc/users.py")
    kw.setdefault("category", Category.security)
    kw.setdefault("severity", Severity.major)
    kw.setdefault("message", "SQL injection risk here.")
    return Finding(**kw)


def test_duplicate_is_collapsed_into_one_entry_marked_as_both():
    merged = merge_findings([_rule(line=10)], [_llm(line=10)])
    assert len(merged) == 1
    assert merged[0].source == SOURCE_BOTH
    assert merged[0].rule_id == "PY-SQL-CONCAT"     # provenance survives


def test_line_window_absorbs_the_models_off_by_a_couple_line_numbers():
    assert len(merge_findings([_rule(line=10)], [_llm(line=12)])) == 1
    # A second SQL injection fifty lines away is a second finding, even though
    # the prose matches — a supplied line number is trusted over keywords.
    assert len(merge_findings([_rule(line=10)], [_llm(line=60)])) == 2


def test_a_missing_line_number_falls_back_to_cwe_keywords():
    merged = merge_findings([_rule()], [_llm(line=None)])
    assert len(merged) == 1 and merged[0].source == SOURCE_BOTH

    # Same file, security, no line, but a different defect -> kept separately.
    apart = merge_findings([_rule()],
                           [_llm(line=None, message="Missing CSRF token check.")])
    assert len(apart) == 2


def test_path_prefixes_and_backticks_do_not_defeat_the_match():
    merged = merge_findings([_rule(line=10)], [_llm(file="b/svc/users.py", line=10)])
    assert len(merged) == 1
    merged = merge_findings([_rule(line=10)], [_llm(file="`svc/users.py`", line=10)])
    assert len(merged) == 1


def test_a_non_security_finding_on_the_same_line_is_never_suppressed():
    other = _llm(line=10, category=Category.bug,
                 message="fetchone() result is not checked for None.")
    merged = merge_findings([_rule(line=10)], [other])
    assert len(merged) == 2
    assert merged[1].category == Category.bug


def test_unrelated_model_findings_are_preserved_in_order():
    a = _llm(file="other.py", line=3, category=Category.refactor, message="a")
    b = _llm(file="other.py", line=4, category=Category.style, message="b")
    merged = merge_findings([_rule()], [a, b])
    assert [f.message for f in merged[1:]] == ["a", "b"]
    assert all(f.source == SOURCE_LLM for f in merged[1:])


def test_agreement_keeps_the_models_suggestion_and_can_only_raise_severity():
    llm = _llm(line=10, severity=Severity.critical,
               suggestion="Bind uid via a placeholder in this call.")
    merged = merge_findings([_rule(line=10)], [llm])
    assert merged[0].suggestion == "Bind uid via a placeholder in this call."
    assert merged[0].severity == Severity.critical

    quiet = _llm(line=10, severity=Severity.info, suggestion="")
    merged = merge_findings([_rule(line=10)], [quiet])
    assert merged[0].severity == Severity.critical      # not downgraded to info
    assert merged[0].suggestion.startswith("Use a parameterised query")


def test_one_model_finding_confirms_only_one_rule_finding():
    rules = [_rule(line=10), _rule(line=11)]
    merged = merge_findings(rules, [_llm(line=10)])
    assert [f.source for f in merged] == [SOURCE_BOTH, SOURCE_RULES]


def test_merge_does_not_mutate_the_inputs():
    rules = [_rule(line=10)]
    merge_findings(rules, [_llm(line=10)])
    assert rules[0].source == SOURCE_RULES


def test_known_issues_block_names_location_cwe_and_rule():
    assert known_issues_block([_rule(line=10)]) == [
        "svc/users.py:10 — CWE-89 (PY-SQL-CONCAT)"]
    assert known_issues_block([]) == []


# ==========================================================================
# rendering
# ==========================================================================

def test_comment_shows_provenance_evidence_and_the_caveat():
    result = ReviewResult(summary="s", findings=merge_findings(
        [_rule(line=10)], [_llm(line=99, category=Category.bug, message="unrelated")]))
    body = format_comment(result)
    assert "static analysis" in body
    assert "PY-SQL-CONCAT" in body
    assert "sql + uid" in body                 # evidence rendered
    assert "deterministic pattern rules" in body
    assert "0.9" not in body                   # no confidence anywhere


def test_comment_for_model_only_findings_is_unchanged():
    result = ReviewResult(summary="s", findings=[_llm(line=3)])
    body = format_comment(result)
    assert "static analysis" not in body
    assert "deterministic pattern rules" not in body
    assert body.endswith("_Generated by AI PR Reviewer._")


# ==========================================================================
# webhook wiring
# ==========================================================================

class _FakeGemini:
    """Records the prompt it was given and replays a canned review."""

    def __init__(self, payload):
        self.prompts = []
        outer = self

        class _Models:
            def generate_content(self, model, contents):
                outer.prompts.append(contents)
                return type("R", (), {"text": payload})()

        self.models = _Models()


_REVIEW_JSON = ('{"summary": "ok", "findings": [{"file": "svc/users.py", '
                '"line": 6, "category": "security", "severity": "major", '
                '"message": "SQL injection via string concatenation.", '
                '"suggestion": "Parameterise it."}]}')


def _wire(monkeypatch, gemini, diff=VULN_DIFF):
    posted = {}
    monkeypatch.setattr(webhook, "fetch_diff", lambda url, token: diff)
    monkeypatch.setattr(webhook, "_get_gemini_client", lambda key: gemini)
    monkeypatch.setattr(webhook, "post_review_comment",
                        lambda repo, num, body, token: posted.update(body=body))
    return posted


PAYLOAD = {"pull_request": {"number": 1, "url": "http://api/pr/1"},
           "repository": {"full_name": "o/r"}}


def test_webhook_posts_one_comment_holding_both_kinds_of_finding(monkeypatch):
    gemini = _FakeGemini(_REVIEW_JSON)
    posted = _wire(monkeypatch, gemini)

    webhook.process_pull_request(PAYLOAD)

    body = posted["body"]
    # The model's SQLi restates the rule's, so they collapse into one entry...
    assert body.count("### ") == 2                 # SQLi (merged) + yaml.load
    assert "static analysis + model" in body
    assert "PY-YAML-LOAD" in body
    # ...and the model was told about them up front so it would not repeat them.
    assert "static analyzer has ALREADY reported" in gemini.prompts[0]
    assert "svc/users.py:6 — CWE-89" in gemini.prompts[0]


def test_webhook_falls_back_to_gemini_only_when_the_scan_explodes(monkeypatch):
    def boom(diff):
        raise RuntimeError("scan blew up")

    monkeypatch.setattr(webhook, "scan_diff", boom)
    gemini = _FakeGemini(_REVIEW_JSON)
    posted = _wire(monkeypatch, gemini)

    webhook.process_pull_request(PAYLOAD)

    body = posted["body"]
    assert "SQL injection via string concatenation." in body   # review still posted
    assert "static analysis" not in body
