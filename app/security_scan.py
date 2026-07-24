"""Deterministic security findings for the deployed PR reviewer.

Why this module exists
----------------------
Without it the reviewer is a single Gemini call: every finding it posts is a
model opinion -- not reproducible, not attributable, and not something a
reviewer can audit. ``mlscan.rules`` is a rule engine that already ships in this
repo, fires only on constructs that are unsafe *by definition*, and imports
nothing but ``re``/``ast``/``bisect``. Wiring it in gives the bot a second,
deterministic opinion for **zero new dependencies**.

Three things this module deliberately does NOT do:

* **It never imports the ML side of mlscan.** ``mlscan.scanner`` /
  ``inference`` / ``features`` / ``security_features`` pull scikit-learn, numpy
  and scipy (measured: +167 MB RSS for one ``scan()``), which does not fit the
  512 MB deployment. Only ``mlscan.rules`` is used, and even that is imported
  *lazily* inside :func:`scan_diff` behind a try/except so a container that
  strips ``mlscan/`` degrades to Gemini-only instead of failing to boot.
* **It never runs the rules over the raw unified diff.** A diff contains
  DELETED lines, so a PR that *removes* ``yaml.load`` would be reported as
  introducing CWE-502; its line numbers are offsets into the diff rather than
  into any file; a multi-file diff cannot attribute a finding to a file; and it
  does not parse, so the precise AST rules are silently skipped. Instead each
  file's new-side image is reconstructed from the hunks -- context *and* added
  lines placed at their real new-file line numbers -- and scanned per file.
* **It never reports a finding on a line the PR did not add.** Context lines are
  scanned, because they are what makes the fragment parse and what the SQL flow
  analysis needs to connect ``sql = "..." + uid`` to ``execute(sql)``. But a
  finding that lands on one is pre-existing debt, and would be re-posted on
  every ``synchronize`` push.

Severity is derived from the CWE class and never from the rule's ``confidence``
float: those confidences are hardcoded literals at each call site in
``mlscan.rules``, not calibrated probabilities, so printing them in a PR comment
would be exactly the kind of unbacked precision claim this project is trying to
avoid.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.review_engine import (
    SOURCE_BOTH,
    SOURCE_RULES,
    Category,
    Finding,
    Severity,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# budget
# --------------------------------------------------------------------------
# The free-tier instance has ~0.1 CPU, and the scan runs in a worker thread that
# still holds the GIL. mlscan's own ceiling (200 KB) costs ~1.8 s per file; 60 KB
# measures ~240 ms, which is the knee of the curve.
MAX_SCANNED_FILES = 20
MAX_FILE_CHARS = 60_000

# --------------------------------------------------------------------------
# what is worth scanning
# --------------------------------------------------------------------------
# Only source files the rule engine actually has rules for. Scanning docs, JSON
# or lockfiles buys nothing and costs precision (a "password" key in a fixture
# JSON is not a credential).
SCANNABLE_SUFFIXES = frozenset({
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".java", ".kt", ".scala",
    ".php", ".rb", ".go", ".cs", ".swift", ".pl",
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh",
})

# Test, fixture and vendored trees are where deliberate vulnerable constructs
# live. Measured on 400 random real-world .py files, 73% of all rule findings
# came from such paths -- this repo's own tests/test_rules.py draws five.
_SKIP_DIR_RE = re.compile(
    r"(^|/)(tests?|testing|fixtures?|testdata|vendor|third_party|node_modules"
    r"|migrations|__pycache__|site-packages|\.venv)(/|$)",
    re.IGNORECASE,
)
_SKIP_NAME_RE = re.compile(
    r"^(conftest\.py|test_.+\.py|.+_test\.[A-Za-z0-9]+|.+\.min\.(js|css)"
    r"|.+\.lock|.+\.map)$",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# CWE -> how we present it
# --------------------------------------------------------------------------
# Derived from the defect class, NOT from rules.py's hardcoded confidence.
_SEVERITY_BY_CWE = {
    "CWE-89": Severity.critical,    # SQL injection -> data breach
    "CWE-94": Severity.critical,    # code injection -> RCE
    "CWE-502": Severity.critical,   # insecure deserialization -> RCE
    "CWE-79": Severity.major,       # XSS -> session/credential theft
    "MEMORY-OOB": Severity.major,   # out-of-bounds access -> memory corruption
    "CWE-476": Severity.minor,      # NULL deref -> crash
    "CWE-200": Severity.minor,      # information exposure
    "CWE-20": Severity.minor,       # residual "improper input validation"
}
_DEFAULT_SEVERITY = Severity.major

_REMEDIATION = {
    "CWE-89": "Use a parameterised query: pass the value as a bind parameter "
              "instead of concatenating or formatting it into the SQL string.",
    "CWE-79": "Escape the value where it is rendered, and do not mark "
              "untrusted strings as safe (mark_safe / |safe / innerHTML).",
    "CWE-94": "Do not evaluate untrusted input. Use ast.literal_eval, an "
              "explicit dispatch table, or an argument list with shell=False.",
    "CWE-502": "Deserialise untrusted data with a safe loader "
               "(yaml.safe_load, json.loads); never unpickle bytes an "
               "attacker can influence.",
    "CWE-200": "Do not disable certificate/transport verification, and read "
               "secrets from the environment instead of committing them.",
    "CWE-476": "Check the pointer against NULL before dereferencing it.",
    "MEMORY-OOB": "Bound the write to the destination's declared size "
                  "(snprintf / strncpy with an explicit length) and validate "
                  "every index against it.",
}

# Words that let us match a rule finding to an LLM finding whose line number is
# missing or wrong. Deliberately specific: "injection" alone would match a rule
# and an LLM finding that are talking about different defects.
_CWE_KEYWORDS = {
    "CWE-89": ("sql injection", "sqli", "sql query", "sql statement"),
    "CWE-79": ("xss", "cross-site scripting", "cross site scripting",
               "html escap", "innerhtml"),
    "CWE-94": ("code injection", "command injection", "remote code",
               "arbitrary code", "eval(", "exec(", "shell=true"),
    "CWE-502": ("deserial", "unserial", "pickle", "yaml.load", "marshal"),
    "CWE-200": ("information exposure", "information disclosure",
                "hardcoded", "hard-coded", "certificate verif",
                "verify=false", "weak hash"),
    "CWE-476": ("null pointer", "null deref", "nullptr"),
    "MEMORY-OOB": ("buffer overflow", "out-of-bounds", "out of bounds",
                   "overrun", "strcpy"),
}

# The model's line number is unreliable by design (review_engine._coerce_line
# already documents that it arrives as ranges, strings and zeros), so an exact
# match would under-merge.
LINE_MATCH_WINDOW = 3


# ==========================================================================
# unified diff parsing
# ==========================================================================

# Standard two-way hunk header. Combined diffs (``@@@``) come from merges and
# are not produced for PR diffs, so they are ignored rather than mis-parsed.
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class DiffFile:
    """One file's new-side view, recovered from its hunks.

    ``new_side`` maps a *new-file* line number to its text for every line the
    diff shows us (context and added). ``added_lines`` is the subset the PR
    actually introduced -- the only lines we are allowed to report on.
    """

    path: str
    added_lines: set[int] = field(default_factory=set)
    new_side: dict[int, str] = field(default_factory=dict)

    def reconstruct(self) -> str:
        """Render the known lines at their true line numbers, gaps blanked.

        Padding the gaps is what keeps a finding's line number equal to the
        file's line number. Blank lines are inert in every language the rule
        engine covers.
        """
        if not self.new_side:
            return ""
        last = max(self.new_side)
        return "\n".join(self.new_side.get(n, "") for n in range(1, last + 1))


def _norm_path(path) -> str:
    """Normalise a diff header or an LLM-reported path to one comparable form."""
    text = str(path or "").strip()
    text = text.split("\t")[0].strip()          # `diff -u` timestamp column
    text = text.strip("`").strip('"').strip()   # model backticks / git quoting
    text = text.replace("\\", "/")
    if text.startswith(("a/", "b/")):
        text = text[2:]
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def parse_unified_diff(diff: str) -> list[DiffFile]:
    """Split a unified diff into per-file new-side views.

    Hunk bodies are consumed using the counts in the ``@@`` header rather than
    by prefix-sniffing, because an added line whose own content starts with
    ``++`` renders as ``+++...`` and would otherwise be mistaken for a file
    header. Files deleted by the PR (``+++ /dev/null``) and binary patches
    (which carry no ``+++`` header at all) produce no entry.
    """
    files: list[DiffFile] = []
    current: DiffFile | None = None
    new_line = 0
    old_left = new_left = 0

    for raw in str(diff or "").splitlines():
        if old_left > 0 or new_left > 0:
            # --- inside a hunk body -------------------------------------
            marker = raw[:1]
            if marker == "+":
                if current is not None:
                    current.new_side[new_line] = raw[1:]
                    current.added_lines.add(new_line)
                new_line += 1
                new_left -= 1
                continue
            if marker == "-":
                old_left -= 1
                continue
            if marker == "\\":  # "\ No newline at end of file"
                continue
            if marker in (" ", ""):
                if current is not None:
                    current.new_side[new_line] = raw[1:]
                new_line += 1
                new_left -= 1
                old_left -= 1
                continue
            # Anything else means the hunk ended early (malformed diff); fall
            # through and re-read this line as a header.
            old_left = new_left = 0

        if raw.startswith("@@"):
            match = _HUNK_RE.match(raw)
            if match is None:
                continue
            old_left = int(match.group(2) or 1)
            new_line = int(match.group(3))
            new_left = int(match.group(4) or 1)
            continue

        if raw.startswith("+++ "):
            path = _norm_path(raw[4:])
            current = None if (not path or path == "dev/null") else DiffFile(path=path)
            if current is not None:
                files.append(current)
            continue

        if raw.startswith("diff --git ") or raw.startswith("--- "):
            # A new file section begins; drop the previous one until its +++
            # header names the new-side path.
            if raw.startswith("diff --git "):
                current = None
            continue

    return [f for f in files if f.added_lines]


def is_scannable_path(path: str) -> bool:
    """True when ``path`` is source code worth spending a rule pass on."""
    path = _norm_path(path)
    if not path:
        return False
    name = path.rsplit("/", 1)[-1]
    dot = name.rfind(".")
    if dot <= 0 or name[dot:].lower() not in SCANNABLE_SUFFIXES:
        return False
    if _SKIP_NAME_RE.match(name) or _SKIP_DIR_RE.search(path):
        return False
    return True


# ==========================================================================
# scanning
# ==========================================================================

def _load_scan_rules():
    """Import ``mlscan.rules.scan_rules`` lazily, or None if unavailable.

    Lazy + guarded on purpose: the rule engine is an optional enhancement, so a
    deployment that ships only ``app/`` must still boot and review.
    """
    try:
        from mlscan.rules import scan_rules
    except Exception:  # noqa: BLE001 - ImportError, or a broken module
        logger.warning("mlscan.rules unavailable; static scan disabled",
                       exc_info=True)
        return None
    return scan_rules


def _to_finding(path: str, raw: dict) -> Finding:
    """Convert one ``mlscan.rules`` finding dict into a review ``Finding``."""
    cwe = str(raw.get("cwe") or "").strip()
    name = str(raw.get("name") or "Potential vulnerability").strip()
    label = f" ({cwe})" if cwe else ""
    return Finding(
        file=path,
        line=raw.get("line"),
        category=Category.security,
        severity=_SEVERITY_BY_CWE.get(cwe, _DEFAULT_SEVERITY),
        message=f"{name} — this line matches a pattern that is unsafe by "
                f"definition{label}.",
        suggestion=_REMEDIATION.get(cwe, ""),
        source=SOURCE_RULES,
        cwe=cwe or None,
        rule_id=str(raw.get("rule_id") or "") or None,
        evidence=str(raw.get("evidence") or ""),
    )


_SEVERITY_RANK = {
    Severity.critical: 0,
    Severity.major: 1,
    Severity.minor: 2,
    Severity.info: 3,
}


def scan_diff(diff: str, *, max_files: int = MAX_SCANNED_FILES,
              max_chars: int = MAX_FILE_CHARS, scanner=None) -> list[Finding]:
    """Run the deterministic rules over the lines a PR diff adds.

    Returns review-engine ``Finding`` objects sorted most-severe first. Never
    raises: an unusable diff, a missing ``mlscan`` or a rule that blows up all
    resolve to an empty list, because the Gemini review must proceed either way.

    ``scanner`` (a ``scan_rules``-shaped callable) is injectable for tests.
    """
    scan_rules = scanner or _load_scan_rules()
    if scan_rules is None:
        return []

    try:
        diff_files = parse_unified_diff(diff)
    except Exception:  # noqa: BLE001 - a malformed diff must not break review
        logger.exception("Could not parse diff; skipping static scan")
        return []

    findings: list[Finding] = []
    scanned = 0
    for diff_file in diff_files:
        if scanned >= max_files:
            logger.info("Static scan file cap (%d) reached; skipping the rest",
                        max_files)
            break
        if not is_scannable_path(diff_file.path):
            continue
        text = diff_file.reconstruct()
        if not text.strip() or len(text) > max_chars:
            continue
        scanned += 1
        try:
            raw_findings = scan_rules(text)
        except Exception:  # noqa: BLE001 - one bad file must not lose the rest
            logger.exception("Rule scan failed for %s", diff_file.path)
            continue
        for raw in raw_findings:
            line = raw.get("line")
            # No line means we cannot attribute it to the change -> drop it.
            if line is None or line not in diff_file.added_lines:
                continue
            try:
                findings.append(_to_finding(diff_file.path, raw))
            except Exception:  # noqa: BLE001 - malformed rule payload
                logger.exception("Could not convert rule finding %r", raw)

    findings.sort(key=lambda f: (_SEVERITY_RANK.get(f.severity, 9), f.file,
                                 f.line or 0))
    return findings


# ==========================================================================
# merging with the model's findings
# ==========================================================================

def _mentions_cwe(llm: Finding, cwe: str | None) -> bool:
    """True when the LLM's prose is clearly about the same defect class."""
    keywords = _CWE_KEYWORDS.get(cwe or "", ())
    if not keywords:
        return False
    blob = f"{llm.message} {llm.suggestion}".lower()
    return any(word in blob for word in keywords)


def _is_duplicate(rule: Finding, llm: Finding) -> bool:
    """Decide whether an LLM finding restates a rule finding.

    Same file, the model called it a security issue, and then one of two tests.
    When the model gave a line number it is trusted to within
    :data:`LINE_MATCH_WINDOW` and nothing else -- a second SQL injection fifty
    lines down is a second finding, not a restatement. Only when the line is
    missing entirely do we fall back to matching the prose against the defect
    class.

    Non-security LLM findings are never suppressed: a real "fetchone() result is
    not checked for None" can sit on the same line as an SQL-injection rule hit
    and say something completely different.
    """
    if llm.category != Category.security:
        return False
    if _norm_path(llm.file) != _norm_path(rule.file):
        return False
    if llm.line is not None and rule.line is not None:
        return abs(llm.line - rule.line) <= LINE_MATCH_WINDOW
    return _mentions_cwe(llm, rule.cwe)


def merge_findings(rule_findings: list[Finding],
                   llm_findings: list[Finding]) -> list[Finding]:
    """Combine deterministic and model findings into one de-duplicated list.

    When both flag the same thing the *rule* finding survives -- it is the one
    carrying a rule id, an exact line and a source-line quote -- upgraded to
    ``source="llm+rules"`` and keeping the model's suggestion when it has one,
    since that suggestion is written against this specific code. Rule findings
    lead the list because they are reproducible; the model's independent
    findings follow in the order it produced them.
    """
    merged = [f.model_copy(deep=True) for f in rule_findings]
    kept_llm: list[Finding] = []

    for llm in llm_findings:
        match = next((r for r in merged
                      if r.source == SOURCE_RULES and _is_duplicate(r, llm)), None)
        if match is None:
            kept_llm.append(llm)
            continue
        match.source = SOURCE_BOTH
        if llm.suggestion.strip():
            match.suggestion = llm.suggestion
        # Agreement can only raise the stakes, never lower them.
        if _SEVERITY_RANK.get(llm.severity, 9) < _SEVERITY_RANK.get(match.severity, 9):
            match.severity = llm.severity

    return merged + kept_llm


def known_issues_block(rule_findings: list[Finding]) -> list[str]:
    """One-line descriptions of what the static analyzer already found.

    Fed to the prompt so the model spends its output budget on logic and design
    instead of restating a pattern match. This is the first of two de-duplication
    layers; :func:`merge_findings` is the second, because the model does not
    reliably obey the instruction.
    """
    lines = []
    for f in rule_findings:
        loc = f.file + (f":{f.line}" if f.line else "")
        label = f.cwe or "security issue"
        lines.append(f"{loc} — {label} ({f.rule_id or 'static rule'})")
    return lines
