"""The review engine: turn a unified diff into structured review findings.

Design notes
------------
- The Gemini client is injected (see ``review_diff``'s ``client`` param) so unit
  tests can pass a fake and never hit the network.
- The model is asked to return strict JSON matching ``ReviewResult`` so the
  output is machine-usable (we post each finding as a comment).
- ``coding_standards`` is injected into the prompt — this is the "RAG-lite"
  knowledge source that gives org-specific reviews without fine-tuning.
- A ``Finding`` carries its ``source``: the model, the deterministic rule engine
  in :mod:`app.security_scan`, or both. Every provenance field is optional and
  defaulted, so parsing a plain Gemini JSON response is unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Transient upstream statuses worth retrying (rate limit / overloaded / gateway).
_RETRYABLE_STATUS = (429, 500, 502, 503, 504)

STANDARDS_PATH = Path(__file__).resolve().parent.parent / "standards" / "coding_standards.md"


class Severity(str, Enum):
    info = "info"
    minor = "minor"
    major = "major"
    critical = "critical"


class Category(str, Enum):
    bug = "bug"
    security = "security"
    refactor = "refactor"
    style = "style"


# LLMs don't always emit our exact enum values, so we normalize common synonyms
# and fall back to a safe bucket instead of crashing the whole review.
_CATEGORY_SYNONYMS = {
    "error-handling": Category.bug,
    "errorhandling": Category.bug,
    "logic": Category.bug,
    "correctness": Category.bug,
    "vulnerability": Category.security,
    "security-vulnerability": Category.security,
    "performance": Category.refactor,
    "maintainability": Category.refactor,
    "readability": Category.style,
    "formatting": Category.style,
    "convention": Category.style,
}
_SEVERITY_SYNONYMS = {
    "high": Severity.major,
    "medium": Severity.minor,
    "moderate": Severity.minor,
    "low": Severity.minor,
    "warning": Severity.minor,
    "blocker": Severity.critical,
    "note": Severity.info,
    "nit": Severity.info,
}


# Where a finding came from. The rule engine's findings are reproducible and
# carry a rule id; the model's are not. A reviewer is entitled to know which is
# which, so provenance travels with the finding instead of being flattened away.
SOURCE_LLM = "llm"
SOURCE_RULES = "rules"
SOURCE_BOTH = "llm+rules"


class Finding(BaseModel):
    file: str = Field(description="Path of the file the finding refers to.")
    line: int | None = Field(default=None, description="1-based line number, if known.")
    category: Category
    severity: Severity
    message: str = Field(description="What is wrong.")
    suggestion: str = Field(default="", description="How to fix it.")

    # Provenance. Defaulted so Gemini's JSON (which never sets them) validates
    # exactly as before; the static scanner fills them in.
    source: str = Field(default=SOURCE_LLM, description="llm | rules | llm+rules")
    cwe: str | None = Field(default=None, description="CWE class, rule findings only.")
    rule_id: str | None = Field(default=None, description="Rule that fired, if any.")
    evidence: str = Field(default="", description="The offending source line.")

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, v):
        return _coerce_enum(v, Category, _CATEGORY_SYNONYMS, Category.refactor)

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, v):
        return _coerce_enum(v, Severity, _SEVERITY_SYNONYMS, Severity.minor)

    @field_validator("line", mode="before")
    @classmethod
    def _coerce_line(cls, v):
        """Models sometimes send line as a string, a range, or 0/None."""
        if v in (None, "", 0, "0"):
            return None
        try:
            return int(str(v).split("-")[0].strip())
        except (ValueError, TypeError):
            return None


def _coerce_enum(value, enum_cls, synonyms, default):
    """Map a raw model value onto ``enum_cls``, via synonyms, else ``default``."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        key = value.strip().lower().replace(" ", "-").replace("_", "-")
        if key in enum_cls._value2member_map_:
            return key
        if key in synonyms:
            return synonyms[key]
    return default


class ReviewResult(BaseModel):
    summary: str = Field(description="One-paragraph overview of the change.")
    findings: list[Finding] = Field(default_factory=list)


def load_standards(path: Path = STANDARDS_PATH) -> str:
    """Load the coding-standards doc, returning empty string if absent."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def build_prompt(diff: str, standards: str,
                 known_issues: list[str] | None = None) -> str:
    """Assemble the review prompt from the diff, standards and prior findings.

    ``known_issues`` lists what the deterministic scanner already caught. Telling
    the model up front stops it spending its output budget restating a pattern
    match, and leaves it on the logic and design problems a regex cannot see.
    """
    known_block = ""
    if known_issues:
        listed = "\n".join(f"- {item}" for item in known_issues)
        known_block = f"""
A static analyzer has ALREADY reported these issues, and they are already in the
review comment:
{listed}
Do NOT repeat them. Report only what the analyzer cannot see (logic errors,
design problems, missing checks, standards violations).
"""

    return f"""You are a senior software engineer reviewing a pull request.
{known_block}

Review ONLY the changes in the unified diff below. Focus on:
- bugs and logic errors
- security vulnerabilities
- refactoring opportunities
- violations of the organization coding standards provided

Organization coding standards:
---
{standards or "(none provided)"}
---

Unified diff to review:
```diff
{diff}
```

Respond with ONLY a JSON object (no markdown fences) matching this schema:
{{
  "summary": "one paragraph",
  "findings": [
    {{
      "file": "path/to/file",
      "line": 42,
      "category": "bug|security|refactor|style",
      "severity": "info|minor|major|critical",
      "message": "what is wrong",
      "suggestion": "how to fix it"
    }}
  ]
}}

Rules:
- "category" MUST be EXACTLY one of: bug, security, refactor, style. Map any
  other kind of issue to the closest of these four.
- "severity" MUST be EXACTLY one of: info, minor, major, critical.
- "line" must be a single integer or null.
If the change looks good, return an empty findings list."""


def _extract_json(text: str) -> str:
    """Strip markdown fences / prose so json.loads gets a clean object."""
    text = text.strip()
    if text.startswith("```"):
        # remove leading ```json / ``` and trailing ```
        text = text.split("```", 2)[1] if "```" in text else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text


def parse_review(raw_text: str) -> ReviewResult:
    """Parse the model's raw text response into a validated ReviewResult."""
    cleaned = _extract_json(raw_text)
    data = json.loads(cleaned)
    return ReviewResult.model_validate(data)


SEVERITY_EMOJI = {
    Severity.critical: "🔴",
    Severity.major: "🟠",
    Severity.minor: "🟡",
    Severity.info: "🔵",
}


# Rendered next to a finding so the reader can tell a matched pattern from a
# model opinion. LLM-only findings get no badge, keeping the existing look.
SOURCE_BADGE = {
    SOURCE_RULES: "🔎 static analysis",
    SOURCE_BOTH: "🔎🤖 static analysis + model",
}

# Printed once when any rule finding is present. States the limitation plainly
# rather than implying the rules have a measured precision they do not have.
RULES_CAVEAT = (
    "_🔎 findings come from deterministic pattern rules that fire only on "
    "constructs unsafe by definition. They are reproducible and carry a rule "
    "id, but they are not a substitute for review._"
)


def format_comment(result: ReviewResult) -> str:
    """Render a ReviewResult as a Markdown PR comment."""
    lines = ["## 🤖 AI PR Review", "", result.summary, ""]
    if not result.findings:
        lines.append("✅ No issues found. Looks good!")
        return "\n".join(lines)

    lines.append(f"**{len(result.findings)} finding(s):**")
    lines.append("")
    for f in result.findings:
        emoji = SEVERITY_EMOJI.get(f.severity, "•")
        loc = f"`{f.file}`" + (f":{f.line}" if f.line else "")
        badge = SOURCE_BADGE.get(f.source, "")
        rule = f" · `{f.rule_id}`" if f.rule_id else ""
        suffix = f"  ·  {badge}{rule}" if badge else ""
        lines.append(
            f"### {emoji} {f.severity.value.upper()} · {f.category.value} — {loc}{suffix}"
        )
        lines.append(f.message)
        if f.evidence:
            lines.append(f"```\n{f.evidence}\n```")
        if f.suggestion:
            lines.append(f"> **Suggestion:** {f.suggestion}")
        lines.append("")
    if any(f.source in SOURCE_BADGE for f in result.findings):
        lines.append(RULES_CAVEAT)
        lines.append("")
    lines.append("---")
    lines.append("_Generated by AI PR Reviewer._")
    return "\n".join(lines)


def _status_of(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP status code from an SDK exception."""
    return getattr(exc, "code", None) or getattr(exc, "status_code", None)


def _generate_with_retry(client, model: str, prompt: str, *, retries: int = 3,
                         base_delay: float = 2.0, sleep=time.sleep) -> str:
    """Call Gemini, retrying transient errors with exponential backoff.

    ``sleep`` is injectable so tests can retry without real delays.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text
        except Exception as exc:  # noqa: BLE001 - inspect status, then decide
            status = _status_of(exc)
            if status not in _RETRYABLE_STATUS or attempt == retries - 1:
                raise
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            logger.warning("Gemini %s on attempt %d/%d; retrying in %.1fs",
                           status, attempt + 1, retries, delay)
            sleep(delay)
    raise last_exc  # pragma: no cover - loop always returns or raises above


def review_diff(diff: str, client, model: str, standards: str | None = None,
                known_issues: list[str] | None = None) -> ReviewResult:
    """Review a unified diff.

    ``client`` is any object exposing ``models.generate_content(model, contents)``
    (the google-genai Client shape), injected so tests can mock it.
    ``known_issues`` are findings the static scanner already reported, passed
    through so the model does not duplicate them.
    """
    if not diff.strip():
        return ReviewResult(summary="Empty diff; nothing to review.", findings=[])

    if standards is None:
        standards = load_standards()

    prompt = build_prompt(diff, standards, known_issues)
    raw_text = _generate_with_retry(client, model, prompt)
    return parse_review(raw_text)
