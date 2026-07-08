"""Unit tests for the review engine, using a fake Gemini client (no network)."""

from types import SimpleNamespace

import pytest

from app.review_engine import (
    Category,
    Finding,
    ReviewResult,
    Severity,
    _generate_with_retry,
    build_prompt,
    parse_review,
    review_diff,
)


class FakeModels:
    def __init__(self, text: str):
        self._text = text

    def generate_content(self, model: str, contents: str):
        return SimpleNamespace(text=self._text)


class FakeClient:
    def __init__(self, text: str):
        self.models = FakeModels(text)


SAMPLE_JSON = """{
  "summary": "Adds a divide helper.",
  "findings": [
    {"file": "calc.py", "line": 3, "category": "bug", "severity": "major",
     "message": "No zero-division guard.", "suggestion": "Raise on b == 0."}
  ]
}"""


def test_parse_plain_json():
    result = parse_review(SAMPLE_JSON)
    assert isinstance(result, ReviewResult)
    assert len(result.findings) == 1
    assert result.findings[0].category == "bug"


def test_parse_json_wrapped_in_fences():
    fenced = f"```json\n{SAMPLE_JSON}\n```"
    result = parse_review(fenced)
    assert result.findings[0].severity == "major"


def test_build_prompt_includes_standards_and_diff():
    prompt = build_prompt("my-diff-content", "my-standards")
    assert "my-diff-content" in prompt
    assert "my-standards" in prompt


def test_review_diff_uses_injected_client():
    client = FakeClient(SAMPLE_JSON)
    result = review_diff("some diff", client=client, model="fake", standards="")
    assert result.summary == "Adds a divide helper."
    assert result.findings[0].file == "calc.py"


def test_empty_diff_short_circuits():
    client = FakeClient("SHOULD NOT BE CALLED")
    result = review_diff("   ", client=client, model="fake", standards="")
    assert result.findings == []


class FlakyModels:
    """Raises a transient error N times, then succeeds."""

    def __init__(self, fail_times: int, status: int, text: str):
        self.calls = 0
        self._fail_times = fail_times
        self._status = status
        self._text = text

    def generate_content(self, model, contents):
        self.calls += 1
        if self.calls <= self._fail_times:
            exc = RuntimeError("transient")
            exc.code = self._status
            raise exc
        return SimpleNamespace(text=self._text)


class FlakyClient:
    def __init__(self, models):
        self.models = models


def test_retry_recovers_from_transient_503():
    models = FlakyModels(fail_times=2, status=503, text=SAMPLE_JSON)
    text = _generate_with_retry(FlakyClient(models), "m", "prompt",
                                sleep=lambda _: None)
    assert models.calls == 3  # 2 failures + 1 success
    assert "summary" in text


def test_retry_gives_up_after_max_and_reraises():
    models = FlakyModels(fail_times=99, status=503, text=SAMPLE_JSON)
    with pytest.raises(RuntimeError):
        _generate_with_retry(FlakyClient(models), "m", "prompt", retries=3,
                             sleep=lambda _: None)
    assert models.calls == 3


def test_non_retryable_error_raises_immediately():
    models = FlakyModels(fail_times=99, status=400, text=SAMPLE_JSON)
    with pytest.raises(RuntimeError):
        _generate_with_retry(FlakyClient(models), "m", "prompt",
                             sleep=lambda _: None)
    assert models.calls == 1  # 400 is not retried


def test_unknown_category_is_coerced_not_crashed():
    # Regression: Gemini once returned category "error-handling", which is not in
    # our enum, and strict validation crashed the whole review.
    f = Finding(file="x.py", category="error-handling", severity="high",
                message="m")
    assert f.category == Category.bug        # synonym mapped
    assert f.severity == Severity.major      # "high" -> major


def test_unrecognized_values_fall_back_to_defaults():
    f = Finding(file="x.py", category="cosmic-rays", severity="apocalyptic",
                message="m")
    assert f.category == Category.refactor   # default bucket
    assert f.severity == Severity.minor      # default severity


def test_line_coercion_handles_strings_and_ranges():
    assert Finding(file="x", category="bug", severity="minor",
                   message="m", line="42").line == 42
    assert Finding(file="x", category="bug", severity="minor",
                   message="m", line="10-14").line == 10
    assert Finding(file="x", category="bug", severity="minor",
                   message="m", line=0).line is None
