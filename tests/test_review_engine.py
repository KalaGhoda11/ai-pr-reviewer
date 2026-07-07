"""Unit tests for the review engine, using a fake Gemini client (no network)."""

from types import SimpleNamespace

from app.review_engine import (
    ReviewResult,
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
