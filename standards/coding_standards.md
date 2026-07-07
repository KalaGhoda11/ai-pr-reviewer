# Organization Coding Standards

These rules are injected into the review prompt so the AI reviewer flags
org-specific practices, not just generic issues. This is the "RAG-lite"
knowledge source (no fine-tuning required).

## General
- Functions should do one thing; keep them under ~40 lines.
- No commented-out code in merged PRs.
- Public functions and modules must have docstrings.

## Security
- Never log secrets, tokens, or full request bodies.
- All external input must be validated before use.
- Use parameterized queries; never string-format SQL.
- Verify webhook signatures before trusting a payload.

## Error handling
- Catch specific exceptions, not bare `except:`.
- Fail loudly at startup on missing required config.

## Python-specific
- Type-hint public function signatures.
- Prefer `pathlib` over `os.path`.
- Use f-strings, not `%` or `.format()`.
