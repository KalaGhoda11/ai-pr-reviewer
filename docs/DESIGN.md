# Design & Testing Document — AI PR Reviewer

MSSE Capstone deliverable. This document explains (1) the architecture and
design decisions, including the software and architectural patterns used, and
(2) the software testing implemented.

---

## 1. Overview

**AI PR Reviewer** is an AI system that automatically reviews GitHub pull
requests. When a PR is opened or updated, GitHub sends a webhook to the service;
the service fetches the PR's diff, asks a Large Language Model (Google Gemini)
to review it against an organization coding-standards document, and posts the
findings back as a comment on the PR.

It is an **AI system built with AI engineering techniques** (prompt design,
structured LLM output, retrieval of domain knowledge) wrapped in a
production-shaped web service (webhook ingestion, signature verification,
asynchronous processing, CI/CD, tests).

---

## 2. Requirements

### Functional
| ID | Requirement |
|----|-------------|
| F1 | Receive GitHub `pull_request` webhook events. |
| F2 | Reject any webhook whose HMAC signature is invalid. |
| F3 | Fetch the unified diff for the pull request. |
| F4 | Produce a review (bugs, security, refactor, style) using an LLM. |
| F5 | Guide the review with an organization coding-standards document. |
| F6 | Post the review back to the PR as a comment. |
| F7 | Expose a health endpoint for the host and monitoring. |

### Non-functional
| ID | Requirement |
|----|-------------|
| N1 | Respond to the webhook within seconds (offload slow LLM work). |
| N2 | Keep all secrets out of source control. |
| N3 | Be testable without network access or live API keys. |
| N4 | Deploy to a free-tier host (Render). |
| N5 | Every change validated by CI before merge. |

---

## 3. Architecture

```
              GitHub
                │  pull_request event (webhook, HMAC-signed)
                ▼
     ┌───────────────────────┐
     │  FastAPI  (app/main)   │
     │   /health   /webhook   │
     └───────────┬───────────┘
                 │ verify signature (github_client.verify_signature)
                 │ 202 "queued"  ──► returns fast (N1)
                 ▼  (BackgroundTask)
     ┌───────────────────────┐        ┌─────────────────────────┐
     │  webhook.process_pr    │───────►│ github_client.fetch_diff│──► GitHub API
     └───────────┬───────────┘        └─────────────────────────┘
                 ▼
     ┌───────────────────────┐        ┌─────────────────────────┐
     │  review_engine         │        │ standards/               │
     │  build_prompt          │◄───────│ coding_standards.md      │ (RAG-lite)
     │  review_diff (Gemini)  │───────►│ Google Gemini API        │
     │  parse_review          │        └─────────────────────────┘
     │  format_comment        │
     └───────────┬───────────┘
                 ▼
     github_client.post_review_comment ──► GitHub API (PR comment)
```

### Module responsibilities (separation of concerns)
| Module | Responsibility |
|--------|----------------|
| `app/main.py` | App assembly, ops endpoints, router wiring. |
| `app/config.py` | Load & validate configuration from environment. |
| `app/webhook.py` | HTTP boundary: verify, filter, orchestrate, respond. |
| `app/review_engine.py` | AI logic: prompt, call LLM, parse, format. Pure/side-effect-free except the injected client. |
| `app/github_client.py` | All GitHub I/O + signature verification. |
| `standards/coding_standards.md` | Domain knowledge injected into the prompt. |

---

## 4. Design decisions & patterns

### D1 — Layered architecture (HTTP → orchestration → domain → I/O)
The HTTP boundary (`webhook.py`) never talks to Gemini or GitHub directly beyond
orchestration; the AI logic (`review_engine.py`) never imports FastAPI. This
keeps the AI core reusable (e.g. from a CLI) and independently testable.

### D2 — Dependency Injection for the LLM client
`review_diff(diff, client, model, standards)` receives its Gemini client as an
argument instead of constructing one. **Why:** unit tests inject a fake client
and run with zero network/API-key dependency (N3). This is the single most
important testability decision in the project.

### D3 — Structured LLM output (contract-first AI)
The model is instructed to return JSON matching a Pydantic schema
(`ReviewResult` / `Finding`), which is then validated. **Why:** turns a
free-text LLM into a reliable component whose output can be programmatically
rendered into comments. `_extract_json` tolerates markdown fences the model
sometimes adds.

### D4 — RAG-lite over fine-tuning
Organization-specific practices come from injecting `coding_standards.md` into
the prompt, not from a fine-tuned model. **Why:** fine-tuning is expensive and
slow to iterate; prompt-injected standards give the same "org-aware" behavior at
zero cost and can be edited in seconds. (Documented as a deliberate trade-off; a
vector-store RAG is listed as future work for large rule sets.)

### D4b — Producer/Consumer via Background Tasks
The webhook returns immediately and the review runs in a FastAPI
`BackgroundTask`. **Why (N1):** Gemini calls take seconds; GitHub expects a
prompt response and will retry/mark failed otherwise.

### D5 — 12-Factor configuration
All secrets and tunables come from environment variables via
`pydantic-settings`, cached with `lru_cache`. `.env` is git-ignored;
`.env.example` documents the contract (N2).

### D6 — Security: constant-time HMAC verification
`verify_signature` uses `hmac.compare_digest` and rejects malformed input,
guarding against timing attacks and forged payloads (F2). This is enforced
before the body is even parsed.

### Pattern summary
- **Architectural:** Layered / Ports-and-Adapters (I/O isolated in `github_client`), Event-Driven (webhook), Producer/Consumer (background task).
- **Design:** Dependency Injection, Factory (`_get_gemini_client`, `get_settings`), Schema/DTO (Pydantic models), Adapter (thin GitHub wrappers).

---

## 5. Technology choices

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Language | Python 3.12 | Strong AI/LLM ecosystem; 3.14 dropped (no prebuilt `pydantic-core` wheel). |
| Web framework | FastAPI | Async, typed, built-in background tasks, auto OpenAPI docs. |
| LLM | Google Gemini (`google-genai`) | Free-tier access; fast `flash` model. |
| GitHub | PyGithub + httpx | Mature API client; httpx for raw-diff media type. |
| Validation/config | Pydantic v2 / pydantic-settings | Schema validation + env config in one stack. |
| Hosting | Render (free tier) | Zero-cost, Git-push deploys, matches rubric. |
| CI | GitHub Actions | Runs the test suite on every push/PR. |
| Tests | pytest | Standard, fixtures, monkeypatch for mocking. |

---

## 6. Testing

### Strategy
Testing focuses on the boundaries and the AI parsing logic — the parts most
likely to break — while keeping the whole suite **hermetic** (no network, no API
keys) so it runs identically on a laptop and in CI.

- **Unit tests** for pure logic: prompt building, JSON extraction/parsing,
  comment formatting.
- **Boundary tests** for security and routing: HMAC verification, event
  filtering, background-task dispatch.
- **Mocking:** a `FakeClient` stands in for Gemini (D2); `monkeypatch` replaces
  `process_pull_request` and sets env secrets. No real GitHub or Gemini calls.

### Test inventory
| Test file | Test | Verifies |
|-----------|------|----------|
| `test_health.py` | `test_health_ok` | `/health` returns 200 + version (F7). |
| | `test_root` | Root metadata endpoint. |
| `test_review_engine.py` | `test_parse_plain_json` | JSON → `ReviewResult` (D3). |
| | `test_parse_json_wrapped_in_fences` | Tolerates ```` ```json ```` fences. |
| | `test_build_prompt_includes_standards_and_diff` | Standards + diff in prompt (F5). |
| | `test_review_diff_uses_injected_client` | DI path end-to-end with fake client (D2). |
| | `test_empty_diff_short_circuits` | No LLM call on empty diff. |
| `test_webhook.py` | `test_verify_signature_valid_and_invalid` | HMAC accept/reject (F2, D6). |
| | `test_webhook_rejects_bad_signature` | 401 on bad signature. |
| | `test_webhook_ping` | `ping` → `pong`. |
| | `test_webhook_ignores_non_pr_event` | Non-PR events ignored. |
| | `test_webhook_queues_review_on_pr_opened` | PR opened dispatches review (F1). |

**Result:** 12 tests, all passing (`pytest -q`).

### Running
```bash
pytest -q
```
CI (`.github/workflows/ci.yml`) runs the same command on every push and PR to
`main`; a red suite blocks the change (N5).

### Test gaps / future work
- No live integration test against real GitHub + Gemini (manual E2E during
  demo instead; would need recorded cassettes to automate).
- No load/performance testing (out of scope for capstone MVP).

---

## 7. Traceability (requirement → where it's met)

| Req | Implemented in | Tested by |
|-----|----------------|-----------|
| F1 | `webhook.github_webhook` | `test_webhook_queues_review_on_pr_opened` |
| F2 | `github_client.verify_signature` | `test_verify_signature_*`, `test_webhook_rejects_bad_signature` |
| F3 | `github_client.fetch_diff` | (manual E2E) |
| F4 | `review_engine.review_diff` | `test_review_diff_uses_injected_client` |
| F5 | `review_engine.build_prompt` + `load_standards` | `test_build_prompt_includes_standards_and_diff` |
| F6 | `github_client.post_review_comment` + `format_comment` | (manual E2E) |
| F7 | `main.health` | `test_health_ok` |

---

## 8. Future work
- Vector-store RAG (pgvector/Chroma) for large standards sets.
- Inline, line-level review comments via the GitHub review API.
- Severity gating (e.g. fail a status check on `critical` findings).
- Recorded-cassette integration tests for automated E2E.
