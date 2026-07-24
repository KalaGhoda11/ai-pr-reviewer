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

The project contains **two AI components**, deliberately built on opposite
architectures. The reviewer above is *online* and generative: a hosted LLM
called over the network at request time. The second, `mlscan`, is *offline* and
discriminative: a classifier trained here, shipped as a file, with a
deterministic rule engine beside it. Section 7 describes it, how it differs
architecturally, and how its rule half is wired into the live webhook flow.

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
| F8 | Report deterministic, reproducible security findings alongside the LLM's, each carrying its provenance. |

### Non-functional
| ID | Requirement |
|----|-------------|
| N1 | Respond to the webhook within seconds (offload slow LLM work). |
| N2 | Keep all secrets out of source control. |
| N3 | Be testable without network access or live API keys. |
| N4 | Deploy to a free-tier host (Render). |
| N5 | Every change validated by CI before merge. |
| N6 | Add **no new runtime dependency** to the deployed service (512 MB free tier), and never let an optional component break a review. |

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

The deployed flow has since gained one more stage — a deterministic static scan
that runs *before* the Gemini call and feeds it. See §7.2 for that diagram.

### Module responsibilities (separation of concerns)
| Module | Responsibility |
|--------|----------------|
| `app/main.py` | App assembly, ops endpoints, router wiring. |
| `app/config.py` | Load & validate configuration from environment. |
| `app/webhook.py` | HTTP boundary: verify, filter, orchestrate, respond. |
| `app/review_engine.py` | AI logic: prompt, call LLM, parse, format. Pure/side-effect-free except the injected client. |
| `app/github_client.py` | All GitHub I/O + signature verification. |
| `app/security_scan.py` | Diff → new-side reconstruction → deterministic rule findings → merge with the LLM's (§7.2). |
| `standards/coding_standards.md` | Domain knowledge injected into the prompt. |
| `mlscan/` | The standalone offline scanner (§7.1). Only `mlscan/rules.py` is reachable from the deployed app. |

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
- **Added by the second AI component (§7):** Pipeline (diff → per-file view → rules → findings), Adapter (`_to_finding` maps a rule payload onto the shared `Finding` DTO), Strategy (interchangeable detectors: ML / rules / hybrid), Null Object + lazy guarded import for graceful degradation, and Dependency Injection again (`scan_diff(..., scanner=…)`).

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

The 12 tests above are the original web-service core, all passing (`pytest -q`).
The suite has since grown to **241 tests** as the second AI component and its
integration landed; §7.4 gives the breakdown and explains how the ML-only tests
skip cleanly so the deployed-app CI job stays green.

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

## 7. The second AI component — `mlscan`, and its integration

### 7.1 What it is, and how it differs architecturally from the reviewer

`mlscan` is a standalone vulnerability scanner: a **trained ML classifier** plus
a **deterministic rule engine**. It answers a narrower question than the Gemini
reviewer ("does this code contain one of 8 known vulnerability classes, or is it
`safe`?") and it answers it on a completely different architecture.

| | Gemini reviewer (`app/review_engine.py`) | `mlscan` |
|---|---|---|
| Inference | Hosted LLM, called over the network per PR | Local; **no LLM, no API key, no network** |
| Artifact | None — the model lives at the vendor | `vuln_clf_v2.joblib`, **6.8 MB**, committed (v1 was 14.8 MB) |
| Cost / latency | Per-token, seconds, rate-limited | Zero marginal cost, milliseconds |
| Training | Prompting only; nothing is trained here | A full offline pipeline: load → de-duplicate → down-sample → vectorize → 12-candidate sweep → calibrate → tune offsets |
| Determinism | Non-deterministic; same diff can yield different prose | Classifier is deterministic given the artifact; the rules are deterministic *and* auditable |
| Failure mode | Plausible-sounding invention | A miss, or a pattern match on code that is fine |
| Measurability | Hard — no labelled ground truth for "good review" | A held-out test split, so every claim is a number |

Because it is measurable, it is held to numbers. The design decisions that came
out of that measurement are the substance of this component:

**D7 — Two scores, always: all-rows and unseen-only.** 9.1% of the public
dataset's test split is byte-identical to rows in its training split, so a naive
score is largely a memorization measurement. Re-scoring the *original* committed
model proves the size of the effect: macro-F1 **0.6543** on all rows versus
**0.4474** on rows it had never seen — a **0.207 memorization premium**. The fix
is to de-duplicate the **train split only** (3,475 rows dropped); the evaluation
splits are never modified, so the two model generations remain comparable. The
headline is therefore the only apples-to-apples figure: unseen-only macro-F1
**0.4854 vs 0.4474 = +0.038**. All-rows, for continuity: macro-F1 0.5908,
accuracy 0.930, weighted-F1 0.926.

**D8 — Model selection on validation, by cost as well as score.** The winner is a
**calibrated LinearSVC (C=2.0)**, picked from 12 candidates on validation
macro-F1 **0.6093**. It beat the LightGBM *in that same bake-off* (0.1125, same
split and features), and it fits in **83.5 s** against the **5,260.7 s (~88 min)**
the previously-shipped **v1** LightGBM took — a ~60× iteration-speed difference
that made the whole 12-candidate sweep affordable in the first place. Those are
two different LightGBM runs: v1's own 0.5743 is an **11-class** macro-F1 over a
different feature space and is *not* comparable to 0.6093, so it is cited here
only for fit time and artifact size (see `docs/ML_SCANNER.md` §3.1).

**D9 — The hybrid is a recall/precision trade, not an upgrade, and it is
reported as one.** Benchmarked as a binary "is this vulnerable" decision over the
17,121-row test split (`mlscan/model/benchmark.json`):

| Detector | Unseen-only (n=15,564) P / R / F1 | All rows (n=17,121) P / R / F1 |
|---|---|---|
| ML | 0.421 / 0.412 / **0.416** | 0.665 / 0.501 / **0.571** |
| Rules | 0.284 / 0.089 / 0.136 | 0.566 / 0.120 / 0.198 |
| Hybrid | 0.361 / 0.437 / **0.396** | 0.609 / 0.524 / **0.563** |

Adding the rules buys **+0.025 recall** and costs **−0.060 precision** on the
unseen slice, for a **lower F1**. The same direction holds on all rows. So the
hybrid is offered as a policy choice (use it when a miss costs more than a false
alarm), never as "better".

**D10 — The rules earn their place on a different axis: quiet, auditable
determinism.** Fire rate on rows the corpus labels safe, read *by source* rather
than pooled:

| Source | n safe | Rules fire | ML fire |
|---|---:|---:|---:|
| `bigvul` (real C functions) | 14,197 | **0.36%** | 1.51% |
| `enriched_bigvul_primevul` | 1,003 | 1.20% | 9.07% |
| `labeled_dataset` | 455 | 15.82% | 14.07% |

The 15.8% is largely **corpus mislabelling**, not detector error: that sub-corpus
labels `eval(user_input)` and `subprocess.run(user_input, shell=True)` as *safe*.
Which is exactly why the pooled number is not the headline — it measures label
noise as much as precision. On the 666 vulnerable/patched pairs, fire rate on the
*patched* function is 0.069 (rules), 0.230 (ML), 0.297 (hybrid) — an **upper
bound** on false positives, since a patch fixes one defect and many "fixes" in
this corpus keep the dangerous construct.

Two other splits shaped the integration decision in §7.2: the detectors are much
stronger on non-C code (ML precision 0.872 / recall 0.717) than on the C family
(0.500 / 0.352), and per-CWE the hybrid only helps on the injection classes
(CWE-89 0.737 vs 0.727, CWE-79 0.653 vs 0.630) while losing on CWE-502, CWE-94,
CWE-476 and MEMORY-OOB.

**D11 — Do not delete the classes the model is bad at.** Dropping CWE-200 lifts
macro-F1 from 0.5908 to 0.6421; also dropping CWE-20 reaches 0.6896 (+0.0988, a
16.7% relative jump). Those two figures are arithmetic — the per-class F1s
re-averaged over the smaller class set, not separate archived runs. This is
**essentially 100% a denominator effect** — a separate (unarchived)
`--drop-weak-classes` retrain moved every surviving class by at most +0.0135. All 9 classes are therefore kept as the
default and the headline. A `--drop-weak-classes` flag exists, defaults to
**off**, and any use must be disclosed as a coverage reduction rather than an
improvement.

Both weak classes have a diagnosed, structural cause rather than a tuning one:

- **CWE-200 is unlearnable from a snippet, with proof.** 66% of its 95 test rows
  appear verbatim in the original training split — yet recall on those memorized
  rows is only **0.14** (compare CWE-502 0.95, CWE-89 0.87, CWE-79 0.83). A class
  the model cannot get right *by rote* has no signal in the representation.
  Cause: 824 of 870 CWE-200 rows come from the two BigVul-derived sources
  (406 `bigvul` + 418 `enriched_bigvul_primevul`) and are C/C++ kernel/curl code
  (822 C, 2 C++), where the information leak is a **missing `memset`** visible
  only in the commit diff, not in the function body. You cannot detect an absence
  from a snippet.
- **CWE-20 is a mislabelled catch-all.** 19.7% of CWE-20 rows have a
  byte-identical twin filed under a *different* label (CWE-89 356, CWE-79 103,
  MEMORY-OOB 92, CWE-94 44, safe 17). Identical C# code containing
  `"SELECT * FROM Users WHERE Username = '" + username + "'"` is CWE-20 in one
  sub-corpus and CWE-89 in another. A share of its error is mathematically
  irreducible.

**D12 — A measured negative result, reported.** Widening the per-class
decision-offset search grid is a **no-op**: offset-max 2.0 / 3.0 / 6.0 / 12.0 all
return the *identical* offset vector. Lowering the step to 0.05 measured +0.005
on validation and **−0.010 on test** — it fits the tuning split. No gain was
available, so the model was **not** re-tuned and the test split was **not**
looked at a second time. Negative results are results.

Full metrics and reproduction commands live in
[ML_SCANNER.md](ML_SCANNER.md); the artifacts of record are
`mlscan/model/metrics_v2.json` (model), `benchmark.json` (detectors) and
`baseline_honest.json` (v1 re-score).

### 7.2 Integration into the deployed webhook flow

Only the **rule half** is wired into the live reviewer. `app/security_scan.py`
runs it over the diff before Gemini is called:

```
   fetch PR diff
        │
        ▼  stage = "static_scan"   (own try/except; never fatal)
   ┌────────────────────────────────────────────────┐
   │ security_scan.scan_diff                        │
   │   parse_unified_diff  → per-file new-side view │
   │   is_scannable_path   → source files only      │
   │   mlscan.rules.scan_rules (lazy import)        │──► deterministic Findings
   │   keep only lines the PR ADDED                 │
   └───────────────┬────────────────────────────────┘
                   │ known_issues_block(...)      ← "already reported, do not repeat"
                   ▼  stage = "gemini_review"
             review_engine.review_diff(diff, known_issues=…)  ──► Gemini
                   │
                   ▼  stage = "merge_findings"
             security_scan.merge_findings(rule_findings, llm_findings)
                   │   source = rules | llm | llm+rules
                   ▼
             format_comment ──► one PR comment
```

**D13 — Zero new dependencies, by construction (N6).** `mlscan/rules.py` is
**pure stdlib** — 1,684 lines that import nothing beyond `ast`, `re`, `bisect`,
`textwrap`, `collections` and `dataclasses`, plus `mlscan.labels`, which imports
nothing at all. `app/security_scan.py` imports *only*
that module — never `mlscan.scanner`, `inference`, `features` or
`security_features`, which pull scikit-learn, numpy and scipy (measured: **+167 MB
RSS** for a single `scan()`, which does not fit the 512 MB free-tier instance).
`requirements.txt` is therefore **unchanged**, and it is verified that no
`sklearn`/`numpy`/`pandas`/`scipy` is importable through `app/main.py`. The
deployed service gets a second opinion for no deployment cost at all.

**D14 — Reconstruct the new-side file; never scan the raw diff.** A unified diff
also contains **deleted** lines, so a PR that *removes* `yaml.load` would be
reported as introducing CWE-502; diff line numbers are offsets into the diff, not
into any file; a multi-file diff cannot attribute a finding to a file; and a diff
does not parse, so the precise AST rules would silently never fire. Instead each
file's new-side image is rebuilt from the hunks — context *and* added lines placed
at their true new-file line numbers, gaps blanked — and scanned per file. Hunk
bodies are consumed using the counts in the `@@` header rather than by
prefix-sniffing, because an added line whose own text starts with `++` renders as
`+++…` and would otherwise be mistaken for a file header.

**D15 — Report only on lines the PR added.** Context lines are *scanned* (they are
what makes the fragment parse, and what lets the SQL flow analysis connect
`sql = "…" + uid` to `execute(sql)`), but a finding that lands on one is
pre-existing debt and would be re-posted on every `synchronize` push. Scope is
further bounded by suffix allow-list, a test/fixture/vendor path skip, 20 files
and 60,000 chars per file (~240 ms on 0.1 CPU, the knee of the curve).

**D16 — Severity from the defect class, never from the rule's confidence.** Those
confidences are hardcoded literals at each call site in `mlscan.rules`, not
calibrated probabilities; printing them in a PR comment would be exactly the kind
of unbacked precision claim this project is trying to avoid. Severity is mapped
from the CWE (injection/deserialization → critical, XSS/OOB → major, NULL-deref /
info-exposure / CWE-20 → minor).

**D17 — "Already reported, do not repeat" prompt-passing.** `known_issues_block`
turns each rule finding into one line (`path:line — CWE — rule_id`), which
`build_prompt` inserts *above* the diff with an instruction to report only what
the analyzer cannot see. **Why:** it stops the LLM spending its output budget
restating pattern matches and points it at logic, design and standards issues.
Because a model does not reliably obey that instruction, `merge_findings` is a
**second** de-duplication layer: an LLM security finding within
`LINE_MATCH_WINDOW = 3` lines of a rule finding in the same file (or, when the
model gave no line, whose prose matches that CWE's keywords) collapses into the
rule finding, which is upgraded to `source = "llm+rules"` and keeps the model's
suggestion. Rule findings lead the comment because they are reproducible, and
carry provenance: `rule_id`, exact line, and the offending source line as
evidence. Non-security LLM findings are never suppressed.

**D18 — Isolation and graceful degradation (N6).** The scan is optional at three
levels: the `mlscan.rules` import is **lazy and guarded**, so a container that
ships only `app/` logs a warning and reviews Gemini-only; `scan_diff` never
raises (a malformed diff, a per-file rule crash, or a malformed rule payload each
resolve to "skip this and continue"); and `process_pull_request` wraps the whole
`static_scan` stage in its own `except` that falls back to `rule_findings = []`.
The Gemini review and the comment post are byte-identical whether the scan works
or not.

### 7.3 What was deliberately NOT integrated

The **classifier** stays out of the deployed service. Three reasons, in order of
weight: the dependency and memory cost above (N6); its honest unseen-only score
(0.4854 macro-F1, binary F1 0.416) is useful for *prioritization*, not for
asserting a defect in someone's PR; and its confidences are per-snippet
probabilities that would read as authority they have not earned. The classifier
is a CLI/CI tool (`python -m mlscan …`, exit code 1 when flagged); the reviewer
ships only the half that is deterministic and auditable.

### 7.4 Testing the second component and its integration

**241 tests, up from 12**, all passing:

| Test file | Tests | Needs ML extras | Covers |
|---|---:|:---:|---|
| `test_rules.py` | 103 | no | Every rule: fires on the unsafe form, stays quiet on the paired safe form. |
| `test_security_scan.py` | 39 | no | Diff parsing, new-side reconstruction, added-line filtering, path filters, severity mapping, merge/de-duplication, failure isolation. |
| `test_security_features.py` | 35 | **yes** | Each hand-crafted indicator column, plus its safe counter-example. |
| `test_benchmark.py` | 27 | 2 of 27 | The pure-stdlib metric layer (precision/recall/F1, grouping, sampling). |
| `test_review_engine.py` | 11 | no | Prompt assembly incl. the `known_issues` block, JSON extraction, formatting. |
| `test_mlscan.py` | 10 | **yes** | Taxonomy (8 + `safe`), probability distribution, end-to-end `scan()`. |
| `test_data_dedup.py` | 9 | **yes** | Train-only de-duplication invariants on a synthetic corpus. |
| `test_webhook.py` | 5 | no | HMAC accept/reject, event filtering, background dispatch. |
| `test_health.py` | 2 | no | `/health`, root metadata. |

**D19 — The ML tests skip; they do not fail, and they do not break collection.**
CI installs `requirements.txt` only — no scikit-learn, numpy, pandas or scipy — so
**56** of the 241 tests cannot run there and the remaining **185** must still
gate the merge. Two details make that work:

1. Every module that needs an ML extra opens with `pytest.importorskip(...)`
   **before any third-party import**, and imports the module under test after it
   (`# noqa: E402`). A module-level `import numpy` placed above the guard fails at
   **collection** time, which *errors the whole run* instead of skipping one file
   — the guard order is the point, not decoration.
2. `test_benchmark.py` keeps its guards **inside** the two test bodies that need
   them, so the other 25 metric tests still run in the dependency-free job.
   `test_mlscan.py` adds a `skipif` for a **missing model artifact**, so a
   checkout without the trained `.joblib` skips rather than errors.

`test_security_scan.py` deliberately uses **no** `importorskip`: the integration
is part of the deployed app, so all 39 of its tests must run in the same job that
gates the deployment. It injects a fake `scan_rules` where it needs one, mirroring
D2's dependency-injection approach for Gemini.

**Consequence:** the deployed-app CI job stays green on `requirements.txt` alone,
while a developer with `requirements-ml.txt` installed runs the full 241.

---

## 8. Traceability (requirement → where it's met)

| Req | Implemented in | Tested by |
|-----|----------------|-----------|
| F1 | `webhook.github_webhook` | `test_webhook_queues_review_on_pr_opened` |
| F2 | `github_client.verify_signature` | `test_verify_signature_*`, `test_webhook_rejects_bad_signature` |
| F3 | `github_client.fetch_diff` | (manual E2E) |
| F4 | `review_engine.review_diff` | `test_review_diff_uses_injected_client` |
| F5 | `review_engine.build_prompt` + `load_standards` | `test_build_prompt_includes_standards_and_diff` |
| F6 | `github_client.post_review_comment` + `format_comment` | (manual E2E) |
| F7 | `main.health` | `test_health_ok` |
| F8 | `security_scan.scan_diff` + `merge_findings` (+ `mlscan/rules.py`) | `test_security_scan.py` (39), `test_rules.py` (103) |
| N6 | `security_scan._load_scan_rules` (lazy/guarded), `webhook.process_pull_request` `static_scan` stage | `test_security_scan.py` isolation tests; CI installs `requirements.txt` only |

---

## 9. Future work
- Vector-store RAG (pgvector/Chroma) for large standards sets.
- Inline, line-level review comments via the GitHub review API.
- Severity gating (e.g. fail a status check on `critical` findings).
- Recorded-cassette integration tests for automated E2E.
- For `mlscan`: fine-tune a code transformer (CodeBERT/GraphCodeBERT) instead of
  TF-IDF, which reads code as text rather than as code — needs a GPU, so it is
  out of scope for this free-tier, offline-by-design module. A cleaner corpus
  would matter more than a bigger model: CWE-20's labels contradict themselves
  (§7.1) and CWE-200's signal is not in the snippet at all.
