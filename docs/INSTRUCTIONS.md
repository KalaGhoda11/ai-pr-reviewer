# AI PR Reviewer — Project Handoff & Instructions

A complete guide to what this project is, how it works, how to run and demo it,
and what's left. Written so someone new can pick it up without prior context.

---

## 1. What it is (in one line)

A GitHub bot that automatically reviews pull requests: when a PR is opened, it
reads the code changes, asks **Google Gemini** to review them for bugs, security
issues, and style problems (guided by a coding-standards file), and posts the
findings back as a comment on the PR.

Built as an **MSSE Capstone** project — a deployed, tested, CI/CD-backed AI system.

## 2. Live links

| Thing | Link |
|---|---|
| **Live service** | https://ai-pr-reviewer-sxdp.onrender.com |
| Health check | https://ai-pr-reviewer-sxdp.onrender.com/health |
| API docs (Swagger) | https://ai-pr-reviewer-sxdp.onrender.com/docs |
| GitHub repo | https://github.com/KalaGhoda11/ai-pr-reviewer |
| Task board (Trello) | https://trello.com/b/fHxg2MVA |
| Example review output | PR #1 in the repo (12 findings posted by the bot) |

## 3. How it works

```
GitHub PR opened
      │  webhook (HMAC-signed) →  POST /webhook
      ▼
FastAPI app on Render
      │  1. verify signature (401 if bad)
      │  2. return 200 fast, do the rest in a background task
      ▼
  fetch the PR diff   →  GitHub REST API (api.github.com/.../pulls/N, diff media type)
      ▼
  review the diff     →  Gemini 2.5-flash  (+ standards/coding_standards.md)
      ▼
  post the comment    →  GitHub API (PR conversation comment)
```

Key files:
- `app/main.py` — app entry, `/health` + `/` endpoints
- `app/webhook.py` — the `/webhook` endpoint + background review task
- `app/review_engine.py` — builds the prompt, calls Gemini, parses/validates the
  JSON, formats the Markdown comment (with retry + output-coercion hardening)
- `app/github_client.py` — signature verification, diff fetch, comment posting
- `app/config.py` — reads all secrets from environment variables
- `standards/coding_standards.md` — the org rules injected into the prompt
- `docs/DESIGN.md` — architecture & testing document (the graded design doc)

## 4. How to see it work (the demo)

1. **Wake the service first** (Render free tier sleeps after ~15 min idle; first
   request takes ~50s): open https://ai-pr-reviewer-sxdp.onrender.com/health and
   wait for `{"status":"ok"}`.
2. In the repo, create a branch, add or edit a `.py` file with some obvious
   problems (or reuse `examples/payment.py`), and **open a pull request**.
3. Within ~10–30s, the **AI PR Reviewer** posts a review comment on the PR with
   findings tagged by severity (🔴 critical → 🔵 info).

That flow — open PR → comment appears — is the demo video.

## 5. Run it locally

Requires **Python 3.12** (3.14 does not have prebuilt wheels for a dependency).

```bash
py -3.12 -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env          # then fill in the values (see section 6)
uvicorn app.main:app --reload
```
Open http://localhost:8000/health and http://localhost:8000/docs

Run the tests:
```bash
pytest -q            # 18 tests, all mocked — no network or API keys needed
```

## 6. Configuration (secrets)

All secrets are **environment variables** — nothing sensitive is in the repo.
`.env.example` documents them; the real values live in **Render → the service →
Environment** (encrypted), not in git.

| Variable | What it is |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (from https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | `gemini-2.5-flash` (2.0-flash has **zero** free-tier quota — do not use) |
| `GITHUB_TOKEN` | Fine-grained token, scoped to this repo: Contents=Read, Pull requests=Read+Write. Used to fetch the diff and post the comment. |
| `GITHUB_WEBHOOK_SECRET` | Shared secret; must match the value in the GitHub webhook settings |

## 7. Deployment & maintenance (Render)

- Hosting is **Render free tier**, configured by `render.yaml` (infrastructure as
  code). The service auto-deploys when you push to `main`.
- **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Health check path:** `/health`
- **Free-tier caveat:** the instance sleeps after inactivity; the first request
  after a nap is slow (~50s cold start). For a live demo, hit `/health` first to
  wake it.
- To change the model or any secret: Render → service → **Environment** → edit →
  Save (this triggers a redeploy).

## 8. The GitHub webhook

Registered at: repo → **Settings → Webhooks**.
- Payload URL: `https://ai-pr-reviewer-sxdp.onrender.com/webhook`
- Content type: `application/json`
- Secret: must equal `GITHUB_WEBHOOK_SECRET` in Render
- Events: **Pull requests** only
- A green check under "Recent Deliveries" (200) means GitHub can reach the service.

## 9. What's left for the capstone

- [ ] **Record the demo video** (open a PR, show the review posting).
- [ ] **Give `quantic-grader` access** — either add it as a read collaborator
      (repo → Settings → Collaborators) or make the repo public.
- [ ] (Optional) Add a link to the deployed service + demo video in the README.

## 10. Security notes — ROTATE THESE

Several credentials were shared in plain text during setup. Before final
submission (especially if the repo is made public), **regenerate/revoke**:
- The **Gemini API key** (`AQ.Ab8RN6…`) — https://aistudio.google.com/apikey
- The **`GITHUB_TOKEN`** fine-grained token — https://github.com/settings/tokens?type=beta
- Any leftover **classic PAT** used for the initial push (`capstone-push`) —
  https://github.com/settings/tokens
After regenerating, update the new values in **Render → Environment**.
The repo itself was verified credential-free (all secrets are env vars only).

## 11. Notes / history

The end-to-end live test surfaced and fixed two real bugs (documented in commits):
1. On a **private** repo, the PR's `diff_url` (`github.com/.../pull/N.diff`)
   returns **404** even with a token — switched to the API endpoint
   (`api.github.com/.../pulls/N`) with the diff media type.
2. `GEMINI_MODEL` was `gemini-2.0-flash`, which has **0 free-tier quota** (429) —
   switched to `gemini-2.5-flash`.
This is a good "testing found real defects" story for the write-up.
