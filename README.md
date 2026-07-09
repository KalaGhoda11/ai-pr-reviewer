# AI PR Reviewer

An LLM-powered code review service. When a pull request is opened, it pulls the
diff, asks **Gemini** to review it for bugs, security issues, and refactoring
opportunities (guided by an organization coding-standards document), and posts
the findings back as PR comments.

Built as an MSSE Capstone project: a deployed, tested, CI/CD-backed AI system.

**🌐 Live:** https://ai-pr-reviewer-sxdp.onrender.com &nbsp;·&nbsp; **📋 Board:** [Trello](https://trello.com/b/fHxg2MVA) &nbsp;·&nbsp; **📖 Handoff guide:** [docs/INSTRUCTIONS.md](docs/INSTRUCTIONS.md) &nbsp;·&nbsp; **🏗 Design doc:** [docs/DESIGN.md](docs/DESIGN.md)

See it work: PR #1 in this repo has a real 12-finding review posted by the bot.

## Architecture (target)

```
GitHub PR event ──► /webhook (FastAPI)
                        │  verify HMAC signature
                        ▼
                   fetch PR diff (GitHub API)
                        │
                        ▼
                 review engine ──► Gemini  (+ coding_standards.md)
                        │
                        ▼
                 post review comments back to the PR
```

## Tech stack

| Layer     | Choice                          |
|-----------|---------------------------------|
| Backend   | Python 3.12+ / FastAPI          |
| LLM       | Google Gemini (`google-genai`)  |
| GitHub    | Webhook + REST (PyGithub)       |
| Hosting   | Render (free tier)              |
| CI/CD     | GitHub Actions                  |
| Tests     | pytest                          |

## Local development

```bash
# create + activate a virtualenv, then:
pip install -r requirements.txt
cp .env.example .env          # fill in your keys
uvicorn app.main:app --reload
```

Visit http://localhost:8000/health and http://localhost:8000/docs

## Testing

```bash
pytest -q
```

## Deliverables (capstone)

- [x] GitHub repository (this repo)
- [x] Deployed live link (Render) — https://ai-pr-reviewer-sxdp.onrender.com
- [x] Agile task board (Trello) — https://trello.com/b/fHxg2MVA
- [x] Design & testing document ([docs/DESIGN.md](docs/DESIGN.md))
- [x] CI/CD pipeline (GitHub Actions)
- [ ] Demo recording
- [ ] Share repo with `quantic-grader`

## Roadmap

- **MVP:** webhook → diff → Gemini review → PR comments
- **Stretch:** vector-store RAG over standards, inline line-level comments,
  severity scoring, a settings UI.
