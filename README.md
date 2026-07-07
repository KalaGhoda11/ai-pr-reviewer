# AI PR Reviewer

An LLM-powered code review service. When a pull request is opened, it pulls the
diff, asks **Gemini** to review it for bugs, security issues, and refactoring
opportunities (guided by an organization coding-standards document), and posts
the findings back as PR comments.

Built as an MSSE Capstone project: a deployed, tested, CI/CD-backed AI system.

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
- [ ] Deployed live link (Render)
- [ ] Agile task board (Trello)
- [ ] Design & testing document (`docs/DESIGN.md`)
- [ ] Demo recording

## Roadmap

- **MVP:** webhook → diff → Gemini review → PR comments
- **Stretch:** vector-store RAG over standards, inline line-level comments,
  severity scoring, a settings UI.
