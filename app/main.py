"""FastAPI application entrypoint.

Day 1 exposes only a health check so we have something deployable and
testable end-to-end before wiring in the review engine and webhook.
"""

from fastapi import FastAPI

from app import __version__
from app.webhook import router as webhook_router

app = FastAPI(
    title="AI PR Reviewer",
    description="LLM-powered GitHub pull request reviewer.",
    version=__version__,
)

app.include_router(webhook_router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness probe used by Render and CI smoke tests."""
    return {"status": "ok", "version": __version__}


@app.get("/", tags=["ops"])
def root() -> dict:
    return {"service": "ai-pr-reviewer", "docs": "/docs"}
