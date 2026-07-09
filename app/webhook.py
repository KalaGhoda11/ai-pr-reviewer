"""GitHub webhook endpoint: receive PR events, review the diff, comment back.

Flow:
    1. Verify the HMAC signature (401 if invalid).
    2. Ignore anything that isn't a PR opened/synchronize/reopened event.
    3. Fetch the diff, review it with Gemini, post the formatted comment.

The review is offloaded to a background task so GitHub gets a fast 202 and does
not time out or retry while Gemini is thinking.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from app.config import get_settings
from app.github_client import fetch_diff, post_review_comment, verify_signature
from app.review_engine import format_comment, load_standards, review_diff

logger = logging.getLogger(__name__)
router = APIRouter()

RELEVANT_ACTIONS = {"opened", "synchronize", "reopened"}


def _get_gemini_client(api_key: str):
    """Lazily construct the Gemini client (import here so tests need no key)."""
    from google import genai

    return genai.Client(api_key=api_key)


def process_pull_request(payload: dict) -> None:
    """Review a PR and post the result. Runs as a background task."""
    settings = get_settings()
    pr = payload["pull_request"]
    repo_full_name = payload["repository"]["full_name"]
    pr_number = pr["number"]

    try:
        diff = fetch_diff(pr["url"], settings.github_token)
        client = _get_gemini_client(settings.gemini_api_key)
        result = review_diff(diff, client=client, model=settings.gemini_model,
                             standards=load_standards())
        post_review_comment(repo_full_name, pr_number, format_comment(result),
                            settings.github_token)
        logger.info("Reviewed %s#%s: %d finding(s)", repo_full_name, pr_number,
                    len(result.findings))
    except Exception:  # noqa: BLE001 - log and swallow so the worker doesn't crash
        logger.exception("Failed to review %s#%s", repo_full_name, pr_number)


@router.post("/webhook", tags=["github"])
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str = Header(default=""),
    x_github_event: str = Header(default=""),
):
    settings = get_settings()
    body = await request.body()

    if not verify_signature(body, x_hub_signature_256, settings.github_webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    if x_github_event == "ping":
        return {"msg": "pong"}

    if x_github_event != "pull_request":
        return {"msg": f"ignored event: {x_github_event}"}

    payload = json.loads(body)
    action = payload.get("action")
    if action not in RELEVANT_ACTIONS:
        return {"msg": f"ignored action: {action}"}

    background_tasks.add_task(process_pull_request, payload)
    return {"msg": "review queued", "pr": payload["pull_request"]["number"]}
