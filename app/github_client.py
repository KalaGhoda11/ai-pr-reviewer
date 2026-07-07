"""Thin wrappers around GitHub: signature verification, diff fetch, comment post.

Kept dependency-light and side-effect-isolated so the webhook handler and tests
can reason about each piece independently.
"""

from __future__ import annotations

import hashlib
import hmac

import httpx
from github import Github


def verify_signature(payload_body: bytes, signature_header: str, secret: str) -> bool:
    """Verify a GitHub webhook's ``X-Hub-Signature-256`` header.

    Uses constant-time comparison to avoid timing attacks. Returns False on any
    missing/malformed input rather than raising, so the caller can 401 cleanly.
    """
    if not secret or not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), payload_body, hashlib.sha256).hexdigest()
    provided = signature_header.removeprefix("sha256=")
    return hmac.compare_digest(expected, provided)


def fetch_diff(diff_url: str, token: str) -> str:
    """Fetch the unified diff for a PR.

    GitHub returns the raw diff when the ``.diff`` URL is requested (or the API
    URL with the ``diff`` media type). We use the PR's ``diff_url`` directly.
    """
    headers = {"Accept": "application/vnd.github.v3.diff"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = httpx.get(diff_url, headers=headers, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    return resp.text


def post_review_comment(repo_full_name: str, pr_number: int, body: str, token: str) -> None:
    """Post a single summary comment on the PR's conversation timeline."""
    gh = Github(token)
    repo = gh.get_repo(repo_full_name)
    pull = repo.get_pull(pr_number)
    pull.create_issue_comment(body)
