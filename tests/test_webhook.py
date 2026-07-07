"""Tests for signature verification and the webhook route (fully mocked)."""

import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from app import webhook
from app.config import get_settings
from app.github_client import verify_signature
from app.main import app

client = TestClient(app)


def _sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_verify_signature_valid_and_invalid():
    secret = "topsecret"
    body = b'{"hello":"world"}'
    good = _sign(body, secret)
    assert verify_signature(body, good, secret) is True
    assert verify_signature(body, "sha256=deadbeef", secret) is False
    assert verify_signature(body, "", secret) is False
    assert verify_signature(body, good, "") is False


def _configure_secret(monkeypatch, secret="webhook-secret"):
    get_settings.cache_clear()
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", secret)
    return secret


def test_webhook_rejects_bad_signature(monkeypatch):
    _configure_secret(monkeypatch)
    resp = client.post("/webhook", content=b"{}",
                       headers={"X-Hub-Signature-256": "sha256=nope",
                                "X-GitHub-Event": "pull_request"})
    assert resp.status_code == 401


def test_webhook_ping(monkeypatch):
    secret = _configure_secret(monkeypatch)
    body = json.dumps({"zen": "hi"}).encode()
    resp = client.post("/webhook", content=body,
                       headers={"X-Hub-Signature-256": _sign(body, secret),
                                "X-GitHub-Event": "ping"})
    assert resp.status_code == 200
    assert resp.json()["msg"] == "pong"


def test_webhook_ignores_non_pr_event(monkeypatch):
    secret = _configure_secret(monkeypatch)
    body = json.dumps({"action": "created"}).encode()
    resp = client.post("/webhook", content=body,
                       headers={"X-Hub-Signature-256": _sign(body, secret),
                                "X-GitHub-Event": "issues"})
    assert resp.status_code == 200
    assert "ignored" in resp.json()["msg"]


def test_webhook_queues_review_on_pr_opened(monkeypatch):
    secret = _configure_secret(monkeypatch)
    called = {}

    def fake_process(payload):
        called["pr"] = payload["pull_request"]["number"]

    monkeypatch.setattr(webhook, "process_pull_request", fake_process)

    body = json.dumps({
        "action": "opened",
        "pull_request": {"number": 7, "diff_url": "http://x/diff"},
        "repository": {"full_name": "o/r"},
    }).encode()
    resp = client.post("/webhook", content=body,
                       headers={"X-Hub-Signature-256": _sign(body, secret),
                                "X-GitHub-Event": "pull_request"})
    assert resp.status_code == 200
    assert resp.json()["pr"] == 7
    assert called.get("pr") == 7
