"""Pure auth-mode and client classification policy."""

from __future__ import annotations

import base64
import enum
import json
from dataclasses import dataclass
from typing import Any


class AuthMode(str, enum.Enum):
    """Auth-mode classes Headroom routes compression policy through."""

    PAYG = "payg"
    OAUTH = "oauth"
    SUBSCRIPTION = "subscription"


SUBSCRIPTION_UA_PREFIXES: tuple[str, ...] = (
    "claude-cli/",
    "claude-code/",
    "codex-cli/",
    "cursor/",
    "claude-vscode/",
    "github-copilot/",
    "anthropic-cli/",
    "antigravity/",
)


CLIENT_UA_MAP: tuple[tuple[str, str], ...] = (
    ("claude-code/", "claude-code"),
    ("claude-cli/", "claude-code"),
    ("claude-vscode/", "claude-vscode"),
    ("anthropic-cli/", "anthropic-cli"),
    ("codex-cli/", "codex"),
    ("cursor/", "cursor"),
    # Current Grok Build releases send "grok-shell/<version>" (verified against
    # grok 0.2.112); older builds sent "grok/". Deliberately NOT added to
    # SUBSCRIPTION_UA_PREFIXES: API-key grok traffic is pay-per-token and
    # should keep the aggressive compression policy.
    ("grok-shell/", "grok_build"),
    ("grok/", "grok_build"),
    ("zed/", "zed"),
    ("aider/", "aider"),
    ("droid/", "droid"),
    ("opencode/", "opencode"),
    ("github-copilot/", "copilot"),
    ("antigravity/", "antigravity"),
    ("strands-agents/", "strands"),
)


CODEX_RESPONSES_PATH = "/v1/responses"
CODEX_RESPONSES_PATHS: tuple[str, ...] = (
    CODEX_RESPONSES_PATH,
    "/v1/codex/responses",
)


@dataclass(frozen=True)
class AuthSignals:
    """Normalized header signals used by pure auth/client classifiers."""

    user_agent: str = ""
    authorization: str = ""
    x_api_key: str = ""
    x_goog_api_key: str = ""
    x_client: str = ""
    chatgpt_account_id: str = ""

    @property
    def user_agent_lower(self) -> str:
        return self.user_agent.lower()


_OPENAI_AUTH_CLAIM_NAMESPACE = "https://api.openai.com/auth"


def _decode_bearer_jwt_payload(authorization: str) -> dict[str, Any] | None:
    """Best-effort JWT payload decode for routing/classification hints only."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token.count(".") < 2:
        return None

    payload = token.split(".", 2)[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def has_chatgpt_codex_subscription_hint(signals: AuthSignals) -> bool:
    """True for ChatGPT/Codex subscription auth forwarded by Codex/Pi.

    Pi's OpenAI-Codex provider uses its own ``pi (...)`` User-Agent but forwards
    ``chatgpt-account-id`` and/or the same account id claim inside the OAuth JWT.
    """
    if signals.chatgpt_account_id.strip():
        return True
    payload = _decode_bearer_jwt_payload(signals.authorization)
    auth_claims = payload.get(_OPENAI_AUTH_CLAIM_NAMESPACE) if payload else None
    account_id = auth_claims.get("chatgpt_account_id") if isinstance(auth_claims, dict) else None
    return isinstance(account_id, str) and bool(account_id.strip())


def classify_auth_signals(signals: AuthSignals) -> AuthMode:
    """Classify auth mode from normalized header values."""
    ua_lower = signals.user_agent_lower
    for prefix in SUBSCRIPTION_UA_PREFIXES:
        if prefix in ua_lower:
            return AuthMode.SUBSCRIPTION

    # ChatGPT/Codex subscription auth under a non-Codex UA (e.g. Pi): keep
    # history rewrites live-zone-only, like Codex CLI.
    if has_chatgpt_codex_subscription_hint(signals):
        return AuthMode.SUBSCRIPTION

    auth = signals.authorization
    # The auth-scheme token ("Bearer") is case-insensitive per RFC 7235 §2.1,
    # so a client sending `Authorization: bearer sk-...` must classify the same
    # as `Bearer`. A case-sensitive `startswith("Bearer ")` sent such a request
    # to the `elif auth:` (OAUTH) fallthrough, misclassifying a PAYG API key —
    # which mis-routes compression policy and mislabels cost/TOIN auth_mode.
    # Only the scheme is case-folded; the credential itself stays case-sensitive.
    scheme, sep, credentials = auth.partition(" ")
    if sep and scheme.lower() == "bearer":
        token = credentials
        if token.startswith("sk-ant-oat"):
            return AuthMode.OAUTH
        if token.startswith("sk-ant-api") or token.startswith("sk-"):
            return AuthMode.PAYG
        if len(token.split(".")) >= 3:
            return AuthMode.OAUTH
    elif auth:
        return AuthMode.OAUTH

    if signals.x_api_key:
        return AuthMode.PAYG
    if signals.x_goog_api_key:
        return AuthMode.PAYG
    return AuthMode.PAYG


def classify_client_signals(signals: AuthSignals, *, default: str | None = None) -> str | None:
    """Identify the client harness from normalized client signals."""
    explicit = signals.x_client.strip().lower()
    if explicit:
        return explicit
    ua_lower = signals.user_agent_lower
    if ua_lower:
        for needle, name in CLIENT_UA_MAP:
            if needle in ua_lower:
                return name
    # Pi's UA is `pi (...)`, so ChatGPT/Codex subscription auth identifies codex.
    if has_chatgpt_codex_subscription_hint(signals):
        return "codex"
    return default


def is_codex_responses_path(path: str) -> bool:
    """Return True for OpenAI Responses endpoints and their subpaths."""
    return any(root == path or path.startswith(root + "/") for root in CODEX_RESPONSES_PATHS)


def should_stamp_codex_client_signals(path: str, signals: AuthSignals) -> bool:
    """Whether an unidentified Responses caller should be stamped as Codex."""
    return is_codex_responses_path(path) and classify_client_signals(signals) is None
