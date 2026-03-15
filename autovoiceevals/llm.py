"""LLM client for Claude API calls.

Thin wrapper: handles retries, timeouts, and JSON extraction.
No domain-specific prompts live here — see evaluator.py for those.
"""

from __future__ import annotations

import json
import time

import anthropic
import httpx


class LLMClient:
    """Claude API client with retry logic."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        timeout: int = 120,
        max_retries: int = 5,
    ):
        self.model = model
        self.max_retries = max_retries

        http = httpx.Client(
            timeout=httpx.Timeout(float(timeout), connect=30.0),
            transport=httpx.HTTPTransport(retries=max_retries),
        )
        self._client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=max_retries,
            timeout=float(timeout),
            http_client=http,
        )

    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Make a Claude API call with exponential backoff retries."""
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return r.content[0].text
            except Exception as e:
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 30)
                    print(
                        f"      (retry {attempt + 1}/{self.max_retries} "
                        f"after {wait}s: {type(e).__name__})"
                    )
                    time.sleep(wait)
                else:
                    raise
        return ""  # unreachable, but keeps type checkers happy

    def call_json(self, system: str, user: str, max_tokens: int = 2048):
        """Call Claude and parse the response as JSON."""
        raw = self.call(system, user, max_tokens)
        return parse_json(raw)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def parse_json(raw: str):
    """Best-effort JSON extraction from LLM output.

    Handles: bare JSON, ```json fences, and embedded objects/arrays.
    Returns None if parsing fails entirely.
    """
    # Strip code fences
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting the first complete JSON structure
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        s = raw.find(start_char)
        e = raw.rfind(end_char) + 1
        if s >= 0 and e > s:
            try:
                return json.loads(raw[s:e])
            except json.JSONDecodeError:
                pass

    return None
