"""LLM client for Groq API calls.

Thin wrapper: handles retries, timeouts, and JSON extraction.
No domain-specific prompts live here — see evaluator.py for those.
"""

from __future__ import annotations

import json
import time

from groq import Groq


class LLMClient:
    """Groq API client with retry logic."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout: int = 120,
        max_retries: int = 5,
    ):
        self.model = model
        self.max_retries = max_retries
        self._client = Groq(api_key=api_key, timeout=float(timeout))

    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Make a Groq API call with exponential backoff retries."""
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return r.choices[0].message.content
            except Exception as e:
                if "decommissioned" in str(e).lower():
                    raise ValueError(
                        f"Groq model '{self.model}' is decommissioned. "
                        "Set llm.model in config.yaml to a currently supported model."
                    ) from e
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

    def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 500,
    ) -> str:
        """Multi-turn conversation. Used for simulated agent conversations.

        Args:
            system: System prompt (the voice agent's prompt).
            messages: Conversation history as [{"role": "user/assistant", "content": "..."}].
            max_tokens: Max response tokens.
        """
        for attempt in range(self.max_retries + 1):
            try:
                full_messages = [{"role": "system", "content": system}] + list(messages)
                r = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=full_messages,
                )
                return r.choices[0].message.content
            except Exception as e:
                if "decommissioned" in str(e).lower():
                    raise ValueError(
                        f"Groq model '{self.model}' is decommissioned. "
                        "Set llm.model in config.yaml to a currently supported model."
                    ) from e
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 30)
                    time.sleep(wait)
                else:
                    raise
        return ""

    def call_json(self, system: str, user: str, max_tokens: int = 2048):
        """Call Groq and parse the response as JSON."""
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
