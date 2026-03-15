"""Vapi API client — chat simulation and assistant management."""

import time
import requests
from dataclasses import dataclass, field


BASE_URL = "https://api.vapi.ai"


@dataclass
class Turn:
    role: str
    content: str
    latency_ms: float = 0.0


@dataclass
class Conversation:
    scenario_id: str
    turns: list[Turn] = field(default_factory=list)
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    error: str = ""

    @property
    def transcript(self) -> str:
        return "\n".join(
            f"{'CALLER' if t.role == 'caller' else 'AGENT'}: {t.content}"
            for t in self.turns
        )

    @property
    def agent_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    @property
    def caller_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "caller"]


def run_conversation(api_key: str, assistant_id: str,
                     scenario_id: str, caller_turns: list[str],
                     max_turns: int = 12) -> Conversation:
    """Run a multi-turn conversation using Vapi Chat API with previousChatId."""
    conv = Conversation(scenario_id=scenario_id)
    prev_chat_id = None
    total_latency = 0.0

    for msg in caller_turns[:max_turns]:
        if not msg or not msg.strip():
            msg = "..."
        conv.turns.append(Turn(role="caller", content=msg))

        body = {"assistantId": assistant_id, "input": msg}
        if prev_chat_id:
            body["previousChatId"] = prev_chat_id

        try:
            t0 = time.time()
            resp = requests.post(
                f"{BASE_URL}/chat",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json=body, timeout=30,
            )
            latency = (time.time() - t0) * 1000

            if resp.status_code not in (200, 201):
                conv.error = f"API {resp.status_code}: {resp.text[:200]}"
                break

            data = resp.json()
            prev_chat_id = data.get("id", prev_chat_id)

            agent_msg = ""
            if "output" in data and data["output"]:
                agent_msg = data["output"][-1].get("content", "")

            conv.turns.append(Turn(role="assistant", content=agent_msg, latency_ms=latency))
            total_latency += latency
            conv.total_cost += data.get("cost", 0.0)

            if any(m in agent_msg.lower() for m in
                   ["have a great day", "goodbye", "talk to you soon", "take care"]):
                break

        except requests.exceptions.Timeout:
            conv.error = "Timeout (>30s)"
            break
        except Exception as e:
            conv.error = str(e)[:200]
            break

        time.sleep(0.3)

    n = len(conv.agent_turns)
    conv.avg_latency_ms = total_latency / n if n else 0
    return conv


def get_assistant(api_key: str, assistant_id: str) -> dict:
    """Fetch assistant config from Vapi."""
    resp = requests.get(
        f"{BASE_URL}/assistant/{assistant_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    return resp.json()


def update_prompt(api_key: str, assistant_id: str, new_prompt: str) -> bool:
    """Update assistant system prompt. Returns True on success."""
    # Must include model, provider, and messages together
    assistant = get_assistant(api_key, assistant_id)
    model_cfg = assistant.get("model", {})

    resp = requests.patch(
        f"{BASE_URL}/assistant/{assistant_id}",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": {
                "model": model_cfg.get("model", "gpt-4o-mini"),
                "provider": model_cfg.get("provider", "openai"),
                "messages": [{"role": "system", "content": new_prompt}],
            }
        },
    )
    return resp.status_code in (200, 201)
