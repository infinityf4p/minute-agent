"""Minimal chat-completions client with tool calling (DeepSeek, OpenAI-compatible).

No SDK: plain HTTPS via urllib. The key comes from DEEPSEEK_API_KEY in the
environment or the repo's .env file (git-ignored). Every call's token usage is
appended to a JSONL log so the cost of a session can be reported.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from minute_agent.paths import REPO_ROOT

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"      # DeepSeek-V4.1-Flash, per GET /models on 2026-11-02


def load_key(name: str = "DEEPSEEK_API_KEY") -> str:
    if os.environ.get(name):
        return os.environ[name]
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(f"{name} not set (environment or .env)")


@dataclass
class LLMClient:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.7
    max_tokens: int = 8000
    timeout: int = 300
    log_path: Path | None = None
    usage: dict = field(default_factory=lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})

    def __post_init__(self):
        self.key = load_key()

    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice=None) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
            if tool_choice:
                body["tool_choice"] = tool_choice
        data = json.dumps(body).encode()
        last_err = None
        for attempt in range(5):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=data, method="POST",
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.load(r)
                u = resp.get("usage", {})
                self.usage["calls"] += 1
                self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += u.get("completion_tokens", 0)
                if self.log_path:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"t": time.time(), "secs": round(time.time() - t0, 1),
                                            "usage": u, "model": self.model}) + "\n")
                return resp["choices"][0]["message"]
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read()[:500]!r}"
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = repr(e)
            time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError(f"LLM call failed: {last_err}")


def tool(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}
