from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

REVIEW_RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "rule_id": {"type": "string"}, "entity_path": {"type": "string"}, "entity_id": {"type": "string"}, "explanation": {"type": "string"},
                    "evidence": {"type": "object", "additionalProperties": False, "properties": {"observed": {"type": "string"}, "expected": {"type": "string"}}, "required": ["observed", "expected"]},
                    "rule_citation": {"type": "object", "additionalProperties": False, "properties": {"catalog_id": {"type": "string"}, "section": {"type": "string"}, "rule_id": {"type": "string"}}, "required": ["catalog_id", "section", "rule_id"]},
                },
                "required": ["rule_id", "entity_path", "entity_id", "explanation", "evidence", "rule_citation"],
            },
        },
        "abstained": {"type": "boolean"}, "abstention_reason": {"type": "string"},
    },
    "required": ["findings", "abstained", "abstention_reason"],
}


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot complete a request."""


@dataclass(frozen=True)
class OllamaResponse:
    content: str
    metadata: dict[str, Any]


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}. Start it with 'ollama serve' and try again: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned a response that was not valid JSON.") from exc

    def models(self) -> list[str]:
        return sorted(item["name"] for item in self._request("/api/tags").get("models", []) if item.get("name"))

    def ensure_models(self, requested: list[str]) -> list[str]:
        available = self.models()
        missing = [model for model in requested if model not in available]
        if missing:
            raise OllamaError(f"Requested model(s) are not installed: {', '.join(missing)}. Available models: {', '.join(available) or 'none'}")
        return available

    def chat(self, model: str, system: str, user: str, seed: int = 2027, max_tokens: int = 1200, keep_alive: str = "20m") -> OllamaResponse:
        payload = {
            "model": model, "stream": False, "format": REVIEW_RESPONSE_SCHEMA, "keep_alive": keep_alive,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": seed, "num_predict": max_tokens},
        }
        response = self._request("/api/chat", payload)
        content = response.get("message", {}).get("content")
        if not isinstance(content, str):
            raise OllamaError(f"Ollama response for {model} did not contain message.content.")
        metadata = {key: response.get(key) for key in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration", "done_reason")}
        return OllamaResponse(content=content, metadata=metadata)
