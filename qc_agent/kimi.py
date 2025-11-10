"""Lightweight client for the Kimi (Moonshot) API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
import logging
import requests


logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str
    content: str


class KimiClientError(RuntimeError):
    """Raised when the Kimi API fails."""


class KimiClient:
    """Minimal blocking client for the `responses` endpoint."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        if not api_key:
            raise ValueError("KIMI_API_KEY is missing; please set the environment variable.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = requests.Session()

    def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/responses"
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = self.session.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code >= 300:
            logger.error("Kimi API error %s: %s", resp.status_code, resp.text[:2000])
            raise KimiClientError(f"Kimi API failed: {resp.status_code}")
        data = resp.json()
        text = self._extract_text(data)
        if text is None:
            raise KimiClientError("Kimi API returned no assistant text.")
        return {"text": text.strip(), "raw": data}

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str | None:
        # Format 1: OpenAI-compatible choices
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
        # Format 2: responses API output blocks
        output = data.get("output")
        if isinstance(output, list):
            parts: List[str] = []
            for block in output:
                for item in block.get("content", []):
                    if item.get("type") in {"output_text", "text"}:
                        parts.append(item.get("text", ""))
            if parts:
                return "".join(parts)
        return None
