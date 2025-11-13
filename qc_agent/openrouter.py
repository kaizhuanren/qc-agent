"""Lightweight client for the OpenRouter (OpenAI-compatible) API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Optional
import json
import logging
import time

import requests

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str
    content: str


class OpenRouterClientError(RuntimeError):
    """Raised when the OpenRouter API fails."""


class OpenRouterClient:
    """Blocking client for the OpenRouter chat completions endpoint with retries."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        retry: int = 3,
        timeout: int = 120,
        log_path: Optional[Path] = None,
        referer: str = "",
        app_name: str = "",
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing; please set the environment variable.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.retry = max(1, retry)
        self.timeout = timeout
        self.session = requests.Session()
        self.log_path = Path(log_path) if log_path else None
        extras: Dict[str, str] = dict(extra_headers or {})
        referer = referer.strip()
        app_name = app_name.strip()
        if referer:
            extras.setdefault("HTTP-Referer", referer)
        if app_name:
            extras.setdefault("X-Title", app_name)
        self.extra_headers = extras
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
        stream: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "stream": stream,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        serialized_messages = [{"role": m.role, "content": m.content} for m in messages]

        last_error: Exception | None = None
        for attempt in range(self.retry):
            log_meta = {
                "model": self.model,
                "attempt": attempt + 1,
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "stream": stream,
                "messages": serialized_messages,
            }
            try:
                resp = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )
                if resp.status_code >= 300:
                    logger.warning("OpenRouter API error %s: %s", resp.status_code, resp.text[:2000])
                    raise OpenRouterClientError(f"OpenRouter API failed: {resp.status_code}")
                if stream:
                    text, data = self._consume_stream(resp)
                else:
                    data = resp.json()
                    text = self._extract_text(data)
                if text and text.strip():
                    self._log_entry({**log_meta, "status": "success", "response": _truncate_json(data)})
                    return {"text": text.strip(), "raw": data}
                logger.warning("OpenRouter empty content, raw=%s", _truncate_json(data))
                self._log_entry({**log_meta, "status": "empty", "response": _truncate_json(data)})
                raise OpenRouterClientError("OpenRouter API returned empty content")
            except (
                requests.Timeout,
                requests.ConnectionError,
                OpenRouterClientError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                backoff = 2 ** attempt
                logger.warning("OpenRouter request failed (attempt %s/%s): %s", attempt + 1, self.retry, exc)
                self._log_entry({**log_meta, "status": "error", "error": str(exc)})
                time.sleep(backoff)
        raise OpenRouterClientError(f"OpenRouter chat failed after {self.retry} attempts: {last_error}")

    def _consume_stream(self, resp: requests.Response) -> tuple[str, Dict[str, Any]]:
        text_parts: List[str] = []
        raw_chunks: List[Dict[str, Any]] = []
        usage: Dict[str, Any] | None = None
        try:
            for line_bytes in resp.iter_lines():
                if not line_bytes:
                    continue
                line = line_bytes.decode('utf-8')
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                raw_chunks.append(chunk)
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        text_parts.append(content)
                    if not usage:
                        usage = choices[0].get("usage")
        finally:
            resp.close()
        raw = {"chunks": raw_chunks}
        if usage:
            raw["usage"] = usage
        return "".join(text_parts), raw

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str | None:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") in {"text", "output_text", "message"}:
                        parts.append(item.get("text", item.get("value", "")))
                if parts:
                    return "".join(parts)
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

    def _log_entry(self, entry: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        entry.setdefault("ts", time.time())
        try:
            payload = json.dumps(entry, ensure_ascii=False)
        except Exception:
            payload = str(entry)
        with self.log_path.open("a", encoding="utf-8") as logfile:
            logfile.write(payload + "\n")


def _truncate_json(data: Any, length: int = 2000) -> str:
    try:
        payload = json.dumps(data, ensure_ascii=False)
    except Exception:
        payload = str(data)
    return payload[:length]
