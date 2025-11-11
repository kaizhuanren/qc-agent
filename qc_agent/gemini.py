"""Gemini (Google GenAI) client with RPM limiting and retries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import logging
import threading
import time

try:
    from google import genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore

logger = logging.getLogger(__name__)


class GeminiClientError(RuntimeError):
    """Raised when the Gemini API fails."""


@dataclass
class GeminiResponse:
    text: str
    raw: Any


class GeminiClient:
    def __init__(self, api_key: str, model: str, rpm_limit: int = 1, max_retries: int = 3) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing; please set the environment variable.")
        if genai is None:
            raise ImportError("google-genai package is not installed.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.rpm_limit = max(1, rpm_limit)
        self.max_retries = max(1, max_retries)
        self._lock = threading.Lock()
        self._last_call_ts = 0.0

    def generate(self, contents: str) -> GeminiResponse:
        last_error: Exception | None = None
        backoff = 2.0
        for attempt in range(self.max_retries):
            try:
                self._respect_rate_limit()
                response = self.client.models.generate_content(model=self.model, contents=contents)
                text = self._extract_text(response)
                if text:
                    return GeminiResponse(text=text, raw=response)
                raise GeminiClientError("Gemini returned empty content")
            except Exception as exc:  # pragma: no cover - network errors
                last_error = exc
                logger.warning("Gemini request failed (attempt %s/%s): %s", attempt + 1, self.max_retries, exc)
                time.sleep(backoff)
                backoff *= 2
        raise GeminiClientError(f"Gemini request failed after {self.max_retries} attempts: {last_error}")

    def _respect_rate_limit(self) -> None:
        with self._lock:
            now = time.time()
            interval = 60.0 / self.rpm_limit
            delta = now - self._last_call_ts
            if delta < interval:
                wait = interval - delta
                time.sleep(wait)
                now = time.time()
            self._last_call_ts = now

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        candidates = getattr(response, "candidates", None)
        if isinstance(candidates, list):
            for cand in candidates:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                parts = getattr(content, "parts", None)
                if isinstance(parts, list):
                    texts = [getattr(part, "text", "") for part in parts]
                    joined = "".join(t for t in texts if t)
                    if joined.strip():
                        return joined.strip()
        return ""
