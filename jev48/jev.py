from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx

from .schema import DecisionExample


class JevClient:
    """Raw HTTP adapter for the Jev decisions API.

    Provider `auto` prefers a direct TypeSafe key, then falls back to OpenRouter's
    decisions endpoint. Both routes return TypeSafe's Jev typed answer shape.
    Jev outputs are external evaluation only and never enter training/selection.
    """

    def __init__(
        self,
        api_key: str | None = None,
        provider: str = "auto",
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float = 45.0,
        retries: int = 5,
    ):
        if provider not in {"auto", "typesafe", "openrouter"}:
            raise ValueError("provider must be auto, typesafe, or openrouter")
        ts = os.environ.get("TYPESAFE_API_KEY")
        or_key = os.environ.get("OPENROUTER_API_KEY")
        if provider == "auto":
            provider = "typesafe" if (api_key or ts) else "openrouter" if or_key else "typesafe"
        if provider == "typesafe":
            key = api_key or ts
            self.endpoint = endpoint or "https://api.typesafe.ai/v1/systemone"
            self.model = model or "jev-latest"
        else:
            key = api_key or or_key
            self.endpoint = endpoint or "https://openrouter.ai/api/alpha/decisions"
            self.model = model or "typesafe/jev-1.13"
        if not key:
            names = "TYPESAFE_API_KEY or OPENROUTER_API_KEY" if provider == "typesafe" and os.environ.get("OPENROUTER_API_KEY") is None else ("TYPESAFE_API_KEY" if provider == "typesafe" else "OPENROUTER_API_KEY")
            raise ValueError(f"Jev API access required ({names})")
        self.api_key = key
        self.provider = provider
        self.timeout = timeout
        self.retries = retries
        self.client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )

    def close(self) -> None:
        self.client.close()

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                response = self.client.post(self.endpoint, json=payload)
                latency_ms = (time.perf_counter() - start) * 1000
                if response.status_code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < self.retries:
                    time.sleep(min(8.0, 0.5 * (2**attempt)) + random.random() * 0.25)
                    continue
                response.raise_for_status()
                return response.json(), latency_ms
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(8.0, 0.5 * (2**attempt)) + random.random() * 0.25)
        raise RuntimeError(f"Jev request failed after {self.retries + 1} attempts: {last}") from last

    def score(self, example: DecisionExample) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "state": example.state,
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": example.question,
                    "criteria": {c.id: c.text for c in example.candidates},
                }
            },
        }
        body, latency_ms = self._post(payload)
        answer = body["answers"]["decision"]
        probs_map = answer["probabilities"]
        probs = [float(probs_map[c.id]) for c in example.candidates]
        total = sum(probs)
        if total <= 0:
            raise ValueError(f"Jev returned non-positive probability mass for {example.id}")
        probs = [p / total for p in probs]
        return {
            "id": example.id,
            "model": body.get("model", self.model),
            "probabilities": probs,
            "best": answer.get("choice"),
            "confidence": answer.get("confidence"),
            "latency_ms": latency_ms,
            "usage": body.get("usage"),
            "provider": body.get("provider", self.provider),
            "gateway": self.provider,
        }
