from __future__ import annotations

import os
import time
import httpx

from .schema import DecisionExample


class JevClient:
    """Minimal raw HTTP adapter for TypeSafe's System One Choice endpoint."""

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.typesafe.ai", model: str = "jev-latest", timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("TYPESAFE_API_KEY is required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def score(self, example: DecisionExample) -> dict:
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
        start = time.perf_counter()
        response = httpx.post(
            self.base_url + "/v1/systemone",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        body = response.json()
        answer = body["answers"]["decision"]
        probs_map = answer["probabilities"]
        probs = [float(probs_map[c.id]) for c in example.candidates]
        return {
            "id": example.id,
            "model": body.get("model", self.model),
            "probabilities": probs,
            "best": answer.get("choice"),
            "confidence": answer.get("confidence"),
            "latency_ms": latency_ms,
            "usage": body.get("usage"),
            "provider": body.get("provider", "TypeSafe"),
        }
