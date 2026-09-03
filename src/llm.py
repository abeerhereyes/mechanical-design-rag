"""Small Ollama client used for grounded answer generation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    """Raised when Ollama cannot generate a usable response."""


@dataclass
class OllamaClient:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    timeout: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

    def generate(self, query: str, chunks: list[dict]) -> str:
        context = "\n\n".join(
            f"[{i}] {item['metadata']['source']} — Section "
            f"{item['metadata']['section']} (p. {item['metadata']['page']})\n"
            f"{item['text']}"
            for i, item in enumerate(chunks, start=1)
        )
        system = (
            "You are a mechanical-design assistant. Answer only from the supplied "
            "context. Do not invent formulas, limits, or facts. Cite supporting "
            "context inline as [1], [2], etc. If the context is insufficient, say "
            "so plainly. Keep units and engineering caveats explicit."
        )
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            "options": {"temperature": 0},
        }
        request = Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url}. Start it with "
                f"`ollama serve` and install the model with `ollama pull {self.model}`."
            ) from exc

        answer = data.get("message", {}).get("content", "").strip()
        if not answer:
            raise OllamaError("Ollama returned an empty response")
        return answer
