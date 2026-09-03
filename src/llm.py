"""Small Ollama client used for grounded answer generation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    """Raised when Ollama cannot generate a usable response."""


@dataclass
class GroundedAnswer:
    course_answer: str
    general_answer: Optional[str]
    citation_ids: list[str]
    insufficient_course_context: bool = False


@dataclass
class OllamaClient:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    timeout: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

    def _chat(self, payload: dict) -> dict:
        request = Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url}. Start it with "
                f"`ollama serve` and install the model with `ollama pull {self.model}`."
            ) from exc

    def generate_structured(
        self,
        query: str,
        chunks: list[dict],
        include_general: bool = True,
    ) -> GroundedAnswer:
        context = "\n\n".join(
            f"[{item['id']}] {item['metadata']['source']} — "
            f"{item['metadata'].get('section', item['metadata'].get('topic', ''))} "
            f"(p. {item['metadata']['page']})\n"
            f"{item['text']}"
            for item in chunks
        )
        system = (
            "You are a course-grounded study assistant. The selected course notes "
            "are authoritative for the course_answer. Do not invent formulas, limits, "
            "facts, or citation IDs. Set citation_ids only to IDs shown in the context. "
            "If notes are insufficient, say so and set insufficient_course_context true. "
            "General knowledge, if requested, must be in general_answer and clearly "
            "separate from the course answer. Mention source inconsistencies without "
            "silently correcting them. Return strict JSON with keys course_answer, "
            "general_answer, citation_ids, insufficient_course_context."
        )
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\nQuestion: {query}\n"
                        f"Include general clarification: {include_general}"
                    ),
                },
            ],
            "options": {"temperature": 0},
        }
        try:
            raw = self._chat(payload).get("message", {}).get("content", "")
            result = json.loads(raw)
        except (json.JSONDecodeError, TypeError, AttributeError) as exc:
            raise OllamaError("Ollama returned an invalid structured response") from exc

        course_answer = str(result.get("course_answer", "")).strip()
        if not course_answer:
            raise OllamaError("Ollama returned an empty response")
        allowed = {item["id"] for item in chunks}
        citation_ids = [
            citation_id
            for citation_id in result.get("citation_ids", [])
            if citation_id in allowed
        ]
        general = result.get("general_answer")
        return GroundedAnswer(
            course_answer=course_answer,
            general_answer=str(general).strip() if general else None,
            citation_ids=citation_ids,
            insufficient_course_context=bool(
                result.get("insufficient_course_context", False)
            ),
        )

    def generate(self, query: str, chunks: list[dict]) -> str:
        """Backward-compatible plain course answer."""
        return self.generate_structured(query, chunks).course_answer
