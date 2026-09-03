"""FastAPI interface for the mechanical-design agent."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from src.agent import MechanicalDesignAgent
from src.llm import OllamaError

app = FastAPI(
    title="Mechanical Design RAG",
    description="Cited document answers and deterministic engineering calculations.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    params: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    id: str
    source: str
    section: str
    page: int


class AskResponse(BaseModel):
    answer: str
    intent: str
    calculation: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    missing_parameters: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_agent() -> MechanicalDesignAgent:
    return MechanicalDesignAgent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        result = await run_in_threadpool(
            get_agent().ask, request.query, request.params
        )
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AskResponse(
        answer=result["answer"],
        intent=result["intent"],
        calculation=result.get("calc_type"),
        parameters=result.get("params", {}),
        missing_parameters=result.get("missing_params", []),
        citations=result.get("citations", []),
    )
