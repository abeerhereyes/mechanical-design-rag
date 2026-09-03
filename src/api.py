"""FastAPI and browser interface for the multi-course student assistant."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from functools import lru_cache
import importlib
import inspect
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent import MechanicalDesignAgent
from src.llm import OllamaError

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent / "web"
CORPUS_DIR = ROOT / "data" / "corpus"
COURSES_DIR = ROOT / "data" / "courses"
DEFAULT_COURSE_ID = "mechanical-design"

app = FastAPI(
    title="Course Notes Assistant",
    description="Cited course-note answers, general context, and engineering calculations.",
    version="2.0.0",
)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    params: dict[str, Any] = Field(default_factory=dict)
    course_id: str = Field(default=DEFAULT_COURSE_ID, min_length=1, max_length=100)
    include_general: bool = True
    formula_id: Optional[str] = None
    units: dict[str, str] = Field(default_factory=dict)


class Citation(BaseModel):
    id: str = ""
    source: str
    section: str = ""
    page: int = Field(default=1, ge=1)
    preview_url: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    course_answer: str
    general_answer: Optional[str] = None
    course_id: str
    intent: str
    calculation: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    missing_parameters: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_agent() -> MechanicalDesignAgent:
    return MechanicalDesignAgent()


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _call_agent(request: AskRequest) -> Mapping[str, Any]:
    """Call both the future multi-course and current legacy agent signatures."""
    method = get_agent().ask
    try:
        signature = inspect.signature(method)
        parameters = signature.parameters
        accepts_kwargs = any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in parameters.values()
        )
    except (TypeError, ValueError):
        parameters, accepts_kwargs = {}, True

    kwargs: dict[str, Any] = {}
    if "params" in parameters or accepts_kwargs:
        kwargs["params"] = request.params
    if "course_id" in parameters or accepts_kwargs:
        kwargs["course_id"] = request.course_id
    if "include_general" in parameters or accepts_kwargs:
        kwargs["include_general"] = request.include_general
    if "formula_id" in parameters or accepts_kwargs:
        kwargs["formula_id"] = request.formula_id
    if "units" in parameters or accepts_kwargs:
        kwargs["units"] = request.units
    result = method(request.query, **kwargs)
    if not isinstance(result, Mapping):
        raise ValueError("The agent returned an invalid response")
    return result


def _normalise_citations(value: Any) -> list[Citation]:
    citations: list[Citation] = []
    if not isinstance(value, (list, tuple)):
        return citations
    for raw in value:
        item = _plain(raw)
        if not isinstance(item, Mapping) or not item.get("source"):
            continue
        try:
            page = max(1, int(item.get("page") or 1))
        except (TypeError, ValueError):
            page = 1
        source = Path(str(item["source"])).name
        citations.append(
            Citation(
                id=str(item.get("id") or ""),
                source=source,
                section=str(item.get("section") or ""),
                page=page,
                preview_url=f"/sources/{source}/pages/{page}",
            )
        )
    return citations


def _normalise_course(raw: Any) -> Optional[dict[str, Any]]:
    item = _plain(raw)
    if isinstance(item, str):
        item = {"id": item, "name": item.replace("-", " ").title()}
    if not isinstance(item, Mapping):
        return None
    course_id = item.get("id") or item.get("course_id") or item.get("slug")
    if not course_id:
        return None
    return {
        "id": str(course_id),
        "name": str(item.get("name") or item.get("title") or course_id),
        "description": str(item.get("description") or ""),
        "available": bool(item.get("available", True)),
    }


def _registered_courses() -> list[dict[str, Any]]:
    """Use an eventual course registry without requiring it today."""
    agent = get_agent()
    providers: list[Any] = [
        agent,
        getattr(agent, "course_registry", None),
        getattr(agent, "registry", None),
    ]
    for module_name in ("src.course_registry", "src.courses"):
        try:
            providers.append(importlib.import_module(module_name))
        except Exception:
            pass
    for provider in providers:
        list_courses = getattr(provider, "list_courses", None)
        if callable(list_courses):
            try:
                courses = [
                    normalised
                    for item in list_courses()
                    if (normalised := _normalise_course(item)) is not None
                ]
            except Exception:
                continue
            if courses:
                return courses
    manifests = []
    for path in sorted(COURSES_DIR.glob("*/manifest.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            course = _normalise_course(item)
        except (OSError, ValueError, TypeError):
            continue
        if course is not None:
            manifests.append(course)
    if manifests:
        return sorted(
            manifests,
            key=lambda course: (course["id"] != DEFAULT_COURSE_ID, course["name"]),
        )
    return [
        {
            "id": DEFAULT_COURSE_ID,
            "name": "Mechanical Design",
            "description": "Springs, bolted joints, welds, and bearings.",
            "available": bool(list(CORPUS_DIR.glob("*.md"))),
        }
    ]


def _fallback_calculation_schemas() -> list[dict[str, Any]]:
    return []


def _formula_schema(formula: Any) -> Optional[dict[str, Any]]:
    """Project a registry definition without exposing its executor callable."""
    item = _plain(formula)
    get = (
        (lambda name, default=None: item.get(name, default))
        if isinstance(item, Mapping)
        else (lambda name, default=None: getattr(formula, name, default))
    )
    formula_id = get("formula_id") or get("id")
    if not formula_id:
        return None
    variables = []
    for variable in get("variables", ()) or ():
        value = _plain(variable)
        if isinstance(value, Mapping):
            variables.append(
                {
                    key: value.get(key)
                    for key in (
                        "name", "description", "unit", "value_type", "required",
                        "default", "minimum", "maximum", "choices",
                    )
                    if key in value
                }
            )
    source = _plain(get("source"))
    if isinstance(source, Mapping):
        source = {
            key: value for key, value in source.items()
            if key in {"document", "pages", "section"}
        }
    return {
        "id": str(formula_id),
        "course_id": str(get("course_id") or DEFAULT_COURSE_ID),
        "title": str(get("title") or formula_id),
        "equation": get("equation"),
        "parameters": variables,
        "outputs": [_plain(output) for output in (get("outputs", ()) or ())],
        "source": source,
        "assumptions": _as_strings(get("assumptions")),
        "warnings": _as_strings(get("discrepancy_warning")),
        "review_state": str(get("review_state") or ""),
    }


def _calculation_schemas(course_id: Optional[str] = None) -> list[dict[str, Any]]:
    agent = get_agent()
    registry = getattr(agent, "calculation_registry", None) or getattr(
        agent, "formula_registry", None
    )
    if registry is None:
        try:
            registry = importlib.import_module("src.calculations").registry
        except Exception:
            registry = None
    list_formulas = getattr(registry, "list", None)
    if callable(list_formulas):
        try:
            formulas = list_formulas(course_id=course_id) if course_id else list_formulas()
            schemas = [
                schema
                for formula in formulas
                if (schema := _formula_schema(formula)) is not None
            ]
            if schemas:
                return schemas
        except Exception:
            pass
    schemas = _fallback_calculation_schemas()
    return [
        schema for schema in schemas
        if course_id is None or schema["course_id"] == course_id
    ]


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/courses")
def list_courses() -> dict[str, list[dict[str, Any]]]:
    return {"courses": _registered_courses()}


@app.get("/ingestion/status")
def ingestion_status(course_id: Optional[str] = None) -> dict[str, Any]:
    agent = get_agent()
    for name in ("get_ingestion_status", "ingestion_status"):
        provider = getattr(agent, name, None)
        if callable(provider):
            try:
                status = _plain(provider())
                if isinstance(status, Mapping):
                    return dict(status)
            except Exception:
                pass
        elif isinstance(provider, Mapping):
            return dict(provider)
    documents = sorted(CORPUS_DIR.glob("*.md"))
    courses = []
    for course in _registered_courses():
        manifest_path = COURSES_DIR / course["id"] / "manifest.json"
        expected = []
        if manifest_path.is_file():
            try:
                expected = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                ).get("sources", [])
            except (OSError, ValueError, TypeError):
                expected = []
        present = sum(
            1 for source in expected
            if (
                manifest_path.parent
                / "sources"
                / str(source.get("file", ""))
            ).is_file()
        )
        if course["id"] == DEFAULT_COURSE_ID:
            present += len(documents)
        courses.append(
            {
                "course_id": course["id"],
                "status": "ready" if present else "registered",
                "documents": present,
                "expected_sources": len(expected),
            }
        )
    if course_id:
        for status in courses:
            if status["course_id"] == course_id:
                return status
        raise HTTPException(status_code=404, detail="Unknown course")
    return {
        "status": "ready" if any(item["documents"] for item in courses) else "registered",
        "documents": sum(item["documents"] for item in courses),
        "sources": [document.name for document in documents],
        "courses": courses,
    }


@app.get("/calculations")
def list_calculations(course_id: Optional[str] = None) -> dict[str, Any]:
    return {"calculations": _calculation_schemas(course_id)}


@app.get("/calculations/{calculation_id}/schema")
def calculation_schema(calculation_id: str) -> dict[str, Any]:
    for schema in _calculation_schemas():
        if schema.get("id") == calculation_id or schema.get("formula_id") == calculation_id:
            return schema
    for schema in _fallback_calculation_schemas():
        if schema["id"] == calculation_id:
            return schema
    raise HTTPException(status_code=404, detail="Unknown calculation")


@app.get("/sources/{source}/pages/{page}")
def source_page(source: str, page: int) -> dict[str, Any]:
    safe_source = Path(source).name
    if safe_source != source or page < 1:
        raise HTTPException(status_code=404, detail="Source page not found")
    path = CORPUS_DIR / safe_source
    if not path.is_file():
        pdf = next(
            (
                directory / "sources" / safe_source
                for directory in COURSES_DIR.iterdir()
                if directory.is_dir()
                and (directory / "sources" / safe_source).suffix.lower() == ".pdf"
                and (directory / "sources" / safe_source).is_file()
            ),
            None,
        ) if COURSES_DIR.is_dir() else None
        if pdf is not None:
            course_id = pdf.parent.parent.name
            return {
                "source": safe_source,
                "page": page,
                "section": "",
                "title": f"{safe_source} — page {page}",
                "content": "",
                "kind": "pdf",
                "document_url": f"/source-files/{course_id}/{safe_source}#page={page}",
            }
        raise HTTPException(status_code=404, detail="Source page not found")
    if path.suffix.lower() != ".md":
        raise HTTPException(status_code=404, detail="Source page not found")

    text = path.read_text(encoding="utf-8")
    marker = re.compile(
        rf"^##\s+(?P<title>.+?)\s+\(p\.\s*{page}\)\s*:\s*(?P<section>.*)$",
        re.MULTILINE | re.IGNORECASE,
    )
    match = marker.search(text)
    if not match:
        raise HTTPException(status_code=404, detail="Source page not found")
    next_heading = re.search(r"^##\s+", text[match.end():], re.MULTILINE)
    end = match.end() + next_heading.start() if next_heading else len(text)
    content = text[match.end():end].strip()
    return {
        "source": safe_source,
        "page": page,
        "section": match.group("title").strip(),
        "title": match.group("section").strip(),
        "content": content,
        "kind": "text",
    }


@app.get("/source-files/{course_id}/{source}", include_in_schema=False)
def source_file(course_id: str, source: str) -> FileResponse:
    if Path(course_id).name != course_id or Path(source).name != source:
        raise HTTPException(status_code=404, detail="Source not found")
    path = COURSES_DIR / course_id / "sources" / source
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise HTTPException(status_code=404, detail="Source not found")
    return FileResponse(path, media_type="application/pdf", filename=source)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        result = await run_in_threadpool(_call_agent, request)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    answer = str(result.get("answer") or "")
    course_answer = str(result.get("course_answer") or answer)
    general = result.get("general_answer")
    return AskResponse(
        answer=answer or course_answer or str(general or ""),
        course_answer=course_answer,
        general_answer=str(general) if general is not None else None,
        course_id=str(result.get("course_id") or request.course_id),
        intent=str(result.get("intent") or "answer"),
        calculation=result.get("calc_type"),
        parameters=dict(result.get("params") or {}),
        missing_parameters=_as_strings(result.get("missing_params")),
        citations=_normalise_citations(result.get("citations")),
        warnings=_as_strings(result.get("warnings")),
        assumptions=_as_strings(result.get("assumptions")),
    )
