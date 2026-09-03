"""Course manifest schema and loader."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .schemas import SCHEMA_VERSION


@dataclass(frozen=True)
class SourceSpec:
    id: str
    file: str
    title: str
    kind: str = "pdf"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CourseManifest:
    id: str
    title: str
    sources: List[SourceSpec]
    description: str = ""
    schema_version: str = SCHEMA_VERSION


def load_manifest(path: Path) -> CourseManifest:
    """Load and validate a JSON manifest (or YAML when PyYAML is installed)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Course manifest not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML manifests require PyYAML; install it or use manifest.json"
            ) from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be an object: {path}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version must be {SCHEMA_VERSION!r}"
        )
    course_id = str(data.get("id", "")).strip()
    title = str(data.get("title", "")).strip()
    if not course_id or not title:
        raise ValueError(f"{path}: non-empty id and title are required")
    raw_sources = data.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError(f"{path}: sources must be a list")
    sources: List[SourceSpec] = []
    seen = set()
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: each source must be an object")
        source = SourceSpec(
            id=str(raw.get("id", "")).strip(),
            file=str(raw.get("file", "")).strip(),
            title=str(raw.get("title", "")).strip(),
            kind=str(raw.get("kind", "pdf")).strip(),
            metadata=dict(raw.get("metadata", {})),
        )
        if not source.id or not source.file or not source.title:
            raise ValueError(f"{path}: every source needs id, file, and title")
        if source.id in seen:
            raise ValueError(f"{path}: duplicate source id {source.id!r}")
        if source.kind != "pdf":
            raise ValueError(f"{path}: unsupported source kind {source.kind!r}")
        seen.add(source.id)
        sources.append(source)
    return CourseManifest(
        id=course_id,
        title=title,
        description=str(data.get("description", "")),
        sources=sources,
    )


def discover_courses(courses_dir: Path) -> Dict[str, tuple[CourseManifest, Path]]:
    courses: Dict[str, tuple[CourseManifest, Path]] = {}
    for path in sorted(Path(courses_dir).glob("*/manifest.*")):
        manifest = load_manifest(path)
        if manifest.id in courses:
            raise ValueError(f"Duplicate course id {manifest.id!r}")
        courses[manifest.id] = (manifest, path.parent)
    return courses
