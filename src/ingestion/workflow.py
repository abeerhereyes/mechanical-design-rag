"""Persistence, review, approval, and index export workflow."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

from .duplicates import nearest_duplicate
from .manifest import CourseManifest, SourceSpec
from .pdf import ExtractionConfig, extract_pdf
from .schemas import (
    ContentKind,
    ContentUnit,
    IngestedDocument,
    ReviewStatus,
    document_from_dict,
    document_to_dict,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_document(path: Path) -> IngestedDocument:
    return document_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _existing_units(raw_dir: Path, exclude_document_id: str) -> List[ContentUnit]:
    units: List[ContentUnit] = []
    for path in sorted(raw_dir.glob("*.json")):
        document = read_document(path)
        if document.document_id != exclude_document_id:
            units.extend(document.units)
    return units


def ingest_source(
    manifest: CourseManifest,
    course_dir: Path,
    source: SourceSpec,
    config: Optional[ExtractionConfig] = None,
    duplicate_threshold: float = 0.88,
) -> Path:
    source_path = Path(course_dir) / "sources" / source.file
    raw_dir = Path(course_dir) / "raw"
    document = extract_pdf(
        source_path,
        manifest.id,
        source.id,
        raw_dir / "assets" / source.id,
        config,
    )
    candidates = _existing_units(raw_dir, document.document_id)
    for unit in document.units:
        duplicate = nearest_duplicate(unit.text, candidates, duplicate_threshold)
        if duplicate:
            unit.near_duplicate_of = duplicate
        candidates.append(unit)
    output = raw_dir / f"{document.document_id}.json"
    _write_json(output, document_to_dict(document))
    return output


def ingest_course(
    manifest: CourseManifest,
    course_dir: Path,
    source_ids: Optional[Iterable[str]] = None,
    config: Optional[ExtractionConfig] = None,
    duplicate_threshold: float = 0.88,
) -> List[Path]:
    selected = set(source_ids or [])
    unknown = selected - {source.id for source in manifest.sources}
    if unknown:
        raise ValueError(f"Unknown source ids: {', '.join(sorted(unknown))}")
    sources = [
        source
        for source in manifest.sources
        if not selected or source.id in selected
    ]
    return [
        ingest_source(manifest, course_dir, source, config, duplicate_threshold)
        for source in sources
    ]


def review_document(
    course_dir: Path,
    document_id: str,
    action: ReviewStatus,
    unit_ids: Optional[Iterable[str]] = None,
    content_kinds: Optional[Iterable[ContentKind]] = None,
) -> Path:
    if action not in {ReviewStatus.APPROVED, ReviewStatus.REJECTED}:
        raise ValueError("Review action must be approved or rejected")
    raw_path = Path(course_dir) / "raw" / f"{document_id}.json"
    document = read_document(raw_path)
    selected = set(unit_ids or [])
    selected_kinds = set(content_kinds or [])
    known = {unit.unit_id for unit in document.units}
    unknown = selected - known
    if unknown:
        raise ValueError(f"Unknown unit ids: {', '.join(sorted(unknown))}")
    for unit in document.units:
        unit_selected = not selected or unit.unit_id in selected
        kind_selected = not selected_kinds or unit.kind in selected_kinds
        if unit_selected and kind_selected:
            unit.review_status = action
    _write_json(raw_path, document_to_dict(document))
    approved = IngestedDocument(
        document_id=document.document_id,
        course_id=document.course_id,
        source_id=document.source_id,
        source_file=document.source_file,
        source_sha256=document.source_sha256,
        units=[
            unit
            for unit in document.units
            if unit.review_status == ReviewStatus.APPROVED
        ],
        schema_version=document.schema_version,
        created_at=document.created_at,
    )
    processed_path = Path(course_dir) / "processed" / f"{document_id}.json"
    _write_json(processed_path, document_to_dict(approved))
    return processed_path


def indexable_units(
    courses_dir: Path, course_id: Optional[str] = None, include_duplicates: bool = False
) -> List[ContentUnit]:
    units: List[ContentUnit] = []
    course_dirs = (
        [Path(courses_dir) / course_id]
        if course_id
        else sorted(path for path in Path(courses_dir).iterdir() if path.is_dir())
    )
    for course_dir in course_dirs:
        for path in sorted((course_dir / "processed").glob("*.json")):
            for unit in read_document(path).units:
                if unit.review_status != ReviewStatus.APPROVED:
                    continue
                if unit.near_duplicate_of and not include_duplicates:
                    continue
                units.append(unit)
    return units


def export_jsonl(
    courses_dir: Path,
    output: Path,
    course_id: Optional[str] = None,
    include_duplicates: bool = False,
) -> int:
    units = indexable_units(courses_dir, course_id, include_duplicates)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(document_to_dict(
                IngestedDocument(
                    document_id=unit.document_id,
                    course_id=unit.course_id,
                    source_id="",
                    source_file=unit.provenance.source_file,
                    source_sha256=unit.provenance.source_sha256,
                    units=[unit],
                )
            )["units"][0], ensure_ascii=False) + "\n"
            for unit in units
        ),
        encoding="utf-8",
    )
    return len(units)
