"""Typed, JSON-serializable records used by the ingestion pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"


class ContentKind(str, Enum):
    PAGE_TEXT = "page_text"
    TEXT_BLOCK = "text_block"
    FORMULA = "formula"
    TABLE = "table"
    VISUAL = "visual"
    VISUAL_CAPTION = "visual_caption"


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Locator:
    page: int
    slide: Optional[int] = None
    bbox: Optional[List[float]] = None


@dataclass
class Provenance:
    source_file: str
    source_sha256: str
    extractor: str
    extracted_at: str = field(default_factory=utc_now)
    model: Optional[str] = None
    parent_unit_id: Optional[str] = None


@dataclass
class ContentUnit:
    unit_id: str
    course_id: str
    document_id: str
    kind: ContentKind
    text: str
    locator: Locator
    provenance: Provenance
    review_status: ReviewStatus
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    near_duplicate_of: Optional[str] = None


@dataclass
class IngestedDocument:
    document_id: str
    course_id: str
    source_id: str
    source_file: str
    source_sha256: str
    units: List[ContentUnit]
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)


def document_to_dict(document: IngestedDocument) -> Dict[str, Any]:
    return asdict(document)


def document_from_dict(value: Dict[str, Any]) -> IngestedDocument:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported ingestion schema {value.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    units = []
    for item in value.get("units", []):
        units.append(
            ContentUnit(
                unit_id=item["unit_id"],
                course_id=item["course_id"],
                document_id=item["document_id"],
                kind=ContentKind(item["kind"]),
                text=item["text"],
                locator=Locator(**item["locator"]),
                provenance=Provenance(**item["provenance"]),
                review_status=ReviewStatus(item["review_status"]),
                content_hash=item["content_hash"],
                metadata=item.get("metadata", {}),
                near_duplicate_of=item.get("near_duplicate_of"),
            )
        )
    return IngestedDocument(
        document_id=value["document_id"],
        course_id=value["course_id"],
        source_id=value["source_id"],
        source_file=value["source_file"],
        source_sha256=value["source_sha256"],
        units=units,
        schema_version=value["schema_version"],
        created_at=value["created_at"],
    )
