"""Review-gated course PDF ingestion."""

from .manifest import CourseManifest, SourceSpec, discover_courses, load_manifest
from .pdf import ExtractionConfig, extract_pdf
from .schemas import ContentUnit, IngestedDocument, ReviewStatus
from .workflow import indexable_units

__all__ = [
    "ContentUnit",
    "CourseManifest",
    "ExtractionConfig",
    "IngestedDocument",
    "ReviewStatus",
    "SourceSpec",
    "discover_courses",
    "extract_pdf",
    "indexable_units",
    "load_manifest",
]
