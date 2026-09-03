import json

from src.ingestion.schemas import (
    ContentKind,
    ContentUnit,
    IngestedDocument,
    Locator,
    Provenance,
    ReviewStatus,
    document_to_dict,
)
from src.ingestion.workflow import indexable_units, read_document, review_document


def _unit(unit_id, kind):
    return ContentUnit(
        unit_id=unit_id,
        course_id="test-course",
        document_id="test-document",
        kind=kind,
        text="reviewable content",
        locator=Locator(page=1),
        provenance=Provenance(
            source_file="notes.pdf",
            source_sha256="abc",
            extractor="test",
        ),
        review_status=ReviewStatus.PENDING_REVIEW,
        content_hash=unit_id,
    )


def test_review_can_approve_text_without_approving_formula(tmp_path):
    course_dir = tmp_path / "test-course"
    raw = course_dir / "raw"
    raw.mkdir(parents=True)
    document = IngestedDocument(
        document_id="test-document",
        course_id="test-course",
        source_id="notes",
        source_file="notes.pdf",
        source_sha256="abc",
        units=[
            _unit("text", ContentKind.PAGE_TEXT),
            _unit("formula", ContentKind.FORMULA),
        ],
    )
    (raw / "test-document.json").write_text(
        json.dumps(document_to_dict(document)), encoding="utf-8"
    )

    processed = review_document(
        course_dir,
        "test-document",
        ReviewStatus.APPROVED,
        content_kinds=[ContentKind.PAGE_TEXT],
    )
    saved = read_document(processed)
    assert [unit.unit_id for unit in saved.units] == ["text"]
    assert [unit.unit_id for unit in indexable_units(tmp_path)] == ["text"]
