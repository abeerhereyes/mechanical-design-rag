"""
Phase 1: Corpus chunking.

Each corpus file is hand-structured into "## Section X (p. N): Title" blocks.
We chunk on those section boundaries (not fixed-size token windows) because:
  - Each section is already a self-contained unit of engineering reasoning
    (one formula + its context), so splitting mid-section would break the
    logical unit a question is usually asking about.
  - It gives us a clean, deterministic citation: doc + section id + page.
INTERVIEW NOTE: fixed-size chunking (e.g. 512 tokens w/ overlap) is the
default in most RAG tutorials, but it's a bad fit for structured technical
docs where formulas and their derivation context must stay together. Being
able to justify "why NOT the default chunking strategy" is a good signal.
"""
import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SECTION_RE = re.compile(
    r"^##\s+Section\s+([A-Za-z0-9]+)\s+\(p\.\s*(\d+)\)\s*:\s*(.+)$", re.MULTILINE
)


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    section: str
    page: int
    title: str
    metadata: dict = field(default_factory=dict)
    parent_id: Optional[str] = None
    content_type: str = "section"


def load_corpus(corpus_dir: str) -> list[Chunk]:
    chunks = []
    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(corpus_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        source_match = re.search(r"^Source:\s*(.+)$", text, re.MULTILINE)
        source_name = source_match.group(1).strip() if source_match else fname

        matches = list(SECTION_RE.finditer(text))
        for i, m in enumerate(matches):
            section_id, page, title = m.group(1), int(m.group(2)), m.group(3).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            chunk_id = f"{fname}::{section_id}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=f"{title}\n{body}",
                    source=source_name,
                    section=section_id,
                    page=page,
                    title=title,
                    metadata={
                        "file": fname,
                        "source": source_name,
                        "section": section_id,
                        "page": page,
                        "title": title,
                        "course_id": "mechanical-design",
                        "content_type": "section",
                    },
                    parent_id=fname,
                )
            )
    return chunks


def load_course_corpus(
    courses_dir: str,
    markdown_corpus_dir: Optional[str] = None,
    course_id: Optional[str] = None,
) -> list[Chunk]:
    """Load approved PDF units plus the legacy reviewed Mechanical Design notes."""
    from src.ingestion.workflow import indexable_units

    root = Path(courses_dir)
    chunks: list[Chunk] = []
    if course_id in {None, "mechanical-design"}:
        corpus = Path(markdown_corpus_dir) if markdown_corpus_dir else root.parent / "corpus"
        if corpus.exists():
            chunks.extend(load_corpus(str(corpus)))

    for unit in indexable_units(root, course_id=course_id):
        source = unit.provenance.source_file
        section = str(
            unit.metadata.get("section")
            or unit.metadata.get("topic")
            or f"page-{unit.locator.page}"
        )
        title = str(
            unit.metadata.get("title")
            or unit.metadata.get("topic")
            or f"{unit.kind.value.replace('_', ' ').title()} — page {unit.locator.page}"
        )
        metadata = {
            "file": source,
            "source": source,
            "section": section,
            "page": unit.locator.page,
            "slide": unit.locator.slide or unit.locator.page,
            "title": title,
            "course_id": unit.course_id,
            "document_id": unit.document_id,
            "content_type": unit.kind.value,
            "content_hash": unit.content_hash,
            "canonical_id": unit.near_duplicate_of or unit.unit_id,
            "review_status": unit.review_status.value,
        }
        for key in ("render", "formula_index", "table_index", "requires_visual_context"):
            value = unit.metadata.get(key)
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
        chunks.append(
            Chunk(
                id=unit.unit_id,
                text=unit.text,
                source=source,
                section=section,
                page=unit.locator.page,
                title=title,
                metadata=metadata,
                parent_id=unit.document_id,
                content_type=unit.kind.value,
            )
        )
    return chunks


if __name__ == "__main__":
    cks = load_corpus(os.path.join(os.path.dirname(__file__), "..", "data", "corpus"))
    print(f"Loaded {len(cks)} chunks")
    for c in cks[:3]:
        print(f"\n[{c.id}] {c.title}")
        print(c.text[:150], "...")
