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
                    },
                )
            )
    return chunks


if __name__ == "__main__":
    cks = load_corpus(os.path.join(os.path.dirname(__file__), "..", "data", "corpus"))
    print(f"Loaded {len(cks)} chunks")
    for c in cks[:3]:
        print(f"\n[{c.id}] {c.title}")
        print(c.text[:150], "...")
