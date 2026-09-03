"""PDF extraction, rendering, optional OCR, and optional vision captioning."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cleanup import remove_recurring_margins
from .duplicates import content_hash
from .schemas import (
    ContentKind,
    ContentUnit,
    IngestedDocument,
    Locator,
    Provenance,
    ReviewStatus,
)


@dataclass
class ExtractionConfig:
    render_dpi: int = 160
    ocr: bool = False
    ocr_min_chars: int = 40
    vision: bool = False
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    vision_model: str = os.getenv("OLLAMA_VISION_MODEL", "llava:latest")
    timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
    header_footer_frequency: float = 0.6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fitz():
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires PyMuPDF. Install it with: "
            "python -m pip install PyMuPDF"
        ) from exc
    return fitz


def _ocr_png(png: bytes) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OCR requires pytesseract and Pillow. Install with: "
            "python -m pip install pytesseract Pillow. The system Tesseract "
            "binary must also be installed."
        ) from exc
    import io

    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(png))).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "OCR was enabled but the Tesseract executable was not found"
        ) from exc


def _caption_png(png: bytes, config: ExtractionConfig) -> str:
    prompt = (
        "Describe the engineering visual content on this page. Transcribe "
        "labels, equations, plot axes, and table meaning faithfully. Do not "
        "invent unreadable values."
    )
    payload = json.dumps(
        {
            "model": config.vision_model,
            "prompt": prompt,
            "images": [base64.b64encode(png).decode("ascii")],
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        config.ollama_base_url.rstrip("/") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Ollama vision request failed at {config.ollama_base_url}: {exc}"
        ) from exc
    caption = str(result.get("response", "")).strip()
    if not caption:
        raise RuntimeError("Ollama vision returned an empty caption")
    return caption


def _looks_like_formula(text: str) -> bool:
    compact = " ".join(text.split())
    if not compact or len(compact) > 500:
        return False
    operators = len(re.findall(r"[=±×÷√∑∫^]|(?:<=|>=)", compact))
    variables = len(re.findall(r"\b[A-Za-z](?:_[A-Za-z0-9]+)?\b", compact))
    return operators >= 1 and variables >= 1


def _table_text(table: Any) -> str:
    rows = table.extract()
    return "\n".join(
        " | ".join("" if cell is None else str(cell).strip() for cell in row)
        for row in rows
    ).strip()


def extract_pdf(
    pdf_path: Path,
    course_id: str,
    source_id: str,
    output_assets_dir: Path,
    config: Optional[ExtractionConfig] = None,
) -> IngestedDocument:
    config = config or ExtractionConfig()
    fitz = _fitz()
    pdf_path, output_assets_dir = Path(pdf_path), Path(output_assets_dir)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF source not found: {pdf_path}")
    source_hash = sha256_file(pdf_path)
    document_id = f"{course_id}-{source_id}-{source_hash[:12]}"
    page_records: List[Dict[str, Any]] = []
    with fitz.open(pdf_path) as document:
        output_assets_dir.mkdir(parents=True, exist_ok=True)
        for index, page in enumerate(document):
            page_number = index + 1
            pixmap = page.get_pixmap(dpi=config.render_dpi, alpha=False)
            png = pixmap.tobytes("png")
            image_name = f"page-{page_number:04d}.png"
            (output_assets_dir / image_name).write_bytes(png)
            native_text = page.get_text("text", sort=True).strip()
            text = native_text
            extractor = "pymupdf"
            if config.ocr and len(native_text) < config.ocr_min_chars:
                ocr_text = _ocr_png(png)
                if len(ocr_text) > len(native_text):
                    text, extractor = ocr_text, "pytesseract"
            blocks = []
            formula_blocks = []
            for block in page.get_text("blocks", sort=True):
                if len(block) < 5 or not str(block[4]).strip():
                    continue
                item = {
                    "bbox": [round(float(value), 2) for value in block[:4]],
                    "text": str(block[4]).strip(),
                    "block_type": int(block[6]) if len(block) > 6 else 0,
                }
                blocks.append(item)
                if _looks_like_formula(item["text"]):
                    formula_blocks.append(item)
            tables = []
            if hasattr(page, "find_tables"):
                try:
                    tables = [
                        {
                            "bbox": [round(float(value), 2) for value in table.bbox],
                            "text": _table_text(table),
                        }
                        for table in page.find_tables().tables
                    ]
                except Exception:
                    tables = []
            visuals = []
            for visual_index, image in enumerate(page.get_images(full=True), 1):
                xref = image[0]
                rects = page.get_image_rects(xref)
                visuals.append(
                    {
                        "visual_index": visual_index,
                        "xref": xref,
                        "bboxes": [
                            [round(rect.x0, 2), round(rect.y0, 2),
                             round(rect.x1, 2), round(rect.y1, 2)]
                            for rect in rects
                        ],
                    }
                )
            caption = _caption_png(png, config) if config.vision else None
            page_records.append(
                {
                    "page": page_number,
                    "text": text,
                    "extractor": extractor,
                    "blocks": blocks,
                    "formulas": formula_blocks,
                    "tables": tables,
                    "visuals": visuals,
                    "caption": caption,
                    "render": image_name,
                }
            )
    cleaned = remove_recurring_margins(
        [record["text"] for record in page_records],
        frequency=config.header_footer_frequency,
    )
    units: List[ContentUnit] = []

    def add_unit(
        kind: ContentKind,
        text: str,
        page: int,
        extractor: str,
        status: ReviewStatus,
        suffix: str,
        metadata: Optional[Dict[str, Any]] = None,
        bbox: Optional[List[float]] = None,
        model: Optional[str] = None,
    ) -> None:
        if not text.strip() and kind not in {ContentKind.VISUAL}:
            return
        unit_id = f"{document_id}:p{page:04d}:{suffix}"
        units.append(
            ContentUnit(
                unit_id=unit_id,
                course_id=course_id,
                document_id=document_id,
                kind=kind,
                text=text.strip(),
                locator=Locator(page=page, slide=page, bbox=bbox),
                provenance=Provenance(
                    source_file=pdf_path.name,
                    source_sha256=source_hash,
                    extractor=extractor,
                    model=model,
                ),
                review_status=status,
                content_hash=content_hash(text),
                metadata=metadata or {},
            )
        )

    for record, page_text in zip(page_records, cleaned):
        page = record["page"]
        add_unit(
            ContentKind.PAGE_TEXT,
            page_text,
            page,
            record["extractor"],
            ReviewStatus.PENDING_REVIEW,
            "text",
            {"layout_blocks": record["blocks"], "render": record["render"]},
        )
        for index, formula in enumerate(record["formulas"], 1):
            add_unit(
                ContentKind.FORMULA,
                formula["text"],
                page,
                "pymupdf-layout-heuristic",
                ReviewStatus.PENDING_REVIEW,
                f"formula-{index:03d}",
                {"formula_index": index},
                formula["bbox"],
            )
        for index, table in enumerate(record["tables"], 1):
            add_unit(
                ContentKind.TABLE,
                table["text"],
                page,
                "pymupdf-table",
                ReviewStatus.PENDING_REVIEW,
                f"table-{index:03d}",
                {"table_index": index},
                table["bbox"],
            )
        for visual in record["visuals"]:
            bbox = visual["bboxes"][0] if visual["bboxes"] else None
            add_unit(
                ContentKind.VISUAL,
                "",
                page,
                "pymupdf-image",
                ReviewStatus.PENDING_REVIEW,
                f"visual-{visual['visual_index']:03d}",
                visual,
                bbox,
            )
        if record["caption"]:
            add_unit(
                ContentKind.VISUAL_CAPTION,
                record["caption"],
                page,
                "ollama-vision",
                ReviewStatus.UNREVIEWED,
                "vision-caption",
                {"render": record["render"], "raw_model_output": True},
                model=config.vision_model,
            )
    return IngestedDocument(
        document_id=document_id,
        course_id=course_id,
        source_id=source_id,
        source_file=pdf_path.name,
        source_sha256=source_hash,
        units=units,
    )
