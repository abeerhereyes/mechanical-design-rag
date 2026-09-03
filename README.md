# Multi-Subject Course RAG

A local study assistant that answers from the selected course's own notes,
shows exact page citations, separates outside knowledge, and uses reviewed
Python formulas for numerical work.

Included courses:

- Mechanical Design
- Aerodynamics
- Quality, Reliability, and Maintenance (QRM)

The design is intentionally course-first. A response may include broader
knowledge, but it is displayed separately and never presented as if it came
from the uploaded notes.

## Why this exists

General-purpose LLMs may choose a different convention, equation, or method
from the one taught in a course. This project reduces that risk through:

- mandatory course selection and course-filtered retrieval;
- document, page, and section provenance;
- explicit warnings when course notation differs from a common convention;
- calculation tools that execute only after their formulas are reviewed;
- refusal or clarification when evidence or parameters are insufficient.

This is still a study aid. Verify safety-critical decisions and graded work
against the original notes, standards, and instructor guidance.

## Architecture

```text
Student question + selected course
                 |
            LangGraph router
          /                    \
 reviewed calculation      document question
          |                      |
 typed parameters       Chroma + BM25 by course
 unit validation             -> RRF -> rerank
 Python formula                    |
          \                 grounded Ollama answer
           \______________________/
               page citations
```

PDF ingestion is a separate review-gated pipeline:

```text
PDF -> native layout + page render -> optional OCR + local vision
    -> raw JSON/assets -> developer review -> approved content -> index
```

Raw OCR or vision output is never automatically calculation-safe.

## Requirements

- Python 3.9 or newer; Python 3.11 is recommended
- Ollama for generated document answers
- Optional: Tesseract for weak/image-only text
- Optional: an Ollama vision model for diagrams, charts, and rasterized slides

## Install

```bash
cd /Users/abeer/Downloads/mech_rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install and start the text model:

```bash
brew install ollama
brew services start ollama
ollama pull llama3.2:3b
```

If you prefer a foreground server, use `ollama serve` instead of the service.
Do not run both at the same time.

Optional OCR and local vision setup:

```bash
python -m pip install -e ".[ocr]"
brew install tesseract
ollama pull llava:latest
```

## Run the student web app

```bash
source .venv/bin/activate
uvicorn src.api:app --reload
```

Open:

- Student chat: http://127.0.0.1:8000/
- API documentation: http://127.0.0.1:8000/docs

The web app provides a course selector, browser-local chat history, separate
course/general answers, source-page previews, calculation details, assumptions,
and discrepancy warnings.

## Run the CLI

```bash
mech-rag --course aerodynamics \
  "Explain how circulation creates lift in the supplied notes"

mech-rag --course qrm \
  "Explain producer risk and consumer risk in acceptance sampling"

mech-rag --course mechanical-design \
  "Calculate spring rate for G=79300 MPa, wire 3 mm, mean D 25 mm, 8 active coils"
```

Use `--no-general` to return only the course-grounded answer.

## API examples

Document question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "course_id": "qrm",
    "query": "What is an operating characteristic curve?",
    "include_general": true
  }'
```

Exact reviewed formula execution can be selected with `formula_id`, `params`,
and canonical `units`:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "course_id": "qrm",
    "query": "Calculate the average run length",
    "formula_id": "qrm.average_run_length",
    "params": {"signal_probability": 0.0027}
  }'
```

Useful endpoints:

- `GET /courses`
- `GET /ingestion/status`
- `GET /calculations?course_id=qrm`
- `GET /calculations/{formula_id}/schema`
- `GET /sources/{filename}/pages/{page}`

## Course data

```text
data/courses/
  mechanical-design/
    manifest.json
  aerodynamics/
    manifest.json
    sources/
    raw/
    processed/
  qrm/
    manifest.json
    sources/
    raw/
    processed/
```

`raw/` contains extraction results and page assets. `processed/` contains only
units explicitly approved for indexing. The current PDF notes are included
with permission.

## Ingest and review PDFs

List courses:

```bash
python -m src.ingestion courses
```

Extract a whole course:

```bash
python -m src.ingestion ingest qrm
```

Use OCR and local visual descriptions where needed:

```bash
python -m src.ingestion ingest aerodynamics \
  --ocr --vision --render-dpi 300 --vision-model llava:latest
```

Check review status:

```bash
python -m src.ingestion status aerodynamics
```

Inspect `data/courses/<course>/raw/` and its page assets before approval.
Approval can be limited to safe content kinds:

```bash
python -m src.ingestion approve aerodynamics DOCUMENT_ID --kind page_text
python -m src.ingestion approve aerodynamics DOCUMENT_ID --kind visual_caption
```

Formula and table units should be checked against their page image before
approval. Vision captions start as `unreviewed`.

Export approved, deduplicated units:

```bash
python -m src.ingestion export approved.jsonl
```

## Add another course

1. Create `data/courses/<course-id>/manifest.json`.
2. Put PDFs in `data/courses/<course-id>/sources/`.
3. Add each source to the manifest.
4. Run ingestion with OCR/vision as appropriate.
5. Review and approve safe units.
6. Add reviewed formulas under `src/calculations/` when numerical tools are
   required.
7. Add retrieval, citation, refusal, and calculation evaluation cases.

The retrieval architecture does not need to change for each new course.

## Calculation registry

The project includes 26 reviewed formulas across the three courses. Each
definition stores:

- course and formula IDs;
- equation and source document/page;
- typed variables, canonical units, and output units;
- assumptions and validation bounds;
- review state and source-convention warnings.

Only `reviewed` formulas execute. Call the registry directly with:

```python
from src.calculations import execute, get_formula, list_formulas

result = execute(
    "qrm.average_run_length",
    {"signal_probability": 0.0027},
)
```

## Test and evaluate

```bash
pytest
python eval/run_eval.py
python eval/run_multicourse_eval.py
```

Tests cover formula review gates and units, course isolation, duplicate
suppression, grounded API fields, source-page previews, and the web interface.
The original 25-question Mechanical Design retrieval benchmark remains
available for regression testing.

## Configuration

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.2:3b
export OLLAMA_VISION_MODEL=llava:latest
export OLLAMA_TIMEOUT_SECONDS=120
```

The application does not load `.env` automatically; export variables in your
shell or configure your process manager.

## Main modules

```text
src/ingestion/           PDF extraction, OCR/vision, review and deduplication
src/calculations/        Reviewed multi-course formula registry
src/chunker.py           Approved-content and legacy Markdown loaders
src/dense_retriever.py   Persistent content-hashed Chroma index
src/sparse_retriever.py  Course-filtered BM25
src/hybrid_retriever.py  Reciprocal-rank fusion
src/reranker.py          Candidate reranking and duplicate suppression
src/agent.py             Course-aware routing, clarification and execution
src/llm.py               Structured grounded Ollama generation
src/api.py               FastAPI, metadata and exact-source endpoints
src/web/                 Browser chat UI
```

## Current limitations

- Aerodynamics is a raster-heavy slide deck. Its 17 native-text pages are
  indexed; remaining visual pages require OCR/vision review for complete
  conceptual coverage.
- Native QRM page text is indexed, but calculation-critical tables, plots, and
  extracted equation objects remain review-gated.
- The included local hash embedder and lexical reranker are lightweight. They
  can later be replaced by local pretrained retrieval models without changing
  course filtering or provenance.
- There are no accounts or server-side chat histories in this local release.
