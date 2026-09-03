# Mechanical Design RAG

A portfolio-grade assistant for mechanical-design questions about bolts,
welds, springs, and bearings. It routes each question to either:

- a deterministic Python engineering calculation, or
- hybrid document retrieval followed by a grounded Ollama answer with citations.

Calculations never rely on the language model for arithmetic.

## Architecture

```text
Question
   |
LangGraph router
   |-- calculation -> parameter extraction -> validate -> Python formula
   |
   `-- lookup -> Chroma + BM25 -> RRF -> lexical reranker
                                      -> Ollama generation -> citations
```

The Chroma index is built automatically on the first document question and is
stored under `data/chroma/`. The CLI and API reuse it on later requests.

## Features

- Section-aware Markdown chunking with source, section, and page metadata
- Local dense retrieval plus BM25 sparse retrieval
- Reciprocal-rank fusion and candidate reranking
- Grounded local generation through Ollama
- Seven calculation routes:
  - metric tensile stress area
  - bolt preload
  - bolt joint load sharing
  - fillet-weld direct shear
  - compression-spring rate
  - spring maximum shear stress
  - bearing L10 life
- Missing-parameter clarification and physical input validation
- Interactive/one-shot CLI and FastAPI API
- 25-query retrieval evaluation set and automated unit/API tests

## Requirements

- Python 3.9 or newer
- [Ollama](https://ollama.com/) for document-answer generation

Calculations can run without Ollama. Lookup questions need the Ollama server.

## Install on macOS

```bash
cd /Users/abeer/Downloads/mech_rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install Ollama and download the default model:

```bash
brew install ollama
ollama pull llama3.2:3b
ollama serve
```

If the Ollama desktop application is already running, do not run a second
`ollama serve` process.

## Run the CLI

Interactive:

```bash
source .venv/bin/activate
mech-rag
```

One question:

```bash
mech-rag "When should a compression spring be checked for buckling?"
```

One calculation:

```bash
mech-rag "Spring rate with G=79300 MPa, wire 3 mm, mean D 25 mm, 8 active coils"
```

The calculation returns approximately `6.423 N/mm`.

## Run the API

```bash
source .venv/bin/activate
uvicorn src.api:app --reload
```

Open the interactive API documentation at:

- http://127.0.0.1:8000/docs

Example request:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"How does bolt preload improve fatigue life?"}'
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Configuration

Environment variables:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.2:3b
export OLLAMA_TIMEOUT_SECONDS=120
```

Defaults are also listed in `.env.example`. The project does not automatically
load `.env`; export values in the shell or use your process manager.

## Test and evaluate

```bash
pytest
python eval/run_eval.py
```

The evaluation compares dense-only, hybrid, and hybrid-plus-reranked retrieval
using Precision@3, Recall@3, and MRR.

## Project layout

```text
data/corpus/             Mechanical-design source notes
src/chunker.py           Section-aware corpus parser
src/dense_retriever.py   Persistent Chroma index
src/sparse_retriever.py  BM25 index
src/hybrid_retriever.py  Reciprocal-rank fusion
src/reranker.py          Candidate reranking
src/calc_tools.py        Validated engineering formulas
src/llm.py               Ollama HTTP client and grounded prompt
src/agent.py             LangGraph workflow and parameter extraction
src/cli.py               Command-line interface
src/api.py               FastAPI application
eval/                    Labeled retrieval evaluation
tests/                   Formula, routing, and API tests
```

## Scope and limitations

- The included corpus is intentionally small and educational; calculations
  should be independently reviewed before safety-critical use.
- Parameter extraction is deterministic and supports common labels and ordered
  numeric input. The API accepts explicit `params` when exact control is needed.
- The bundled dense embedder and reranker are lightweight local lexical models,
  not pretrained semantic models. Their interfaces can later be replaced with
  Sentence Transformers without changing the agent workflow.
