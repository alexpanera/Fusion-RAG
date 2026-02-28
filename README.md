# ragbook: Local Fusion Textbook RAG (Ollama + Hybrid Retrieval)

A free, local-first MVP RAG pipeline for textbook PDFs:
- PDF ingestion with page-aware extraction
- Section/paragraph chunking
- Hybrid retrieval (FAISS dense + BM25 sparse)
- Grounded answers via local Ollama LLM
- Citations in `[p.X-Y | chunk_####]` format
- Evaluation harness to CSV

## Requirements

- Python 3.10+
- `pip` (no conda)
- Windows/Linux/macOS
- Ollama running locally

## Input PDFs

- This repo does not guarantee any bundled source PDFs for your use case.
- Upload your own readable, text-based `.pdf` files into the repo, for example under `data/`.
- Then run `ingest` to build your own local index before asking questions.
- Scanned image-only PDFs may need OCR first; this project works best with selectable text PDFs.

## 1) Install Ollama + model

Install Ollama: https://ollama.com/download

Then pull at least one model:

```bash
ollama pull qwen2.5:3b
```

Fallbacks:

```bash
ollama pull qwen:0.5b
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b
ollama pull llama3.1:8b
ollama pull mistral:7b
```

Model auto-selection order:
1. `qwen2.5:3b`
2. `qwen:0.5b`
3. `qwen2.5:0.5b`
4. `qwen2.5:14b`
5. `qwen2.5:7b`
6. `llama3.1:8b`
7. `mistral:7b`

You can override with `OLLAMA_MODEL`.

## 2) Install Python deps

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3) CLI usage

### Ingest + index (single PDF)

```bash
python -m ragbook ingest --pdf path/to/book.pdf --out data/index
```

### Ingest + index (multiple PDFs into one index)

```bash
python -m ragbook ingest --pdf path/to/book1.pdf path/to/book2.pdf --out data/index
```

Artifacts in `data/index/`:
- `chunks.jsonl`
- `faiss.index`
- `bm25_tokens.pkl`
- `emb_cache.db*`
- `meta.json`

### Ask

```bash
python -m ragbook ask --index data/index --q "What is beta_N?" --top_k 1
```

### Retrieval-only debug

```bash
python -m ragbook ask --index data/index --q "What is beta_N?" --retrieval_only
```

### Eval

Input JSONL lines like:

```json
{"question":"What is beta_N?","expected_keywords":["beta","normalization"]}
```

Run:

```bash
python -m ragbook eval --index data/index --eval data/eval.jsonl --out results.csv
```

Metrics:
- `retrieval_hit_at_k`: any retrieved chunk contains any expected keyword
- `answer_contains_keyword`: answer contains any expected keyword

## Environment variables

- `OLLAMA_HOST` (default `http://localhost:11434`)
- `OLLAMA_MODEL` (optional model override)
- `OLLAMA_TIMEOUT_SEC` (optional Ollama request timeout, default `600`)
- `EMBED_MODEL` (optional embedding model override; default `BAAI/bge-small-en-v1.5`)
- `RAG_TOP_K` (optional default top-k, default `6`)
- `RAG_MAX_CONTEXT_CHARS` (optional prompt context cap, default `2000`)

## Notes

- PDF extraction uses PyMuPDF first; if it fails/sparse, falls back to pdfplumber.
- Header/footer cleanup removes lines repeated across many pages.
- Chunking uses token estimate heuristic: `len(text)/4`.
- Embeddings are cached on disk to speed re-indexing.
- For smaller Ollama models, `--top_k 1` or `--top_k 2` usually gives better latency and fewer prompt issues.
- All inference is local (no paid APIs).
