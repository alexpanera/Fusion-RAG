# ragbook: Local PDF RAG (Ollama + Hybrid Retrieval)

A local-first RAG pipeline for PDF documents — ask questions about your papers and books with fully private, offline inference:
- PDF ingestion with section-aware chunking
- Hybrid retrieval (FAISS dense + BM25 sparse)
- Grounded answers via local Ollama LLM
- Citations in `[p.X-Y | chunk_####]` format
- Evaluation harness to CSV

---

## Requirements

- Python 3.10+
- Windows / Linux / macOS
- Internet connection (first run only, to download the embedding model ~130 MB)

---

## Step 1 — Install Ollama

Ollama is the local LLM server that runs models on your machine.

### 1a. Install the Ollama app

Download and run the installer from **https://ollama.com/download**.

After installing, the Ollama service starts automatically in the background (look for the llama icon in the system tray on Windows). Verify it is running by opening `http://localhost:11434` in your browser — it should say `Ollama is running`.

> **Note:** `pip install ollama` installs only a Python client library — it does NOT install the Ollama server. You must install the app from the link above.

### 1b. Pull a model

**Recommended: qwen2.5:7b** (best quality for local CPU inference)

If `ollama` is available in your terminal:
```bash
ollama pull qwen2.5:7b
```

If the `ollama` command is not found (but the server is running), pull via Python instead:
```bash
python -c "
import requests, json
resp = requests.post('http://localhost:11434/api/pull', json={'name': 'qwen2.5:7b'}, stream=True, timeout=600)
for line in resp.iter_lines():
    if line:
        data = json.loads(line)
        status = data.get('status', '')
        if 'total' in data:
            pct = int(data.get('completed', 0) / data['total'] * 100)
            print(f'\r{status}: {pct}%', end='', flush=True)
        else:
            print(status)
print('Done')
"
```

Supported models (auto-selected in preference order):

| Model | Size | Notes |
|-------|------|-------|
| `qwen2.5:7b` | ~5 GB | **Recommended** — best quality on CPU |
| `qwen2.5:14b` | ~9 GB | Higher quality, needs 12+ GB RAM |
| `llama3.1:8b` | ~5 GB | Good alternative |
| `mistral:7b` | ~4 GB | Good alternative |
| `qwen2.5:3b` | ~2 GB | Lighter, lower quality |

---

## Step 2 — Set up Python environment

```bash
python -m venv .venv
```

Activate it:
```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

> The first run also downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) from HuggingFace automatically and caches it locally.

---

## Step 3 — Add your PDFs

Place your text-based `.pdf` files anywhere you like (e.g. `data/`). A pre-built sample index is included at `data/index/` so you can skip straight to asking questions.

> **Scanned PDFs** (image-only, no selectable text) will not work. Run OCR first, or use a different PDF.

---

## Step 4 — Build an index

Skip this step to use the included sample index at `data/index/`.

```bash
# Single PDF
python -m ragbook ingest --pdf data/mybook.pdf --out data/index

# Multiple PDFs into one shared index
python -m ragbook ingest --pdf data/book1.pdf data/book2.pdf --out data/index
```

Output files in the index directory:
- `chunks.jsonl` — extracted text chunks with metadata
- `faiss.index` — dense vector index
- `bm25_tokens.pkl` — sparse BM25 index
- `emb_cache.db` — embedding cache (speeds up re-indexing)
- `meta.json` — index metadata

---

## Step 5 — Ask questions

```bash
python -m ragbook ask --index data/index --q "What is plasma confinement?"
```

Answers include inline citations (`[p.X-Y | chunk_####]`) and a source list.

### Useful flags

```bash
# Retrieve more context chunks (default: 2)
python -m ragbook ask --index data/index --q "What is a tokamak?" --top_k 4

# Debug: show retrieved passages without calling the LLM
python -m ragbook ask --index data/index --q "What is a tokamak?" --retrieval_only

# Use a specific Ollama model
python -m ragbook ask --index data/index --q "What is tritium?" --ollama_model qwen2.5:7b

# Verbose logging
python -m ragbook ask --index data/index --q "What is tritium?" --log-level DEBUG
```

---

## Evaluate retrieval quality (optional)

Create `data/eval.jsonl` with one question per line:
```json
{"question": "What is beta_N?", "expected_keywords": ["beta", "normalization"]}
{"question": "What is a tokamak?", "expected_keywords": ["tokamak", "magnetic"]}
```

Run the eval:
```bash
python -m ragbook eval --index data/index --eval data/eval.jsonl --out results.csv
```

Output CSV columns:
- `retrieval_hit_at_k` — 1 if any retrieved chunk contains an expected keyword
- `answer_contains_keyword` — 1 if the LLM answer contains an expected keyword

---

## Environment variables

All optional. Set in your shell or a `.env` file before running.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | auto-selected | Override which model to use |
| `OLLAMA_TIMEOUT_SEC` | `600` | Request timeout in seconds |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model override |
| `RAG_TOP_K` | `2` | Default number of chunks to retrieve |
| `RAG_MAX_CONTEXT_CHARS` | `3000` | Max characters of context sent to the LLM |

> **Tuning tip:** `RAG_TOP_K=2` and `RAG_MAX_CONTEXT_CHARS=3000` are tuned for qwen2.5:7b on CPU. If you have a GPU or use a larger context window, raise both values (e.g. `TOP_K=6`, `MAX_CONTEXT_CHARS=8000`).

---

## Troubleshooting

**`ollama` command not found** — The app is installed but not in your PATH. Use the Python pull snippet in Step 1b instead.

**`Ollama is not running` error** — Open the Ollama app from your Start Menu. Check `http://localhost:11434` in your browser to confirm.

**`No chunks were created` error** — Your PDF is likely scanned (image-only). Try selecting text in a PDF viewer — if you can't select anything, it needs OCR preprocessing first.

**"Not enough evidence" answers** — The question may not match your document's content. Try `--retrieval_only` to see what passages were retrieved. If retrieval looks correct but the model still fails, try a larger model (`qwen2.5:7b`).

**Slow answers on CPU** — The 7B model on CPU takes 30–120 seconds per answer. Use `qwen2.5:3b` for faster (lower quality) responses, or run on a machine with a GPU.

**First run is slow** — The embedding model (~130 MB) downloads once on first use and is cached locally afterwards.

---

## How it works

```
PDF → text extraction (PyMuPDF / pdfplumber)
    → section-aware chunking (heading detection + overlap)
    → embeddings (BAAI/bge-small-en-v1.5, local)
    → FAISS dense index + BM25 sparse index

Question → embed query → hybrid retrieval (65% dense + 35% sparse)
         → top-k chunks → prompt assembly
         → Ollama LLM (local) → answer with citations
```

- PDF extraction uses PyMuPDF first, falls back to pdfplumber if extraction is sparse.
- Repeated headers/footers are auto-detected and removed before chunking.
- Unicode math symbols (σ, α, ×) are normalized to ASCII for reliable LLM reasoning.
- Embeddings are cached on disk — re-indexing the same PDF is much faster the second time.
- All inference is fully local with no paid APIs or data sharing.
