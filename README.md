# ragbook: Local Fusion Textbook RAG (Ollama + Hybrid Retrieval)

A free, local-first MVP RAG pipeline for textbook PDFs:
- PDF ingestion with page-aware extraction
- Section/paragraph chunking
- Hybrid retrieval (FAISS dense + BM25 sparse)
- Grounded answers via local Ollama LLM
- Citations in `[p.X-Y | chunk_####]` format
- Evaluation harness to CSV

---

## Requirements

- Python 3.10+
- Windows / Linux / macOS
- Internet connection (first run only, to download the embedding model)

---

## Step 1 — Install Ollama

Ollama is the local LLM server that runs models on your machine. There are two things to install: the **Ollama app** and a **model**.

### 1a. Install the Ollama app

Download and run the installer from **https://ollama.com/download**.

After installing, the Ollama service starts automatically in the background (look for the llama icon in the system tray on Windows). You can verify it is running by opening your browser at `http://localhost:11434` — it should say `Ollama is running`.

> **Note:** `pip install ollama` installs only a Python client library — it does NOT install the Ollama server. You must install the app from the link above.

### 1b. Pull a model

If `ollama` is available in your terminal:
```bash
ollama pull qwen2.5:3b
```

If the `ollama` command is not found in your terminal (but the server is running), pull via Python instead:
```bash
python -c "
import requests, json
resp = requests.post('http://localhost:11434/api/pull', json={'name': 'qwen2.5:3b'}, stream=True, timeout=600)
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

This downloads ~2 GB. Other supported models (auto-selected by preference order):

| Model | Size |
|-------|------|
| `qwen2.5:3b` | ~2 GB (recommended) |
| `qwen:0.5b` | ~400 MB |
| `qwen2.5:0.5b` | ~400 MB |
| `qwen2.5:7b` | ~5 GB |
| `qwen2.5:14b` | ~9 GB |
| `llama3.1:8b` | ~5 GB |
| `mistral:7b` | ~4 GB |

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

> The first run will also download the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) from HuggingFace automatically.

---

## Step 3 — Add your PDFs

Place your text-based `.pdf` files anywhere you like (e.g. `data/`). This repo includes two sample PDFs under `data/` and a pre-built index under `data/index/` so you can skip straight to asking questions.

> **Scanned PDFs** (image-only, no selectable text) will not work. Run OCR on them first, or use a different PDF.

---

## Step 4 — Build an index

Skip this if you want to use the pre-built sample index at `data/index/`.

```bash
# Single PDF
python -m ragbook ingest --pdf data/fusionenergy.pdf --out data/index

# Multiple PDFs into one index
python -m ragbook ingest --pdf data/book1.pdf data/book2.pdf --out data/index
```

This creates the following files in the output directory:
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

The answer is printed with inline citations (`[p.X-Y | chunk_####]`) and a source list at the end.

### Useful flags

```bash
# Return more context chunks (default is 6)
python -m ragbook ask --index data/index --q "What is a tokamak?" --top_k 3

# Debug: show retrieved passages without calling the LLM
python -m ragbook ask --index data/index --q "What is a tokamak?" --retrieval_only

# Use a specific Ollama model
python -m ragbook ask --index data/index --q "What is tritium?" --ollama_model qwen2.5:7b

# Verbose logging
python -m ragbook ask --index data/index --q "What is tritium?" --log-level DEBUG
```

---

## Evaluate retrieval quality (optional)

Create an eval file (`data/eval.jsonl`) with one question per line:
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

All optional. Set them in your shell or a `.env` file before running.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | auto-selected | Override which model to use |
| `OLLAMA_TIMEOUT_SEC` | `600` | Request timeout in seconds |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model override |
| `RAG_TOP_K` | `6` | Default number of chunks to retrieve |
| `RAG_MAX_CONTEXT_CHARS` | `6000` | Max characters of context sent to the LLM |

---

## Troubleshooting

**`ollama` command not found** — The Ollama app is installed but not in your PATH. Use the Python pull snippet in Step 1b, or find `ollama.exe` in your Start Menu, right-click → Open file location, and run it from there.

**`Ollama is not running` error** — Open the Ollama app from your Start Menu / Applications. Check `http://localhost:11434` in your browser to confirm it is running.

**`No chunks were created` error** — Your PDF is likely scanned (image-only). Open it in a PDF viewer and try selecting text — if you can't select any text, it needs OCR preprocessing first.

**Slow answers** — Use a smaller model (`qwen2.5:0.5b`) or reduce context with `--top_k 1`.

**First run is slow** — The embedding model (~130 MB) is downloaded once on first use and cached locally afterwards.

---

## Notes

- PDF extraction uses PyMuPDF first; falls back to pdfplumber if extraction is sparse or fails.
- Repeated headers and footers are automatically detected and removed before chunking.
- Embeddings are cached on disk — re-indexing the same PDF is much faster the second time.
- All inference is fully local with no paid APIs required.
