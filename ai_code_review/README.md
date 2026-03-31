# AI Code Review Assistant

An internship-level project that lets users upload Python files and receive
AI-powered code reviews using a Retrieval-Augmented Generation (RAG) pipeline.

## Project Structure

```
ai_code_review/
├── backend/
│   ├── main.py             # FastAPI server  (upload endpoint)
│   ├── code_processing.py  # AST-based code chunking
│   └── rag_search.py       # FAISS index + LLM review (stubs)
├── frontend/
│   └── app.py              # Streamlit UI
├── data/
│   └── uploads/            # Saved user files
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
cd ai_code_review
pip install -r requirements.txt
```

### 2. Run the FastAPI backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Interactive API docs: <http://localhost:8000/docs>

### 3. Run the Streamlit frontend

Open a second terminal:

```bash
cd frontend
streamlit run app.py
```

## API Reference

| Method | Endpoint  | Description                        |
|--------|-----------|------------------------------------|
| GET    | `/`       | Health check                       |
| POST   | `/upload` | Upload a `.py` file for processing |

### `POST /upload` — example response

```json
{
  "filename": "my_module.py",
  "saved_to": "data/uploads/my_module.py",
  "status": "uploaded successfully"
}
```

## Module Overview

### `backend/code_processing.py`

| Symbol | Description |
|---|---|
| `CodeChunk` | Dataclass holding a single function/class chunk and its metadata |
| `chunk_code(code_text)` | Parse a source string → list of `CodeChunk` |
| `chunk_file(file_path)` | Read a `.py` file from disk, delegate to `chunk_code` |

### `backend/rag_search.py` (stubs — fill in as project grows)

| Symbol | Description |
|---|---|
| `embed_chunks(chunks)` | Embed `CodeChunk` list with sentence-transformers |
| `build_index(embeddings)` | Create a FAISS `IndexFlatL2` from embeddings |
| `save_index / load_index` | Persist/load the FAISS index |
| `search(query, index, chunks)` | Retrieve top-k relevant chunks |
| `review_with_llm(query, chunks)` | Call OpenAI and return review text |

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | When using LLM review | Your OpenAI API key |

## Roadmap

- [x] File upload endpoint
- [x] AST-based code chunking
- [ ] Embedding pipeline (sentence-transformers)
- [ ] FAISS index build/search
- [ ] LLM review integration (OpenAI)
- [ ] Streamlit Q&A panel wired to RAG
