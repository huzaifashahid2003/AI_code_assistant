# AI Code Review Assistant

An AI-powered code review tool that analyses Python source files and produces
a structured, section-by-section review report using a Retrieval-Augmented
Generation (RAG) pipeline backed by Google Gemini.

---

## Features

- **Upload any `.py` file** or use the built-in sample to try it instantly
- **Structured AI review** with four dedicated sections:
  - Code Quality Issues
  - Performance Improvements
  - Naming & Style Suggestions
  - Potential Bugs
- **Line-level referencing** — suggestions cite exact function/class names and line ranges
- **Download report** — export the review as a `.txt` file from the UI
- **RAG pipeline** — relevant code chunks are retrieved via FAISS before being sent to the LLM
- **Graceful fallback** — runs in placeholder mode when no API key is configured
- **Structured error handling** — empty files, invalid types, encoding errors, and API failures all return clear messages
- **Server-side logging** — every upload, chunk count, and review generation is logged

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Code Parsing | Python `ast` (built-in) |
| Embeddings & Vector Search | FAISS (`faiss-cpu`) + NumPy |
| LLM | Google Gemini 1.5 Flash (`google-generativeai`) |
| HTTP client (frontend) | `requests` |
| Validation | Pydantic v2 |

---

## Project Structure

```
ai_code_review/
├── backend/
│   ├── main.py             # FastAPI server: /upload and /review endpoints
│   ├── code_processing.py  # AST-based code chunking (CodeChunk dataclass)
│   └── rag_search.py       # FAISS index, Gemini embeddings & review generation
├── frontend/
│   └── app.py              # Streamlit UI
├── data/
│   └── uploads/            # Uploaded files stored here (auto-created)
├── samples/
│   └── sample_code.py      # Demo file with intentional issues for testing
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- `pip`
- *(Optional)* A [Google AI Studio](https://aistudio.google.com/) API key for real Gemini reviews

### 1. Install dependencies

```bash
cd ai_code_review
pip install -r requirements.txt
```

### 2. Set the Gemini API key *(optional but recommended)*

**Windows (PowerShell)**
```powershell
$env:GEMINI_API_KEY = "your_api_key_here"
```

**macOS / Linux (bash)**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

> If the key is not set the app still runs, returning a structured placeholder
> review instead of a real AI response.

---

## How to Run

> Open **two separate terminals** inside the `ai_code_review/` folder.

### Terminal 1 — Start the FastAPI backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Interactive API docs: <http://localhost:8000/docs>

### Terminal 2 — Start the Streamlit frontend

```bash
cd frontend
streamlit run app.py
```

The UI opens automatically at <http://localhost:8501>.

---

## Example Usage

1. Open the Streamlit app at <http://localhost:8501>.
2. Upload `samples/sample_code.py` (or tick **Use built-in sample code**).
3. Leave the default query or type your own (e.g. *"Find all potential bugs"*).
4. Click **Analyze Code**.
5. Read the structured review rendered on the page.
6. Click **Download Review (.txt)** to save the report.

### Example review output

```
## 1. Code Quality Issues
- **`read_config` (Lines 72-75):** File is opened without a context manager.
  *Suggestion:* Use `with open(filepath) as f:` to ensure the file is always closed.

## 2. Performance Improvements
- **`sum_of_squares` (Lines 31-38):** Builds an intermediate list before summing.
  *Suggestion:* Replace with `return sum(n * n for n in numbers)`.

## 3. Naming & Style Suggestions
- **`userAccount` (Lines 49-67):** Class name violates PascalCase convention.
  *Suggestion:* Rename to `UserAccount`.

## 4. Potential Bugs
- **`divide` (Lines 22-24):** No guard against division by zero.
  *Suggestion:* Add `if b == 0: raise ValueError("Divisor cannot be zero.")` before the division.
```

---

## API Reference

| Method | Endpoint  | Description                          |
|--------|-----------|--------------------------------------|
| GET    | `/`       | Health check                         |
| POST   | `/upload` | Upload a `.py` file                  |
| POST   | `/review` | Run an AI review on an uploaded file |

### `POST /upload` — request / response

```bash
curl -X POST http://localhost:8000/upload \
     -F "file=@samples/sample_code.py"
```

```json
{
  "filename": "sample_code.py",
  "saved_to": "data/uploads/sample_code.py",
  "status": "uploaded successfully"
}
```

### `POST /review` — request / response

```bash
curl -X POST http://localhost:8000/review \
     -H "Content-Type: application/json" \
     -d '{"filename": "sample_code.py", "query": "Find all bugs and style issues"}'
```

```json
{
  "filename": "sample_code.py",
  "query": "Find all bugs and style issues",
  "total_chunks": 8,
  "reviewed_chunks": 3,
  "review": "## 1. Code Quality Issues ...\n"
}
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| *"Cannot reach the backend"* | Start the FastAPI server first (`uvicorn main:app --reload`) |
| *"Only Python (.py) files are accepted"* | Ensure you are uploading a `.py` file |
| *"No top-level functions or classes found"* | The file must contain at least one `def` or `class` |
| Placeholder review shown | Set `GEMINI_API_KEY` and install `google-generativeai` |
| `faiss-cpu` install fails on Windows | Use `pip install faiss-cpu --only-binary :all:` |

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
