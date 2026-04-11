# AI Code Review Assistant

An AI-powered code review tool that analyses Python source files and produces a structured, section-by-section review report using a **Retrieval-Augmented Generation (RAG)** pipeline backed by **Groq (Llama 3.3 70B)**.

---

## Features

- **Upload any `.py` file** or use the built-in sample to try it instantly
- **Structured AI review** with four dedicated sections:
  - Code Quality Issues
  - Performance Improvements
  - Naming & Style Suggestions
  - Potential Bugs
- **Per-chunk structured analysis** — each function/class gets its own severity rating (`high` / `medium` / `low` / `none`), issue description, actionable suggestion, and exact problematic line numbers
- **AI-generated corrected code** — the backend returns a fully fixed version of the uploaded file
- **Line-level referencing** — suggestions cite exact function/class names and line ranges
- **Download report** — export the review as a `.txt` file from the UI
- **RAG pipeline** — relevant code chunks are retrieved via a pure-NumPy TF-IDF vector store before being sent to the LLM
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
| Embeddings & Vector Search | Pure-NumPy TF-IDF (`numpy`) |
| LLM | Groq — `llama-3.3-70b-versatile` |
| HTTP client (frontend) | `requests` |
| Validation | Pydantic v2 |
| Containerisation | Docker + Docker Compose |

---

## Project Structure

```
ai_code_review/
├── backend/
│   ├── main.py             # FastAPI server — /upload and /review endpoints
│   ├── code_processing.py  # AST-based code chunking (CodeChunk dataclass)
│   ├── analyzer.py         # Per-chunk LLM review (issue / severity / lines)
│   └── rag_search.py       # TF-IDF vector store, RAG retrieval & LLM calls
├── frontend/
│   └── app.py              # Streamlit UI
├── data/
│   └── uploads/            # Uploaded files stored here (auto-created)
├── samples/
│   └── sample_code.py      # Demo file with intentional bugs for testing
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- `pip`
- A [Groq](https://console.groq.com/) API key for real AI reviews *(free tier available)*

### 1. Clone / navigate to the project

```bash
cd ai_code_review
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set the Groq API key

Create a `.env` file in the `ai_code_review/` directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Or export it as an environment variable:

**Windows (PowerShell)**
```powershell
$env:GROQ_API_KEY = "your_groq_api_key_here"
```

**macOS / Linux**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

> If the key is not set the app still runs, returning a placeholder review instead of a real AI response.

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
6. Expand **Per-chunk structured review** to see per-function severity ratings and exact problematic lines.
7. Expand **Corrected Code** to see the AI-fixed version of your file.
8. Click **Download Review (.txt)** to save the report.

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

| Method | Endpoint  | Description |
|--------|-----------|-------------|
| GET    | `/`       | Health check |
| POST   | `/upload` | Upload a `.py` file |
| POST   | `/review` | Run an AI review on an uploaded file |

### `GET /` — response

```json
{ "status": "ok", "message": "AI Code Review Assistant is running." }
```

### `POST /upload`

**Request** — multipart form with a `.py` file:

```bash
curl -X POST http://localhost:8000/upload \
     -F "file=@samples/sample_code.py"
```

**Response:**

```json
{
  "filename": "sample_code.py",
  "saved_to": "data/uploads/sample_code.py",
  "status": "uploaded successfully"
}
```

**Errors:** `400` — file is not a `.py` file.

---

### `POST /review`

**Request body:**

```json
{
  "filename": "sample_code.py",
  "query": "Find all bugs and style issues"
}
```

```bash
curl -X POST http://localhost:8000/review \
     -H "Content-Type: application/json" \
     -d '{"filename": "sample_code.py", "query": "Find all bugs and style issues"}'
```

**Response:**

```json
{
  "filename": "sample_code.py",
  "query": "Find all bugs and style issues",
  "total_chunks": 8,
  "reviewed_chunks": 8,
  "review": "## 1. Code Quality Issues\n ...",
  "chunk_reviews": [
    {
      "chunk_name": "divide",
      "chunk_type": "function",
      "start_line": 16,
      "end_line": 18,
      "source": "def divide(a, b):\n    return a / b\n",
      "issue": "No guard against division by zero.",
      "suggestion": "Add `if b == 0: raise ValueError(...)` before the division.",
      "severity": "high",
      "problematic_lines": [17]
    }
  ],
  "corrected_code": "..."
}
```

**Errors:**
- `404` — file not found in uploads (upload it first)
- `400` — empty file, encoding error, or no functions/classes found

---

## Module Overview

### `backend/main.py`
FastAPI application entry point. Registers CORS middleware, defines the `/upload` and `/review` endpoints, and orchestrates chunking → vector search → LLM review → corrected code generation.

### `backend/code_processing.py`

| Symbol | Description |
|---|---|
| `CodeChunk` | Dataclass holding one function/class chunk with `name`, `chunk_type`, `source`, `start_line`, `end_line`, `docstring` |
| `chunk_code(code_text)` | Parse a source string with `ast` → list of `CodeChunk` |
| `chunk_file(file_path)` | Read a `.py` file from disk and delegate to `chunk_code` |

### `backend/analyzer.py`

| Symbol | Description |
|---|---|
| `review_chunk(chunk)` | Send a single `CodeChunk` to Groq and return a structured dict with `issue`, `suggestion`, `severity`, and `problematic_lines` |

### `backend/rag_search.py`

| Symbol | Description |
|---|---|
| `_TFIDFVectorizer` | Pure-NumPy TF-IDF vectorizer (no sklearn required) |
| `VectorStore` | Dataclass holding the TF-IDF matrix and chunk list |
| `create_faiss_index(chunks)` | Build a `VectorStore` from a list of text chunks |
| `retrieve_relevant_chunks(store, query, k)` | Cosine-similarity retrieval — returns top-k chunks |
| `generate_code_review(chunks)` | Call Groq to produce the full structured Markdown review |
| `generate_corrected_code(code_text)` | Call Groq to produce a fully corrected version of the source |

### `frontend/app.py`
Streamlit UI that handles file upload, calls the backend `/upload` and `/review` endpoints, and renders the review, per-chunk severity table, corrected code panel, and download button.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes (for real reviews) | Your Groq API key — get one at [console.groq.com](https://console.groq.com/) |

---

## Docker

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose plugin)

### 1. Build and start both services

```bash
cd ai_code_review
docker compose up --build
```

This starts:
- **Backend** → <http://localhost:8000> (API docs at `/docs`)
- **Frontend** → <http://localhost:8501>

### 2. Pass your Groq API key

Create a `.env` file in `ai_code_review/`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Then run:

```bash
docker compose up --build
```

### 3. Stop the containers

```bash
docker compose down
```

> Uploaded files are stored in a named Docker volume (`uploads_data`) so they persist across container restarts.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| *"Cannot reach the backend"* | Start the FastAPI server first (`uvicorn main:app --reload`) |
| *"Only Python (.py) files are accepted"* | Ensure you are uploading a `.py` file |
| *"No top-level functions or classes found"* | The file must contain at least one `def` or `class` |
| Placeholder review shown | Set `GROQ_API_KEY` and install `groq` (`pip install groq`) |
| `faiss-cpu` install fails on Windows | Run `pip install faiss-cpu --only-binary :all:` |
| Backend returns `404` for `/review` | Upload the file via `/upload` (or the UI) before reviewing |

