"""
app.py — Streamlit frontend for AI Code Review Assistant.

Features:
    - Upload a Python (.py) file (or use built-in sample code).
    - Enter a review query and click "Analyze Code".
    - Sends the file to FastAPI /upload, then calls /review for AI feedback.
    - Displays review suggestions as rendered Markdown.
    - Lets the user download the review as a .txt file.
"""

from pathlib import Path

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKEND_URL = "http://localhost:8000"
UPLOADS_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FILENAME = "sample_code.py"
SAMPLE_CODE = """\
def add(a, b):
    return a + b


def divide(a, b):
    return a / b  # potential ZeroDivisionError — no guard for b == 0


class Calculator:
    def __init__(self):
        self.history = []

    def compute(self, op, a, b):
        if op == "add":
            result = add(a, b)
        elif op == "divide":
            result = divide(a, b)
        else:
            result = None
        self.history.append((op, a, b, result))
        return result
"""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def save_file_locally(filename: str, content: bytes) -> None:
    """Persist uploaded file bytes to the local uploads directory."""
    dest = UPLOADS_DIR / Path(filename).name
    dest.write_bytes(content)


def upload_to_backend(filename: str, content: bytes) -> None:
    """POST the file to the FastAPI /upload endpoint."""
    response = requests.post(
        f"{BACKEND_URL}/upload",
        files={"file": (filename, content, "text/x-python")},
        timeout=30,
    )
    response.raise_for_status()


def request_review(filename: str, query: str) -> dict:
    """POST to the FastAPI /review endpoint and return the JSON response."""
    response = requests.post(
        f"{BACKEND_URL}/review",
        json={"filename": filename, "query": query},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def run_analysis(filename: str, content: bytes, query: str) -> None:
    """Orchestrate upload → review and render results in the Streamlit UI."""
    with st.spinner("Uploading and analysing — this may take a few seconds..."):
        try:
            upload_to_backend(filename, content)
            data = request_review(filename, query)
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach the backend. "
                "Make sure FastAPI is running on http://localhost:8000."
            )
            return
        except requests.exceptions.Timeout:
            st.error("Request timed out. The backend may be overloaded — try again.")
            return
        except requests.exceptions.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            st.error(f"Backend error: {detail}")
            return
        except requests.exceptions.RequestException as exc:
            st.error(f"Unexpected network error: {exc}")
            return

    review_text = data.get("review", "").strip()
    if not review_text:
        st.info("The backend returned an empty review. Try a different query or file.")
        return

    total = data.get("total_chunks", "?")
    reviewed = data.get("reviewed_chunks", "?")

    # Persist in session state so the download button survives re-runs
    st.session_state["review_text"] = review_text
    st.session_state["review_filename"] = filename
    st.session_state["review_total"] = total
    st.session_state["review_reviewed"] = reviewed


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Code Review Assistant",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 AI Code Review Assistant")
st.caption("Upload a Python file and get an AI-powered code review instantly.")

# Initialise session state keys
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""
if "review_filename" not in st.session_state:
    st.session_state["review_filename"] = ""
if "review_total" not in st.session_state:
    st.session_state["review_total"] = "?"
if "review_reviewed" not in st.session_state:
    st.session_state["review_reviewed"] = "?"

st.divider()

# --- File upload ---
uploaded_file = st.file_uploader(
    "Upload a Python file (.py)",
    type=["py"],
    help="Only .py files are accepted.",
)

use_sample = st.checkbox(
    "Use built-in sample code instead (no upload required)",
    value=False,
)

# --- Code preview ---
if uploaded_file is not None:
    with st.expander("Preview uploaded file", expanded=False):
        st.code(
            uploaded_file.getvalue().decode("utf-8", errors="replace"),
            language="python",
        )
elif use_sample:
    with st.expander("Preview sample code", expanded=True):
        st.code(SAMPLE_CODE, language="python")

st.divider()

# --- Query input ---
query = st.text_input(
    "Review query",
    value="Review my code for bugs, issues, and improvements.",
    placeholder="e.g. Are there any security issues or bugs in this code?",
)

# --- Analyze button ---
if st.button("Analyze Code", type="primary"):
    if not query.strip():
        st.warning("Please enter a review query before analyzing.")
    elif uploaded_file is not None:
        save_file_locally(uploaded_file.name, uploaded_file.getvalue())
        run_analysis(uploaded_file.name, uploaded_file.getvalue(), query.strip())
    elif use_sample:
        sample_bytes = SAMPLE_CODE.encode("utf-8")
        save_file_locally(SAMPLE_FILENAME, sample_bytes)
        run_analysis(SAMPLE_FILENAME, sample_bytes, query.strip())
    else:
        st.warning(
            "No file selected. Upload a .py file or check "
            "'Use built-in sample code instead'."
        )

# --- Review results (rendered after analysis or from session state) ---
if st.session_state["review_text"]:
    st.divider()
    total = st.session_state["review_total"]
    reviewed = st.session_state["review_reviewed"]
    fname = st.session_state["review_filename"]

    st.success(f"Review complete — {reviewed} of {total} code chunk(s) analysed from `{fname}`.")

    st.subheader("Review Suggestions")
    st.markdown(st.session_state["review_text"])

    st.divider()
    st.download_button(
        label="⬇️  Download Review (.txt)",
        data=st.session_state["review_text"],
        file_name=f"review_{Path(fname).stem}.txt",
        mime="text/plain",
        help="Save the full review report as a plain-text file.",
    )
