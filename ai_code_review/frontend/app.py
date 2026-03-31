"""
app.py — Streamlit frontend for AI Code Review Assistant.

Current features:
    - Upload a Python (.py) file to the FastAPI backend.
    - Display the server response (filename + status).

Planned features (TODO):
    - Show parsed code chunks side-by-side with the source.
    - Ask review questions and display LLM answers.
    - Highlight code issues returned by the backend.
"""

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKEND_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Code Review Assistant",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 AI Code Review Assistant")
st.caption("Upload a Python file to get started.")

# ---------------------------------------------------------------------------
# File upload widget
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a Python file",
    type=["py"],
    help="Only .py files are accepted.",
)

if uploaded_file is not None:
    st.code(uploaded_file.getvalue().decode("utf-8"), language="python")

    if st.button("Upload & Analyse"):
        with st.spinner("Sending file to backend…"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/x-python")},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                st.success(f"✅ {data['status']}")
                st.json(data)

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the backend. Make sure FastAPI is running on port 8000.")
            except requests.exceptions.HTTPError as exc:
                st.error(f"Backend error: {exc.response.json().get('detail', str(exc))}")

# ---------------------------------------------------------------------------
# Placeholder — Q&A section (to be wired up after RAG is implemented)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Ask a Review Question")
st.info("RAG-based Q&A will appear here once the embedding pipeline is ready.", icon="ℹ️")

query = st.text_input("Your question", placeholder="e.g. Are there any security issues in this code?")
if st.button("Ask") and query:
    st.warning("LLM integration not yet implemented.", icon="⚠️")
