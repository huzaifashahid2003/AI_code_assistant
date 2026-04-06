from pathlib import Path

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"
UPLOADS_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def save_file_locally(filename: str, content: bytes) -> None:
    dest = UPLOADS_DIR / Path(filename).name
    dest.write_bytes(content)


def upload_to_backend(filename: str, content: bytes) -> None:
    response = requests.post(
        f"{BACKEND_URL}/upload",
        files={"file": (filename, content, "text/x-python")},
        timeout=30,
    )
    response.raise_for_status()


def request_review(filename: str, query: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/review",
        json={"filename": filename, "query": query},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def run_analysis(filename: str, content: bytes, query: str) -> None:
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
    st.session_state["corrected_code"] = data.get("corrected_code", "")
    st.session_state["chunk_reviews"] = data.get("chunk_reviews", [])

st.set_page_config(
    page_title="AI Code Review Assistant",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 AI Code Review Assistant")
st.caption("Upload a Python file and get an AI-powered code review instantly.")

if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""
if "review_filename" not in st.session_state:
    st.session_state["review_filename"] = ""
if "review_total" not in st.session_state:
    st.session_state["review_total"] = "?"
if "review_reviewed" not in st.session_state:
    st.session_state["review_reviewed"] = "?"
if "corrected_code" not in st.session_state:
    st.session_state["corrected_code"] = ""
if "chunk_reviews" not in st.session_state:
    st.session_state["chunk_reviews"] = []

st.divider()

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
if st.session_state["review_text"]:
    st.divider()
    total = st.session_state["review_total"]
    reviewed = st.session_state["review_reviewed"]
    fname = st.session_state["review_filename"]

    st.success(f"Review complete — {reviewed} of {total} code chunk(s) analysed from `{fname}`.")

    st.subheader("Review Suggestions")

    chunk_reviews = st.session_state.get("chunk_reviews", [])
    if chunk_reviews:
        _SEVERITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🟢", "none": "✅"}
        for rev in chunk_reviews:
            icon = _SEVERITY_ICON.get(rev.get("severity", "none"), "⚪")
            label = (
                f"{icon} `{rev['chunk_name']}` "
                f"({rev['chunk_type']}) — "
                f"Lines {rev['start_line']}–{rev['end_line']} — "
                f"severity: **{rev['severity']}**"
            )
            with st.expander(label, expanded=(rev.get("severity") in {"high", "medium"})):
                # ── Source code with line-by-line highlighting ──────────────
                source_lines = rev["source"].splitlines()
                start = rev["start_line"]
                bad = set(rev.get("problematic_lines", []))

                html_parts = [
                    '<div style="'
                    'font-family:monospace;font-size:13px;line-height:1.7;'
                    'background:#0e1117;border-radius:6px;padding:8px 4px;'
                    'overflow-x:auto;">'
                ]
                for i, line in enumerate(source_lines):
                    lineno = start + i
                    # Escape HTML special characters
                    escaped = (
                        line
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    if lineno in bad:
                        html_parts.append(
                            f'<div style="background:rgba(255,80,80,0.18);'
                            f'border-left:3px solid #e05252;padding:1px 8px;border-radius:2px">'
                            f'<span style="color:#e05252;min-width:38px;display:inline-block">'
                            f'{lineno}</span>'
                            f'<span style="color:#ff6b6b"> 🔴 {escaped}</span>'
                            f'</div>'
                        )
                    else:
                        html_parts.append(
                            f'<div style="padding:1px 8px">'
                            f'<span style="color:#555;min-width:38px;display:inline-block">'
                            f'{lineno}</span>'
                            f'<span style="color:#cdd3de"> {escaped}</span>'
                            f'</div>'
                        )
                html_parts.append("</div>")
                st.markdown("".join(html_parts), unsafe_allow_html=True)

                # ── Structured feedback below the code block ─────────────
                st.markdown(f"**Issue:** {rev['issue']}")
                st.markdown(f"**Suggestion:** {rev['suggestion']}")
    else:
        # Fallback: render the plain markdown review if no per-chunk data
        st.markdown(st.session_state["review_text"])

    st.divider()
    corrected = st.session_state.get("corrected_code", "")
    if corrected:
        st.download_button(
            label="⬇️  Download correct.py",
            data=corrected,
            file_name=f"correct_{Path(fname).stem}.py",
            mime="text/x-python",
            help="Download the corrected Python file with all issues fixed.",
        )
