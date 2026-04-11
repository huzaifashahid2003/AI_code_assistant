# ── Base image ──────────────────────────────────────────────────────────────
# Python 3.11 slim keeps the image small while matching the minimum requirement
# of Python 3.10+ used by this project.
FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
# build-essential is needed by some packages (e.g. faiss-cpu) on certain arches.
# We clean up the apt cache in the same layer to minimise image size.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
# Copy only requirements first so Docker can cache this layer independently of
# source-code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ───────────────────────────────────────────────────────
COPY . .

# Ensure the uploads directory exists inside the image (it is also mounted as a
# volume by docker-compose, but having it here prevents a start-up race).
RUN mkdir -p /app/data/uploads

# ── Default command ──────────────────────────────────────────────────────────
# docker-compose overrides this per service; the default here starts the backend
# so the image is also useful standalone.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
