# ─── Stage 1: builder ─────────────────────────────────────────────────────
# Compiles deps (LightGBM needs build tools) into an isolated venv we can copy.
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only what's needed for `pip install -e .` to succeed.
# Note: pyproject.toml replaces setup.py as of Phase 2 Session 2B.
COPY requirements.txt pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e .


# ─── Stage 2: runtime ─────────────────────────────────────────────────────
# Slim image with only the runtime libs LightGBM needs.
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 app

# Bring the prebuilt venv across (no compilers in runtime image)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# App code + model artifacts
COPY --chown=app:app api/    ./api/
COPY --chown=app:app src/    ./src/
COPY --chown=app:app models/ ./models/

# Drop root before running the service
USER app

EXPOSE 8000

# In-image healthcheck that hits /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
