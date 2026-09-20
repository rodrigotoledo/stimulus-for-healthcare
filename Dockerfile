# HuggingFace Dialog Selector — one image serving the Stimulus UI and the JSON
# API from the same FastAPI process.

# ---------------------------------------------------------------------------
# Stage 1: compile Tailwind + daisyUI into app/static/css/app.css.
# The served page has no CSS runtime and makes no CDN request.
# ---------------------------------------------------------------------------
FROM node:22-alpine AS css

WORKDIR /build

# Dependency manifests first so the CSS layer caches independently of the app.
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

COPY tailwind.config.js ./
COPY app/templates ./app/templates
COPY app/static/css/tailwind.css ./app/static/css/tailwind.css
COPY app/static/js ./app/static/js

RUN npx tailwindcss -c tailwind.config.js \
      -i app/static/css/tailwind.css \
      -o app/static/css/app.css \
      --minify

# ---------------------------------------------------------------------------
# Stage 2: the application image. No Node toolchain ships.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# The compiled stylesheet from stage 1 — every other asset comes from app/.
COPY --from=css /build/app/static/css/app.css ./app/static/css/app.css

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
