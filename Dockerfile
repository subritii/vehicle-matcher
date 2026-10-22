# The full version: web app + OCR (PaddleOCR, Tesseract), barcode and plate readers.
# The vision model runs in the ollama service (docker-compose.yml). See the README for memory.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_CACHE_HOME=/home/app/.cache/paddlex \
    DATA_DIR=/app/data/scale DEMO_PHOTOS_DIR=/app/demo_photos PORT=8000

# OpenCV and Paddle need libGL, glib and OpenMP; Tesseract is the fallback OCR engine.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-ocr.txt requirements-ocr-excludes.txt ./
# uv understands --excludes: keeps a single OpenCV (see requirements-ocr.txt)
RUN pip install uv \
    && uv pip install --system -r requirements-ocr.txt --excludes requirements-ocr-excludes.txt \
    && pip uninstall -y uv

RUN useradd --create-home --uid 1000 app
COPY --chown=app:app . .
# Volumes mounted here start with these folders' contents and owner.
RUN mkdir -p data/scale demo_photos /home/app/.cache tests/fixtures/exteriors \
    && chown -R app:app data demo_photos /home/app/.cache tests/fixtures/exteriors
USER app

EXPOSE 8000
# The first start builds the scale demo (a few minutes) before the server answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=4)"
CMD ["python", "tools/docker_start.py"]
