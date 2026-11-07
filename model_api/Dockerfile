# Model API image - CPU-only TensorFlow (no GPU needed for a small MobileNetV2 on a free VM)
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt
COPY model_api/server.py .
COPY models/champion.keras models/champion.keras
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1
CMD ["gunicorn", "-b", "0.0.0.0:8000", "-w", "1", "--timeout", "120", "server:app"]
