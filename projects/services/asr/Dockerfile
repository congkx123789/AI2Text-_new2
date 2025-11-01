FROM python:3.11-slim

WORKDIR /build
RUN pip install --no-cache-dir uv
COPY pyproject.toml .
RUN uv pip install --system -e .

WORKDIR /app
COPY app/ /app/app/

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

