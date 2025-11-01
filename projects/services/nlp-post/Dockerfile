FROM python:3.11-slim

WORKDIR /build
RUN pip install --no-cache-dir uv
COPY pyproject.toml .
RUN uv pip install --system -e .

WORKDIR /app
COPY app/ /app/app/

USER 1000
CMD ["python", "-m", "app.worker"]

