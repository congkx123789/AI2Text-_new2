FROM python:3.11-slim

RUN pip install uv
COPY pyproject.toml /tmp/
RUN cd /tmp && uv pip install --system -e .

COPY app/ /app/app/
WORKDIR /app

USER 1000
CMD ["python", "-m", "app.worker"]

