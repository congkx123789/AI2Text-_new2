FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/asr/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/asr .
COPY libs /app/libs

EXPOSE 8000

# Run streaming server + the batch worker side-by-side (dev-friendly)
CMD /bin/sh -c "uvicorn streaming_server:app --host 0.0.0.0 --port 8000 & python -u worker.py"
