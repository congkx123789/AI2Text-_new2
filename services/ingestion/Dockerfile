FROM python:3.11-slim

WORKDIR /app

# Install ffmpeg for audio transcoding (Phase A)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY services/ingestion/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ingestion/app.py .
COPY libs /app/libs

# Create mount points for hot folder
RUN mkdir -p /mnt/inbox /mnt/processed

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
