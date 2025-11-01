FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY services/embeddings/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/embeddings/app.py .
COPY libs /app/libs

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
