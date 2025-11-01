FROM python:3.11-slim

WORKDIR /app

COPY services/metadata/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/metadata/app.py .
COPY services/metadata/migrations /app/migrations
COPY libs /app/libs

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
