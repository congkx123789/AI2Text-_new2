FROM python:3.11-slim

WORKDIR /app

COPY services/nlp-post/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/nlp-post/app.py .
COPY libs /app/libs

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
