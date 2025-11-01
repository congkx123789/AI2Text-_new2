FROM python:3.11-slim

WORKDIR /app

COPY services/api-gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/api-gateway/app.py .
COPY libs /app/libs

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
