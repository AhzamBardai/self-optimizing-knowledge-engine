FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn httpx --no-cache-dir
COPY server.py .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
