FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_BACKEND=fastembed \
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    FASTEMBED_CACHE_DIR=/app/.fastembed \
    DATA_DIR=/app/data \
    CHROMA_DIR=/app/chroma_healthcare

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py support_rag.py ./
COPY data ./data

# Build the local demo index into the image so Cloud Run cold starts do not
# need to download documents and create Chroma data on the first request.
RUN python -c "from support_rag import build_index; build_index()"

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
