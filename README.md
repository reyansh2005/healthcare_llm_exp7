# Experiment 7: Production-Ready Healthcare RAG API

This experiment turns the Experiment 5 healthcare RAG prototype into a small FastAPI service. It validates question length, retrieves context from TXT/PDF/DOCX files, calls a cloud-hosted OpenAI-compatible LLM, logs operational metadata without secrets, handles failures safely, and measures response time. It is an educational demonstration, not a medical decision tool.

## Setup

1. Use Python 3.10 or later and create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in `LLM_API_BASE`, `LLM_API_KEY`, and `LLM_MODEL_ID`. Keep the key private.
4. Put `wellness.txt`, `clinic_faq.pdf`, and `medication_safety.docx` in `data/`.
5. Optionally build the index manually:
   ```powershell
   python -c "from support_rag import build_index; build_index()"
   ```

The first embedding/index operation downloads the sentence-transformers model and may take a few minutes. The model, vector store, retriever, and cloud client are reused after their first use.

## Run and test

Start the API:

```powershell
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs>, call `GET /health`, then try `POST /ask` with a supported and an unsupported question.

Run automated tests from this directory:

```powershell
pytest -q
```

## Lab acceptance checklist

| Check | Expected result |
|---|---|
| Health | `GET /health` returns `{"status":"ok"}` |
| Supported question | Grounded answer, source filenames, and latency are returned |
| Unsupported question | Exact refusal: `The provided documents do not contain this information.` |
| Empty question | HTTP 422 |
| Over-long question | HTTP 422 for more than 1000 characters |
| Backend failure | HTTP 500 with `Internal server error` only |
| Latency | `latency_ms` is present and rounded to one decimal |
| Secret handling | No API key in source, logs, or screenshots |

Retries/back-off, authentication, HTTPS, rate limiting, monitoring, secure secret management, load testing, evaluation, and recovery planning remain production follow-ups. Never commit `.env` or an API key.
