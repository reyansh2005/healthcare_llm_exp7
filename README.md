# Experiment 7: Production-Ready Healthcare RAG API

This project converts the Experiment 5 healthcare RAG prototype into a small FastAPI service. It loads TXT, PDF, and DOCX documents, creates a persistent ChromaDB index, retrieves relevant passages, and sends the retrieved context to a cloud-hosted OpenAI-compatible LLM.

The production-oriented improvements are request validation, controlled error handling, logging, response-time measurement, reusable heavy objects, environment-based configuration, and endpoint tests. This remains an educational application and must not be used for diagnosis, prescriptions, or medical decisions.

## 1. Complete Theory

### Prototype versus production-oriented application

A notebook prototype is useful for experimenting with loaders, embeddings, retrieval, prompts, and models. It is usually run manually and may load expensive objects repeatedly. An API service must provide a stable interface and behave predictably when users send invalid input or when a dependency fails.

This project adds:

- FastAPI endpoints for other applications to call.
- Pydantic validation for questions from 1 to 1,000 characters.
- Generic HTTP 500 responses for provider, file, index, and network failures.
- Logs containing operational metadata but not questions, prompts, keys, or responses.
- Response-time measurement in milliseconds.
- Lazy, reusable embedding, ChromaDB, retriever, and LLM objects.
- Automated API tests using FastAPI `TestClient`.

### Retrieval-Augmented Generation

Retrieval-Augmented Generation, or RAG, combines information retrieval with language generation. Instead of asking the language model to answer only from its training data, the application searches a supplied knowledge base and places the most relevant passages in the prompt.

The three stages are:

1. **Retrieval:** find passages semantically similar to the question.
2. **Augmentation:** add those passages to a grounded prompt.
3. **Generation:** ask the LLM to answer using the supplied context.

RAG allows the knowledge base to change without retraining the LLM and lets the API return source filenames. It does not guarantee correctness: weak retrieval or a small model can still produce unsupported details.

### Document loading and chunking

The knowledge base contains three file types:

| File | Loader | Purpose |
|---|---|---|
| `wellness.txt` | `TextLoader` | General wellness education |
| `clinic_faq.pdf` | `PyPDFLoader` | Clinic questions and answers |
| `medication_safety.docx` | `Docx2txtLoader` | Medication safety information |

Each loader creates a LangChain `Document` containing `page_content` and `metadata`. Metadata preserves source filenames and PDF page numbers.

Whole documents are too large and mix topics, so `RecursiveCharacterTextSplitter` creates chunks:

```text
chunk_size = 500 characters
chunk_overlap = 80 characters
```

The overlap keeps context from being lost at a chunk boundary.

### Embeddings and semantic search

`sentence-transformers/all-MiniLM-L6-v2` converts each chunk and question into a numerical vector. Semantically similar text produces vectors that are close together, enabling semantic search rather than exact keyword matching.

The vectors are normalized and stored in ChromaDB using cosine distance:

```text
cosine distance = 1 - cosine similarity
0.0             = identical direction
smaller value   = more similar
larger value    = less related
```

The same embedding model is used for stored chunks and incoming questions. The lab uses CPU embeddings for deterministic behavior.

### Out-of-scope refusal

A vector store returns the nearest chunks even when a question is unrelated. Without a guard, the LLM may answer from pretrained knowledge and hallucinate.

This project applies a distance threshold:

```text
if best_distance > 0.70:
    return "The provided documents do not contain this information."
```

The refusal happens before the cloud LLM is called. The threshold is a practical experiment setting, not proof that every answer below it is correct.

### Prompt grounding and safety

The prompt instructs the model to answer only from retrieved context, refuse when context is insufficient, avoid diagnosis and prescription changes, and advise immediate professional help for urgent symptoms. Retrieved passages are labeled with source filenames.

The LLM is accessed through the OpenAI-compatible `openai` SDK. The provider URL, API key, model name, and temperature are read from `.env`; no provider key is hard-coded in Python.

### Validation, errors, logging, and latency

`AskRequest` accepts a question from 1 to 1,000 characters. Invalid requests are rejected by FastAPI/Pydantic with HTTP 422 before the RAG pipeline runs. Valid requests are timed using `time.perf_counter()`.

Unexpected failures are logged on the server and returned to callers only as:

```json
{"detail": "Internal server error"}
```

This prevents prompts, API keys, configuration, and stack traces from being returned to clients.

## 2. End-to-End Project Flow

### Startup flow

```text
Start uvicorn
    |
    +--> import app.py
    |       define FastAPI routes and Pydantic models
    |
    +--> import support_rag.py
            load .env configuration
            do not load heavy models or files yet
```

Heavy resources are lazy, so `/health` can work before embeddings, data files, or the cloud LLM are used.

### `/ask` flow

```text
POST /ask with question
        |
        v
Pydantic validates 1-1000 characters
        |
        v
Start latency timer
        |
        v
answer_question(question)
        |
        +--> create/load embedding singleton
        +--> create/load persistent Chroma singleton
        |       |
        |       +--> if absent: load files, split, embed, persist
        |
        +--> similarity search, k=3
        +--> calculate best cosine distance
                |
                +--> distance > 0.70: return exact refusal
                |
                +--> otherwise: build prompt and call cloud LLM
        |
        v
Deduplicate source filenames
        |
Log safe metadata and return answer, sources, latency
```

### Failure flow

```text
Invalid request       --> HTTP 422
Provider/index error  --> server log + HTTP 500
Unsupported question  --> exact refusal, no LLM call
```

## 3. File Responsibilities

| File or directory | Responsibility |
|---|---|
| `app.py` | FastAPI app, `/health`, `/ask`, validation, timing, source cleanup, safe errors, and API logs |
| `support_rag.py` | Loaders, chunking, embeddings, ChromaDB, retrieval, refusal threshold, prompt, and cloud LLM call |
| `requirements.txt` | Runtime and test dependencies |
| `.env.example` | Safe configuration template with no real secret |
| `.env` | Local provider configuration; ignored and never committed |
| `.gitignore` | Ignores secrets, virtual environments, caches, and generated ChromaDB |
| `data/README.md` | Instructions for the three source documents |
| `data/wellness.txt` | Wellness source document |
| `data/clinic_faq.pdf` | Clinic FAQ source document |
| `data/medication_safety.docx` | Medication safety source document |
| `chroma_healthcare/` | Generated persistent vector index; ignored by Git |
| `tests/test_api.py` | Endpoint and controlled-error tests |
| `README.md` | Theory, flow, setup, operation, testing, and limitations |

## 4. Main Code

### API models and endpoints in `app.py`

```python
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    started = time.perf_counter()
    try:
        answer, docs, best_distance = answer_question(request.question)
        latency_ms = (time.perf_counter() - started) * 1000
        sources = []
        for document in docs:
            source = str(document.metadata.get("source", "unknown"))
            source_name = source.replace("\\", "/").rsplit("/", 1)[-1]
            if source_name not in sources:
                sources.append(source_name)
        return AskResponse(
            answer=answer,
            sources=sources,
            latency_ms=round(latency_ms, 1),
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from None
```

### Lazy resources in `support_rag.py`

```python
_embedding_model = None
_vector_store = None
_retriever = None
_llm_client = None
_singleton_lock = threading.RLock()
```

The getters create these objects only on first use and then reuse them. The lock prevents concurrent requests from loading the same heavy object twice.

### Loading and indexing

```python
documents = []
documents.extend(TextLoader("data/wellness.txt", encoding="utf-8").load())
documents.extend(PyPDFLoader("data/clinic_faq.pdf").load())
documents.extend(Docx2txtLoader("data/medication_safety.docx").load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
)
chunks = splitter.split_documents(documents)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="healthcare",
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="./chroma_healthcare",
)
```

If the persistent directory already exists, it is opened rather than rebuilt.

### Retrieval and generation

```python
docs_with_scores = vector_store.similarity_search_with_score(question, k=3)
docs = [document for document, _ in docs_with_scores]
best_distance = min(score for _, score in docs_with_scores)

if best_distance > DISTANCE_THRESHOLD:
    return REFUSAL, docs, best_distance

context = "\n\n".join(
    f"[Source: {document.metadata.get('source', 'unknown')}]\n"
    f"{document.page_content}"
    for document in docs
)
prompt = (
    f"{SYSTEM_INSTRUCTION}\n\nCONTEXT:\n{context}\n\n"
    f"QUESTION: {question}\n\nANSWER:"
)
```

The complete implementation is in `support_rag.py`; this excerpt shows the controlling retrieval, refusal, and prompt behavior.

## 5. Setup and Configuration

Use Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` with a provider and a currently available model:

```env
LLM_API_BASE=https://api.groq.com/openai/v1
LLM_API_KEY=your_private_provider_key
LLM_MODEL_ID=your_current_provider_model_id
LLM_TEMPERATURE=0.0
CHROMA_DIR=./chroma_healthcare
DATA_DIR=./data
```

Provider model names change, so use a model currently listed by the provider. Never commit `.env`, paste an API key into screenshots, or put a real key in `.env.example`.

Place these files in `data/`:

```text
data/wellness.txt
data/clinic_faq.pdf
data/medication_safety.docx
```

Build the index manually, or let the first `/ask` request create it:

```powershell
python -c "from support_rag import build_index; build_index()"
```

The first indexing operation downloads the embedding model and may take a few minutes.

## 6. Run the API

```powershell
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs>.

`GET /health` checks only whether the FastAPI process is alive:

```json
{"status":"ok"}
```

`POST /ask` runs the RAG pipeline:

```json
{
  "question": "How should medicines be stored?"
}
```

Example response:

```json
{
  "answer": "...",
  "sources": ["medication_safety.docx"],
  "latency_ms": 142.6
}
```

An unrelated question such as `What is the capital of France?` should return:

```text
The provided documents do not contain this information.
```

## 7. Testing

Run the automated API tests from the project directory:

```powershell
python -m pytest -q
```

The tests cover health, valid requests, empty questions, over-long questions, generic backend failures, source de-duplication, and optional live-path failure behavior.

## 8. Acceptance Checklist

| Check | Expected result |
|---|---|
| Health | `GET /health` returns `{"status":"ok"}` |
| Supported question | Grounded answer, source filenames, and latency are returned |
| Unsupported question | Exact refusal is returned before the LLM call |
| Empty question | HTTP 422 |
| Over-long question | HTTP 422 for more than 1,000 characters |
| Backend failure | HTTP 500 with `Internal server error` only |
| Latency | `latency_ms` is present and rounded to one decimal |
| Reuse | Embeddings, Chroma, retriever, and LLM client are reused |
| Secret handling | No API key is stored in source control or logs |

## 9. Render Deployment

Render can deploy this project as a Docker web service directly from GitHub. Select the **Free** instance type for a lab demo. Free services sleep when idle, so the first request after inactivity may be slow. The local Chroma index is built into the image; persistent user-generated data is not supported by this demo.

Render uses the lower-memory `fastembed` ONNX backend through Docker environment variables. Local development keeps the original `sentence-transformers/all-MiniLM-L6-v2` backend unless `EMBEDDING_BACKEND=fastembed` is set. This distinction is needed because the Render Free instance has limited memory.

Create a Render Web Service connected to this repository with these settings:

| Setting | Value |
|---|---|
| Runtime | Docker |
| Dockerfile path | `./Dockerfile` |
| Instance type | Free |
| Health check path | `/health` |
| Build command | Leave empty; Dockerfile handles it |
| Start command | Leave empty; Dockerfile handles it |

Add these environment variables in Render. Mark `LLM_API_KEY` as a secret:

```text
LLM_API_BASE=https://api.groq.com/openai/v1
LLM_API_KEY=your_private_provider_key
LLM_MODEL_ID=your_current_provider_model_id
LLM_TEMPERATURE=0.0
DATA_DIR=/app/data
CHROMA_DIR=/app/chroma_healthcare
```

Render supplies `PORT` automatically. The Dockerfile uses it and binds Uvicorn to `0.0.0.0`.

After deployment, Render provides a URL such as `https://experiment7-api.onrender.com`. Test it with:

```powershell
$SERVICE_URL = "https://your-service-name.onrender.com"
Invoke-RestMethod "$SERVICE_URL/health"

Invoke-RestMethod `
    -Method Post `
    -Uri "$SERVICE_URL/ask" `
    -ContentType "application/json" `
    -Body '{"question":"How should medicines be stored?"}'
```

Open `$SERVICE_URL/docs` for the interactive API documentation. Do not commit `.env` or paste the provider key into GitHub.

## 10. Limitations and Production Follow-ups

This is not a fully production-ready public medical service. A real deployment would also require authentication, authorization, HTTPS, rate limiting, secure secret management, structured redacted logs, monitoring, alerting, model evaluation, prompt-injection defenses, load testing, retries with back-off, high availability, and a recovery plan.

Free cloud inference can have quotas, rate limits, changing model names, and provider outages. Do not send confidential medical, financial, or personal data to a third-party model API.

The retrieval threshold prevents many unrelated questions from reaching the LLM, but it does not prove that an answer is correct. The LLM can still add unsupported details when context is ambiguous. Users should verify answers against returned source documents and consult a qualified healthcare professional for medical concerns. This project is for educational purposes only and must not be used for diagnosis, prescriptions, or medical decisions.