# Experiment 7 Deployment Theory and Observations

## 1. Aim

The aim of deployment was to expose the Experiment 5 healthcare Retrieval-Augmented Generation system as an internet-accessible FastAPI service. The service had to validate user input, retrieve information from the supplied healthcare documents, call a cloud-hosted LLM, handle failures safely, measure latency, and run within a free hosting environment.

The application was deployed using Render as a Docker/Python web service. Render was selected because it provides a simple GitHub-based deployment workflow and a free web-service option for educational demonstrations.

This deployment is for educational use only. It is not a production medical service and must not be used for diagnosis, prescriptions, or individual treatment decisions.

## 2. Deployment Theory

### 2.1 Application deployment

Deployment is the process of making an application available outside the local development computer. A deployed API requires:

- Application source code
- Runtime dependencies
- A process that listens for HTTP requests
- Environment-specific configuration
- Secure storage for credentials
- A hosting platform that can run the process

The local application is started with Uvicorn. Render starts the same application on a public server and forwards internet requests to it.

The deployed request path is:

```text
User browser or client
        |
        v
Render HTTPS URL
        |
        v
Uvicorn web server
        |
        v
FastAPI application
        |
        +--> Validate request
        +--> Search ChromaDB
        +--> Build grounded prompt
        +--> Call Groq cloud LLM
        +--> Return JSON response
```

### 2.2 Why a cloud LLM is used

The LLM is not installed inside the Render instance. The application sends the prompt to a cloud-hosted OpenAI-compatible endpoint. Groq provides the model execution and returns the generated answer.

This reduces local memory usage because the model weights for `openai/gpt-oss-120b` are stored and executed by Groq, not by Render. Render only stores the API client, retrieved context, and response data in memory.

The provider settings are supplied as environment variables:

```text
LLM_API_BASE
LLM_API_KEY
LLM_MODEL_ID
LLM_TEMPERATURE
```

The API key must never be hard-coded in Python, committed to GitHub, or included in screenshots.

### 2.3 Retrieval-Augmented Generation during deployment

The deployed service uses the same RAG pipeline as the local application:

1. Load `wellness.txt`, `clinic_faq.pdf`, and `medication_safety.docx`.
2. Split the documents into chunks of 500 characters with 80-character overlap.
3. Convert chunks into embedding vectors.
4. Store the vectors in ChromaDB using cosine distance.
5. Embed each user question.
6. Retrieve the three closest chunks.
7. Reject questions whose best cosine distance is greater than `0.70`.
8. Build a prompt containing the retrieved context.
9. Send the prompt to the cloud LLM.
10. Return the answer, source filenames, and latency.

The refusal string is:

```text
The provided documents do not contain this information.
```

The refusal occurs before the LLM is called. This prevents clearly unrelated questions from being answered using the model's general pretrained knowledge.

### 2.4 Embedding memory optimization

The first implementation used `sentence-transformers/all-MiniLM-L6-v2` through the Hugging Face integration. That path loads PyTorch and used too much memory for Render's free instance.

The deployed Render path was changed to:

```text
Backend: fastembed
Model: BAAI/bge-small-en-v1.5
Runtime: ONNX-based embedding inference
```

`fastembed` is lighter because it uses ONNX Runtime instead of loading the complete PyTorch Transformer stack. The embedding model is still downloaded and cached, but its runtime memory requirement is lower.

The code keeps the original MiniLM backend available for local development. The Render Docker configuration selects `fastembed` through environment variables.

### 2.5 Temporary storage

The deployed service uses temporary filesystem paths:

```text
/tmp/.fastembed       embedding model cache
/tmp/chroma_healthcare ChromaDB index
```

These paths are writable on Render's Python runtime. They are not permanent storage. The files may disappear when the service restarts, sleeps, or is redeployed.

This is acceptable for an educational demo with a small static knowledge base. A production system should use persistent object storage and a managed vector database.

## 3. Project Structure Used for Deployment

```text
experiment7/
├── app.py
├── support_rag.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env.example
├── .gitignore
├── README.md
├── DEPLOYMENT_THEORY_AND_OBSERVATIONS.md
├── data/
│   ├── wellness.txt
│   ├── clinic_faq.pdf
│   ├── medication_safety.docx
│   └── README.md
└── tests/
    └── test_api.py
```

### File responsibilities

| File | Responsibility |
|---|---|
| `app.py` | Defines FastAPI, `/health`, `/ask`, validation, timing, logging, and generic errors |
| `support_rag.py` | Loads documents, creates embeddings, manages ChromaDB, retrieves context, applies refusal logic, and calls Groq |
| `requirements.txt` | Lists runtime, LangChain, ChromaDB, embedding, OpenAI, and testing dependencies |
| `Dockerfile` | Defines a container image and starts Uvicorn on the platform port |
| `.dockerignore` | Prevents secrets, local environments, tests, and generated files from entering the image |
| `.env.example` | Safe configuration template without a real key |
| `.env` | Local-only configuration and API key; ignored by Git |
| `data/` | Contains the knowledge-base documents |
| `tests/test_api.py` | Tests the API without requiring a live provider |
| `DEPLOYMENT_THEORY_AND_OBSERVATIONS.md` | Records deployment theory, observations, fixes, and results |

## 4. Render Deployment Procedure

### Step 1: Prepare the repository

The project was committed to the GitHub repository:

```text
https://github.com/reyansh2005/healthcare_llm_exp7
```

The `.env` file, virtual environment, ChromaDB directory, and Python cache files were excluded using `.gitignore`.

### Step 2: Create the Render service

A Render Web Service was connected to the GitHub repository. The service was configured with a Python-style start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

For a Python runtime, the build command is:

```bash
pip install -r requirements.txt
```

Render provides the `PORT` environment variable. Binding to `0.0.0.0` is required so Render can route traffic to the application.

### Step 3: Configure environment variables

The following non-secret values are required on Render:

```text
EMBEDDING_BACKEND=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
FASTEMBED_CACHE_DIR=/tmp/.fastembed
LLM_API_BASE=https://api.groq.com/openai/v1
LLM_MODEL_ID=openai/gpt-oss-120b
LLM_TEMPERATURE=0.0
DATA_DIR=./data
CHROMA_DIR=/tmp/chroma_healthcare
```

`LLM_API_KEY` is added separately as a protected Render environment variable. It is not stored in GitHub or in this report.

### Step 4: Test the deployment

The service health endpoint is tested first:

```text
GET https://healthcare-llm-exp7-1.onrender.com/health
```

Expected response:

```json
{"status":"ok"}
```

The RAG endpoint is then tested with:

```json
{
  "question": "What should someone do about a missed medicine dose?"
}
```

The interactive documentation is available at:

```text
https://healthcare-llm-exp7-1.onrender.com/docs
```

## 5. Observations During Deployment

### Observation 1: Root endpoint behavior

Requesting `/` returned HTTP 404 because no root route was defined. This was expected. The available endpoints are:

- `/health` for service availability
- `/ask` for RAG questions
- `/docs` for interactive API documentation

### Observation 2: Hugging Face model downloads

The embedding backend downloaded model files from Hugging Face during initialization. HTTP 200 responses indicated successful downloads. Some optional Hugging Face metadata requests returned 404, but those were not fatal errors.

The first request is slower because the embedding model must be downloaded and initialized. Subsequent requests can reuse the cached model while the instance remains alive.

### Observation 3: Groq model availability

An earlier model configuration returned HTTP 404 because the configured Groq model was unavailable. The model was changed to a currently available model:

```text
openai/gpt-oss-120b
```

Provider model names can change, so the active model must be checked in the provider console.

### Observation 4: Render memory failure

The first Render deployment exceeded the free instance memory limit:

```text
Instance failed: Ran out of memory
```

The cause was the PyTorch-based `sentence-transformers` embedding path. The solution was to add `fastembed` and use `BAAI/bge-small-en-v1.5` for the Render deployment.

### Observation 5: Permission failure for `/app`

The Render logs showed:

```text
PermissionError: [Errno 13] Permission denied: '/app'
```

The Python runtime was configured with Docker-only paths such as `/app/.fastembed`. Render's Python runtime could not create that directory. The solution was to use writable temporary paths and add code-level fallbacks:

```text
FASTEMBED_CACHE_DIR=/tmp/.fastembed
CHROMA_DIR=/tmp/chroma_healthcare
```

The code also falls back automatically when a configured path is missing or not writable.

### Observation 6: Empty GPT-OSS response

Groq returned HTTP 200, but the response content was empty. The cause was that the GPT-OSS model used the small completion budget for reasoning tokens.

The provider call was updated for GPT-OSS to use:

```text
max_completion_tokens = 512
reasoning_effort = low
```

Other OpenAI-compatible models continue to use the normal `max_tokens` setting.

### Observation 7: Controlled API errors

The client receives only:

```json
{"detail":"Internal server error"}
```

The server logs contain the exception type and operational details. This prevents provider keys, prompts, configuration, and internal stack traces from being exposed through the HTTP response.

## 6. Final Deployment Flow

```text
GitHub repository
        |
        v
Render builds Python service
        |
        +--> pip install -r requirements.txt
        +--> start Uvicorn on $PORT
        |
        v
GET /health
        |
        +--> FastAPI process responds
        |
        v
POST /ask
        |
        +--> Validate question
        +--> Load or reuse fastembed
        +--> Load or create ChromaDB in /tmp
        +--> Retrieve top three chunks
        +--> Apply distance threshold
        +--> Call Groq GPT-OSS
        +--> Return answer, sources, latency
```

## 7. Test Results

The automated API test suite was run after the deployment fixes:

```text
6 passed, 1 warning
```

The tests cover:

- Health endpoint success
- Valid question response
- Source filename de-duplication
- Empty-question validation
- Over-long-question validation
- Generic backend failure response

The warning was an upstream LangChain deprecation warning and did not cause test failure.

## 8. Final Result

The application was successfully prepared for Render deployment with:

- FastAPI HTTP endpoints
- Cloud-hosted Groq LLM inference
- TXT, PDF, and DOCX retrieval
- ChromaDB vector search
- Cosine-distance refusal logic
- Low-memory FastEmbed embeddings
- Render-compatible writable paths
- GPT-OSS completion handling
- Input validation
- Controlled errors
- Latency measurement
- Source reporting
- GitHub-based deployment

## 9. Limitations

This deployment is not fully production-ready. Important limitations are:

- Render Free services can sleep when idle.
- The first request can be slow because model files may need to download.
- `/tmp` storage is temporary and can be deleted on restart.
- ChromaDB is local to one service instance.
- The free instance has limited memory and CPU.
- Free LLM providers can impose quotas and rate limits.
- The LLM can still produce unsupported details despite prompt grounding.
- There is no authentication or rate limiting.
- The service should not receive confidential medical or personal data.
- The API must not be used for medical decisions.

A production system should add persistent vector storage, authentication, HTTPS configuration, rate limiting, monitoring, secret management, evaluation, retries, load testing, multiple instances, and a recovery plan.

## 10. Security Observation

The provider API key must be rotated whenever it is exposed in a chat message, screenshot, terminal output, Git history, or public repository. The real key belongs only in the local `.env` file or the Render secret environment variable. It must never be placed in `.env.example`, README files, source code, or this report.
