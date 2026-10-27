"""FastAPI application for the healthcare RAG service."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from support_rag import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("rag_api")

app = FastAPI(title="Healthcare RAG API", version="1.0.0")


class AskRequest(BaseModel):
    """Request body for a RAG question."""

    question: str = Field(..., min_length=1, max_length=1000)


class AskResponse(BaseModel):
    """Successful RAG response."""

    answer: str
    sources: list[str]
    latency_ms: float


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a validated question and report response latency."""
    started = time.perf_counter()
    q_len = len(request.question)
    try:
        answer, docs, best_distance = answer_question(request.question)
        latency_ms = (time.perf_counter() - started) * 1000
        unique_sources: list[str] = []
        for document in docs:
            source = str(document.metadata.get("source", "unknown"))
            source_name = source.replace("\\", "/").rsplit("/", 1)[-1]
            if source_name not in unique_sources:
                unique_sources.append(source_name)
        logger.info(
            "RAG success q_len=%d best_distance=%.4f latency_ms=%.1f",
            q_len,
            best_distance,
            latency_ms,
        )
        return AskResponse(
            answer=answer,
            sources=unique_sources,
            latency_ms=round(latency_ms, 1),
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            "RAG failure q_len=%d latency_ms=%.1f exception=%s",
            q_len,
            latency_ms,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from None
