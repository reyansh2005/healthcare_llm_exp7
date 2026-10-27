"""Focused API tests that do not require a live model or knowledge base."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from pathlib import Path

import app

client = TestClient(app.app)


def fake_answer_question(question: str) -> tuple[str, list[Document], float]:
    return (
        "Use the supplied educational information and contact a professional for urgent concerns.",
        [
            Document(page_content="one", metadata={"source": "data/wellness.txt"}),
            Document(page_content="two", metadata={"source": "data/wellness.txt"}),
            Document(page_content="three", metadata={"source": "data/clinic_faq.pdf"}),
        ],
        0.25,
    )


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_question_returns_answer_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "answer_question", fake_answer_question)
    response = client.post("/ask", json={"question": "How can I stay well?"})
    body = response.json()
    assert response.status_code == 200
    assert body["answer"]
    assert body["sources"] == ["wellness.txt", "clinic_faq.pdf"]
    assert isinstance(body["latency_ms"], float)


def test_empty_question_is_rejected() -> None:
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_overlong_question_is_rejected() -> None:
    response = client.post("/ask", json={"question": "x" * 1001})
    assert response.status_code == 422


def test_backend_failure_returns_generic_500(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(question: str) -> tuple[str, list[Document], float]:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(app, "answer_question", fail)
    response = client.post("/ask", json={"question": "Will this fail?"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


@pytest.mark.skipif(
    not all(
        (Path("data") / filename).is_file()
        for filename in ("wellness.txt", "clinic_faq.pdf", "medication_safety.docx")
    ),
    reason="knowledge-base files are not present",
)
def test_live_path_hides_llm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import support_rag

    def fail_llm(prompt: str, max_new_tokens: int) -> str:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(support_rag, "_call_llm", fail_llm)
    response = client.post("/ask", json={"question": "What is wellness?"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
