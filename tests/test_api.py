from fastapi.testclient import TestClient

from app.api import app
from app.evaluation import EvaluationResult
from app.miner import RawDocument


class DummyStore:
    def search(self, query: str, top_k: int = 5):
        lower = query.lower()
        if "dangerous" in lower or "conditions" in lower:
            text = "High blood pressure and sepsis are serious conditions discussed in the paper."
        elif "risk" in lower or "factors" in lower:
            text = "Risk factors include delayed diagnosis and poor monitoring in critical care settings."
        else:
            text = "test chunk"
        return [
            {
                "chunk_id": "arxiv:1234.5678:0",
                "source": "arxiv",
                "doc_id": "1234.5678",
                "title": "A test paper",
                "url": "https://arxiv.org/abs/1234.5678",
                "text": text,
                "score": 0.91,
            }
        ][:top_k]


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint(monkeypatch):
    from app import api

    class FakeVectorStore:
        @classmethod
        def load(cls, path: str):
            return DummyStore()

    monkeypatch.setattr(api, "get_vector_store_class", lambda: FakeVectorStore)

    client = TestClient(app)
    response = client.post(
        "/query",
        json={"query": "molecule generation", "top_k": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "molecule generation"
    assert len(body["results"]) == 1
    assert body["results"][0]["source"] == "arxiv"


def test_build_index_endpoint(monkeypatch):
    from app import api

    class FakeStore:
        def __init__(self, model_name: str = ""):
            self.model_name = model_name

        def build(self, chunks):
            self.chunks = chunks

        def save(self, out_dir: str):
            self.out_dir = out_dir

    def fake_mine_documents(topic: str, arxiv_docs: int = 4, pubmed_docs: int = 4, timeout: int = 30):
        return [
            RawDocument(
                source="arxiv",
                doc_id="1234.5678",
                title="A test paper",
                url="https://arxiv.org/abs/1234.5678",
                text="Sentence one. Sentence two.",
            )
        ]

    monkeypatch.setattr(api, "mine_documents", fake_mine_documents)
    monkeypatch.setattr(api, "get_vector_store_class", lambda: FakeStore)
    monkeypatch.setenv("SCISEARCH_VECTOR_STORE_DIR", "data/test_store")

    client = TestClient(app)
    response = client.post(
        "/build-index",
        json={"topic": "test topic", "arxiv_docs": 1, "pubmed_docs": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["topic"] == "test topic"
    assert body["mined_documents"] == 1
    assert body["chunks"] >= 1
    assert body["out_dir"] == "data/test_store"


def test_build_index_requires_positive_source_count():
    client = TestClient(app)
    response = client.post(
        "/build-index",
        json={"topic": "test topic", "arxiv_docs": 0, "pubmed_docs": 0},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert "request_id" in body


def test_request_id_header_on_response():
    client = TestClient(app)
    response = client.get("/health", headers={"x-request-id": "req-123"})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "req-123"


def test_answer_endpoint(monkeypatch):
    from app import api

    class FakeVectorStore:
        @classmethod
        def load(cls, path: str):
            return DummyStore()

    monkeypatch.setattr(api, "get_vector_store_class", lambda: FakeVectorStore)

    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "question": "What are dangerous conditions and what are the risk factors?",
            "max_substeps": 4,
            "top_k_per_step": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["question"].startswith("What are dangerous conditions")
    assert len(body["sub_questions"]) >= 1
    assert len(body["steps"]) >= 1
    assert "Answer based on indexed documents" in body["answer"]
    assert len(body["citations"]) >= 1


def test_evaluate_endpoint(monkeypatch):
    from app import api

    class FakeVectorStore:
        @classmethod
        def load(cls, path: str):
            return DummyStore()

    def fake_evaluate_answer_quality(**kwargs):
        return EvaluationResult(
            overall_score=0.88,
            correctness_score=0.9,
            faithfulness_score=0.85,
            fact_coverage_score=0.92,
            citation_recall_score=0.8,
            missing_facts=[],
            unsupported_sentences=[],
            retrieved_doc_ids=["1234.5678"],
        )

    monkeypatch.setattr(api, "get_vector_store_class", lambda: FakeVectorStore)
    monkeypatch.setattr(api, "evaluate_answer_quality", fake_evaluate_answer_quality)

    client = TestClient(app)
    response = client.post(
        "/evaluate",
        json={
            "question": "What are dangerous conditions and risk factors?",
            "reference_answer": "Serious conditions include sepsis; risk factors include delayed diagnosis.",
            "required_facts": ["sepsis is dangerous", "delayed diagnosis is a risk factor"],
            "expected_doc_ids": ["1234.5678"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["overall_score"] == 0.88
    assert body["fact_coverage_score"] == 0.92
    assert len(body["sub_questions"]) >= 1
    assert len(body["citations"]) >= 1
