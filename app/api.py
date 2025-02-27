from __future__ import annotations

import logging
import time
from uuid import uuid4
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, model_validator

from .chunking import chunk_documents
from .config import get_settings
from .evaluation import evaluate_answer_quality
from .logging_utils import RequestLoggerAdapter, configure_logging
from .miner import mine_documents
from .reasoning import compose_answer, decompose_question
from .tracing import get_trace_client

initial_settings = get_settings()
configure_logging(initial_settings.log_level)
logger = RequestLoggerAdapter(logging.getLogger(__name__), {})
trace_client = get_trace_client()

app = FastAPI(title=initial_settings.app_name, version=initial_settings.app_version)


def get_vector_store_class():
    # Delayed import keeps API startup/tests lightweight.
    from .vector_store import VectorStore

    return VectorStore


class BuildIndexRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    arxiv_docs: int = Field(default=8, ge=0, le=20)
    pubmed_docs: int = Field(default=0, ge=0, le=20)
    chunk_size: int = Field(default=550, ge=200, le=4000)
    chunk_overlap: int = Field(default=1, ge=0, le=10)

    @model_validator(mode="after")
    def validate_sources(self) -> "BuildIndexRequest":
        if self.arxiv_docs + self.pubmed_docs <= 0:
            raise ValueError("At least one source document count must be positive.")
        return self


class IndexedDocument(BaseModel):
    source: str
    doc_id: str
    title: str
    url: str
    pdf_url: str | None = None
    pdf_path: str | None = None


class BuildIndexResponse(BaseModel):
    topic: str
    mined_documents: int
    chunks: int
    out_dir: str
    documents: List[IndexedDocument]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResult(BaseModel):
    chunk_id: str
    source: str
    doc_id: str
    title: str
    url: str
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    top_k: int
    results: List[QueryResult]


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=3)
    max_substeps: int | None = Field(default=None, ge=1, le=8)
    top_k_per_step: int | None = Field(default=None, ge=1, le=10)


class AnswerStep(BaseModel):
    sub_question: str
    results: List[QueryResult]


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    url: str
    score: float


class AnswerResponse(BaseModel):
    question: str
    sub_questions: List[str]
    steps: List[AnswerStep]
    answer: str
    citations: List[Citation]


class EvaluateRequest(BaseModel):
    question: str = Field(..., min_length=3)
    reference_answer: str = Field(..., min_length=3)
    required_facts: List[str] = Field(default_factory=list)
    expected_doc_ids: List[str] = Field(default_factory=list)
    max_substeps: int | None = Field(default=None, ge=1, le=8)
    top_k_per_step: int | None = Field(default=None, ge=1, le=10)


class EvaluateResponse(BaseModel):
    question: str
    answer: str
    sub_questions: List[str]
    overall_score: float
    correctness_score: float
    faithfulness_score: float
    fact_coverage_score: float
    citation_recall_score: float
    missing_facts: List[str]
    unsupported_sentences: List[str]
    citations: List[Citation]


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid4()))
        request.state.request_id = request_id
        started = time.perf_counter()

        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["x-request-id"] = request_id

        logger.info(
            "request.completed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response


app.add_middleware(RequestContextMiddleware)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    req_id = _request_id(request)
    logger.warning(
        "request.http_error",
        extra={
            "request_id": req_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": exc.status_code,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(code="http_error", message=str(exc.detail), request_id=req_id).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    req_id = _request_id(request)
    logger.warning(
        "request.validation_error",
        extra={"request_id": req_id, "path": request.url.path, "method": request.method, "status_code": 422},
    )
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(code="validation_error", message="Invalid request payload.", request_id=req_id).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    req_id = _request_id(request)
    logger.exception(
        "request.unhandled_error",
        extra={"request_id": req_id, "path": request.url.path, "method": request.method, "status_code": 500},
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(code="internal_error", message="Internal server error.", request_id=req_id).model_dump(),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    settings = get_settings()
    vector_store_class = get_vector_store_class()
    try:
        vector_store_class.load(settings.vector_store_dir)
        return {"status": "ready", "vector_store_dir": settings.vector_store_dir}
    except Exception:
        return {"status": "not_ready", "vector_store_dir": settings.vector_store_dir}


@app.post("/build-index", response_model=BuildIndexResponse)
def build_index(payload: BuildIndexRequest) -> BuildIndexResponse:
    settings = get_settings()
    try:
        vector_store_class = get_vector_store_class()
        docs = mine_documents(
            topic=payload.topic,
            arxiv_docs=payload.arxiv_docs,
            pubmed_docs=payload.pubmed_docs,
            timeout=settings.request_timeout_seconds,
        )
        chunks = chunk_documents(
            docs,
            max_chars=payload.chunk_size,
            overlap_sentences=payload.chunk_overlap,
        )
        store = vector_store_class(model_name=settings.embedding_model_name)
        store.build(chunks)
        store.save(settings.vector_store_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build index: {exc}") from exc

    response = BuildIndexResponse(
        topic=payload.topic,
        mined_documents=len(docs),
        chunks=len(chunks),
        out_dir=settings.vector_store_dir,
        documents=[
            IndexedDocument(
                source=doc.source,
                doc_id=doc.doc_id,
                title=doc.title,
                url=doc.url,
                pdf_url=doc.pdf_url,
                pdf_path=doc.pdf_path,
            )
            for doc in docs
        ],
    )
    trace_client.capture(
        name="build-index",
        input_payload=payload.model_dump(),
        output_payload={"mined_documents": response.mined_documents, "chunks": response.chunks},
        metadata={"vector_store_dir": settings.vector_store_dir},
    )
    return response


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    settings = get_settings()
    try:
        vector_store_class = get_vector_store_class()
        store = vector_store_class.load(settings.vector_store_dir)
        results = store.search(payload.query, top_k=payload.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Vector store not found.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to query index: {exc}") from exc

    normalized = [QueryResult(**item) for item in results]
    response = QueryResponse(query=payload.query, top_k=payload.top_k, results=normalized)
    trace_client.capture(
        name="query",
        input_payload=payload.model_dump(),
        output_payload={"results_count": len(response.results)},
        metadata={"vector_store_dir": settings.vector_store_dir},
    )
    return response


@app.post("/answer", response_model=AnswerResponse)
def answer(payload: AnswerRequest) -> AnswerResponse:
    settings = get_settings()
    max_substeps = payload.max_substeps or settings.answer_max_substeps_default
    top_k_per_step = payload.top_k_per_step or settings.answer_top_k_per_step_default
    try:
        vector_store_class = get_vector_store_class()
        store = vector_store_class.load(settings.vector_store_dir)

        sub_questions = decompose_question(payload.question, max_steps=max_substeps)
        if not sub_questions:
            sub_questions = [payload.question]

        raw_steps = []
        for sub_question in sub_questions:
            results = store.search(sub_question, top_k=top_k_per_step)
            normalized = [QueryResult(**item) for item in results]
            raw_steps.append({"sub_question": sub_question, "results": normalized})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Vector store not found.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc

    answer_text = compose_answer(
        payload.question,
        [
            {
                "sub_question": step["sub_question"],
                "results": [item.model_dump() for item in step["results"]],
            }
            for step in raw_steps
        ],
    )

    citation_map: dict[str, Citation] = {}
    for step in raw_steps:
        for result in step["results"]:
            if result.chunk_id in citation_map:
                continue
            citation_map[result.chunk_id] = Citation(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                title=result.title,
                url=result.url,
                score=result.score,
            )

    response = AnswerResponse(
        question=payload.question,
        sub_questions=sub_questions,
        steps=[AnswerStep(sub_question=step["sub_question"], results=step["results"]) for step in raw_steps],
        answer=answer_text,
        citations=list(citation_map.values()),
    )
    trace_client.capture(
        name="answer",
        input_payload=payload.model_dump(),
        output_payload={
            "sub_questions": response.sub_questions,
            "citations_count": len(response.citations),
        },
        metadata={"vector_store_dir": settings.vector_store_dir},
    )
    return response


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(payload: EvaluateRequest) -> EvaluateResponse:
    settings = get_settings()
    max_substeps = payload.max_substeps or settings.answer_max_substeps_default
    top_k_per_step = payload.top_k_per_step or settings.answer_top_k_per_step_default

    try:
        vector_store_class = get_vector_store_class()
        store = vector_store_class.load(settings.vector_store_dir)

        sub_questions = decompose_question(payload.question, max_steps=max_substeps)
        if not sub_questions:
            sub_questions = [payload.question]

        raw_steps = []
        for sub_question in sub_questions:
            results = store.search(sub_question, top_k=top_k_per_step)
            normalized = [QueryResult(**item) for item in results]
            raw_steps.append({"sub_question": sub_question, "results": normalized})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Vector store not found.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate answer: {exc}") from exc

    answer_text = compose_answer(
        payload.question,
        [
            {
                "sub_question": step["sub_question"],
                "results": [item.model_dump() for item in step["results"]],
            }
            for step in raw_steps
        ],
    )

    citation_map: dict[str, Citation] = {}
    evidence_texts: List[str] = []
    cited_doc_ids: List[str] = []
    for step in raw_steps:
        for result in step["results"]:
            evidence_texts.append(result.text)
            cited_doc_ids.append(result.doc_id)
            if result.chunk_id in citation_map:
                continue
            citation_map[result.chunk_id] = Citation(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                title=result.title,
                url=result.url,
                score=result.score,
            )

    metrics = evaluate_answer_quality(
        question=payload.question,
        generated_answer=answer_text,
        reference_answer=payload.reference_answer,
        required_facts=payload.required_facts,
        expected_doc_ids=payload.expected_doc_ids,
        evidence_texts=evidence_texts,
        cited_doc_ids=cited_doc_ids,
        embedding_model_name=settings.embedding_model_name,
    )

    response = EvaluateResponse(
        question=payload.question,
        answer=answer_text,
        sub_questions=sub_questions,
        overall_score=metrics.overall_score,
        correctness_score=metrics.correctness_score,
        faithfulness_score=metrics.faithfulness_score,
        fact_coverage_score=metrics.fact_coverage_score,
        citation_recall_score=metrics.citation_recall_score,
        missing_facts=metrics.missing_facts,
        unsupported_sentences=metrics.unsupported_sentences,
        citations=list(citation_map.values()),
    )
    trace_client.capture(
        name="evaluate",
        input_payload=payload.model_dump(),
        output_payload={
            "overall_score": response.overall_score,
            "fact_coverage_score": response.fact_coverage_score,
            "faithfulness_score": response.faithfulness_score,
        },
        metadata={"vector_store_dir": settings.vector_store_dir},
    )
    return response
