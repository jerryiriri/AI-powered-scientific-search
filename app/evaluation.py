from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List

from langchain_huggingface import HuggingFaceEmbeddings


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class EvaluationResult:
    overall_score: float
    correctness_score: float
    faithfulness_score: float
    fact_coverage_score: float
    citation_recall_score: float
    missing_facts: List[str]
    unsupported_sentences: List[str]
    retrieved_doc_ids: List[str]


def _split_sentences(text: str) -> List[str]:
    parts = [segment.strip() for segment in _SENTENCE_SPLIT.split(text) if segment.strip()]
    return [p for p in parts if len(p) >= 8]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _max_similarity(target: str, candidates: Iterable[str], embeddings: HuggingFaceEmbeddings) -> float:
    candidates_list = [c for c in candidates if c.strip()]
    if not target.strip() or not candidates_list:
        return 0.0

    target_vec = embeddings.embed_query(target)
    candidate_vecs = embeddings.embed_documents(candidates_list)
    return max(_cosine(target_vec, vec) for vec in candidate_vecs)


def evaluate_answer_quality(
    *,
    question: str,
    generated_answer: str,
    reference_answer: str,
    required_facts: List[str],
    expected_doc_ids: List[str],
    evidence_texts: List[str],
    cited_doc_ids: List[str],
    embedding_model_name: str,
    fact_threshold: float = 0.62,
    faithfulness_threshold: float = 0.58,
) -> EvaluationResult:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    answer_sentences = _split_sentences(generated_answer)
    evidence_sentences: List[str] = []
    for text in evidence_texts:
        evidence_sentences.extend(_split_sentences(text))

    correctness = _max_similarity(generated_answer, [reference_answer], embeddings)

    missing_facts: List[str] = []
    if required_facts:
        covered = 0
        for fact in required_facts:
            score = _max_similarity(fact, answer_sentences, embeddings)
            if score >= fact_threshold:
                covered += 1
            else:
                missing_facts.append(fact)
        fact_coverage = covered / len(required_facts)
    else:
        fact_coverage = 1.0

    unsupported_sentences: List[str] = []
    factual_sentences = [
        sentence
        for sentence in answer_sentences
        if not sentence.lower().startswith("synthesis:") and "based on indexed documents" not in sentence.lower()
    ]

    if factual_sentences:
        supported = 0
        for sentence in factual_sentences:
            score = _max_similarity(sentence, evidence_sentences, embeddings)
            if score >= faithfulness_threshold:
                supported += 1
            else:
                unsupported_sentences.append(sentence)
        faithfulness = supported / len(factual_sentences)
    else:
        faithfulness = 0.0

    expected_set = {doc_id for doc_id in expected_doc_ids if doc_id}
    cited_set = {doc_id for doc_id in cited_doc_ids if doc_id}
    citation_recall = len(expected_set & cited_set) / len(expected_set) if expected_set else 1.0

    overall = (
        0.35 * max(0.0, min(1.0, correctness))
        + 0.30 * fact_coverage
        + 0.20 * faithfulness
        + 0.15 * citation_recall
    )

    return EvaluationResult(
        overall_score=round(overall, 4),
        correctness_score=round(max(0.0, min(1.0, correctness)), 4),
        faithfulness_score=round(max(0.0, min(1.0, faithfulness)), 4),
        fact_coverage_score=round(max(0.0, min(1.0, fact_coverage)), 4),
        citation_recall_score=round(max(0.0, min(1.0, citation_recall)), 4),
        missing_facts=missing_facts,
        unsupported_sentences=unsupported_sentences,
        retrieved_doc_ids=sorted(cited_set),
    )
