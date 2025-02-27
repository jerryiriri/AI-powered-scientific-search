from __future__ import annotations

import re
from typing import Iterable, List


_SPLIT_PATTERN = re.compile(
    r"\b(?:and|then|also|plus|as well as|while|compared to|versus|vs\.?|with)\b|[;]",
    flags=re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")

_STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "and",
    "or",
    "that",
    "this",
    "what",
    "which",
    "how",
    "why",
    "when",
    "where",
}


def decompose_question(question: str, max_steps: int = 4) -> List[str]:
    cleaned = " ".join(question.strip().split())
    if not cleaned:
        return []

    base = cleaned.rstrip(" ?")
    clauses = [part.strip(" ,") for part in _SPLIT_PATTERN.split(base)]
    clauses = [clause for clause in clauses if len(clause.split()) >= 3]

    if not clauses:
        clauses = [base]

    deduped: List[str] = []
    seen = set()
    for clause in clauses:
        norm = clause.lower()
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(clause)
        if len(deduped) >= max_steps:
            break

    return deduped


def _token_set(text: str) -> set[str]:
    tokens = [token.lower() for token in _WORD_PATTERN.findall(text)]
    return {token for token in tokens if token not in _STOP_WORDS}


def _best_snippet(text: str, query: str, max_chars: int = 260) -> str:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    if not sentences:
        compact = " ".join(text.split())
        return compact[:max_chars]

    query_tokens = _token_set(query)
    best_sentence = sentences[0]
    best_score = -1

    for sentence in sentences:
        score = len(_token_set(sentence) & query_tokens)
        if score > best_score:
            best_sentence = sentence
            best_score = score

    compact = " ".join(best_sentence.split())
    return compact if len(compact) <= max_chars else f"{compact[: max_chars - 3]}..."


def compose_answer(question: str, step_results: Iterable[dict]) -> str:
    steps = list(step_results)
    if not steps:
        return "I could not break the question into sub-steps."

    evidence_found = any(step.get("results") for step in steps)
    if not evidence_found:
        return (
            "I could not find enough evidence in the indexed documents to answer this question. "
            "Try rephrasing the question or indexing more relevant papers."
        )

    lines: List[str] = []
    lines.append("Answer based on indexed documents:")
    lines.append("")

    for idx, step in enumerate(steps, start=1):
        sub_question = step["sub_question"]
        hits = step.get("results", [])
        lines.append(f"Step {idx}: {sub_question}")
        if not hits:
            lines.append("- No direct evidence found in indexed chunks.")
            continue

        for hit in hits[:2]:
            snippet = _best_snippet(hit.get("text", ""), sub_question)
            title = hit.get("title", "Unknown source")
            doc_id = hit.get("doc_id", "")
            url = hit.get("url", "")
            lines.append(f"- {snippet} (source: {title} [{doc_id}] {url})")

    lines.append("")
    lines.append(
        "Synthesis: The points above are aggregated from the top matching sub-queries. "
        "Use the cited sources to validate details before making high-stakes decisions."
    )
    return "\n".join(lines)
