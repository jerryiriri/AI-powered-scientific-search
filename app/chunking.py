from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .miner import RawDocument


@dataclass
class Chunk:
    chunk_id: str
    source: str
    doc_id: str
    title: str
    url: str
    text: str


def _to_documents(docs: Iterable[RawDocument]) -> List[Document]:
    return [
        Document(
            page_content=doc.text,
            metadata={
                "source": doc.source,
                "doc_id": doc.doc_id,
                "title": doc.title,
                "url": doc.url,
            },
        )
        for doc in docs
    ]


def chunk_documents(
    docs: Iterable[RawDocument],
    max_chars: int = 1200,
    overlap_sentences: int = 1,
) -> List[Chunk]:
    # Approximate sentence overlap in characters for splitter compatibility.
    chunk_overlap = max(0, overlap_sentences) * 120

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=min(chunk_overlap, max(0, max_chars - 1)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    split_docs = splitter.split_documents(_to_documents(docs))
    chunk_counters: dict[tuple[str, str], int] = {}
    chunks: List[Chunk] = []

    for doc in split_docs:
        source = str(doc.metadata.get("source", "unknown"))
        doc_id = str(doc.metadata.get("doc_id", "unknown"))
        title = str(doc.metadata.get("title", ""))
        url = str(doc.metadata.get("url", ""))

        key = (source, doc_id)
        chunk_idx = chunk_counters.get(key, 0)
        chunk_counters[key] = chunk_idx + 1

        chunks.append(
            Chunk(
                chunk_id=f"{source}:{doc_id}:{chunk_idx}",
                source=source,
                doc_id=doc_id,
                title=title,
                url=url,
                text=doc.page_content,
            )
        )

    return chunks
