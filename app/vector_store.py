from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .chunking import Chunk


class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store: FAISS | None = None

    def build(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        docs = [
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "url": chunk.url,
                },
            )
            for chunk in chunks
        ]

        self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def save(self, output_dir: str) -> None:
        if self.vector_store is None:
            raise RuntimeError("Index has not been built yet.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.vector_store.save_local(str(out))
        with (out / "store_config.json").open("w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name}, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, input_dir: str) -> "VectorStore":
        in_path = Path(input_dir)
        with (in_path / "store_config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)

        store = cls(model_name=config["model_name"])
        store.vector_store = FAISS.load_local(
            str(in_path),
            store.embeddings,
            allow_dangerous_deserialization=True,
        )
        return store

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if self.vector_store is None:
            raise RuntimeError("Index is not initialized.")

        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        payload: List[dict] = []

        for doc, distance in results:
            # Convert L2 distance to a bounded similarity-like score.
            score = 1.0 / (1.0 + float(distance))
            payload.append(
                {
                    "chunk_id": str(doc.metadata.get("chunk_id", "")),
                    "source": str(doc.metadata.get("source", "")),
                    "doc_id": str(doc.metadata.get("doc_id", "")),
                    "title": str(doc.metadata.get("title", "")),
                    "url": str(doc.metadata.get("url", "")),
                    "text": doc.page_content,
                    "score": score,
                }
            )

        return payload
