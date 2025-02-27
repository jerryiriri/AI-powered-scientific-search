from __future__ import annotations

from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader

ARXIV_API = "http://export.arxiv.org/api/query"
PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


@dataclass
class RawDocument:
    source: str
    doc_id: str
    title: str
    url: str
    text: str
    pdf_url: str | None = None
    pdf_path: str | None = None


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_pdf_text(pdf_path: Path) -> str:
    pages = PyPDFLoader(str(pdf_path)).load()
    return _clean_text(" ".join(page.page_content for page in pages))


def fetch_arxiv_docs(
    topic: str,
    max_docs: int = 8,
    timeout: int = 30,
    pdf_dir: str = "data/papers/arxiv",
) -> List[RawDocument]:
    if max_docs <= 0:
        return []

    output_dir = Path(pdf_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": max_docs,
    }
    response = requests.get(ARXIV_API, params=params, timeout=timeout)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    docs: List[RawDocument] = []

    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.find("atom:id", ns).text.rsplit("/", 1)[-1]
        title = _clean_text(entry.find("atom:title", ns).text)
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_path = output_dir / f"{arxiv_id.replace('/', '_')}.pdf"

        pdf_response = requests.get(pdf_url, timeout=timeout)
        pdf_response.raise_for_status()
        pdf_path.write_bytes(pdf_response.content)

        full_text = _extract_pdf_text(pdf_path)
        if len(full_text) < 500:
            # Rare parser failures fallback to summary to avoid empty docs.
            summary = entry.find("atom:summary", ns).text
            full_text = _clean_text(summary)

        text = f"Title: {title}\nSource URL: {abs_url}\nFull Paper Text: {full_text}"
        docs.append(
            RawDocument(
                source="arxiv",
                doc_id=arxiv_id,
                title=title,
                url=abs_url,
                text=text,
                pdf_url=pdf_url,
                pdf_path=str(pdf_path),
            )
        )
        time.sleep(0.25)

    return docs


def fetch_pubmed_docs(topic: str, max_docs: int = 4, timeout: int = 20) -> List[RawDocument]:
    params = {
        "db": "pubmed",
        "term": topic,
        "retmax": max_docs,
        "retmode": "json",
        "sort": "relevance",
    }
    response = requests.get(PUBMED_ESEARCH, params=params, timeout=timeout)
    response.raise_for_status()
    pmids = response.json().get("esearchresult", {}).get("idlist", [])

    docs: List[RawDocument] = []
    for pmid in pmids:
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        page = requests.get(url, timeout=timeout)
        page.raise_for_status()
        soup = BeautifulSoup(page.text, "html.parser")

        title_node = soup.find("h1", class_="heading-title")
        title = _clean_text(title_node.get_text(" ", strip=True) if title_node else f"PubMed {pmid}")

        abstract_sections = soup.select("div.abstract-content")
        if abstract_sections:
            abstract_text = " ".join(_clean_text(node.get_text(" ", strip=True)) for node in abstract_sections)
        else:
            abstract_text = ""

        if not abstract_text:
            continue

        text = f"Title: {title}\nAbstract: {abstract_text}"
        docs.append(RawDocument(source="pubmed", doc_id=pmid, title=title, url=url, text=text))
        time.sleep(0.25)

    return docs


def mine_documents(
    topic: str,
    arxiv_docs: int = 8,
    pubmed_docs: int = 0,
    timeout: int = 30,
) -> List[RawDocument]:
    docs: List[RawDocument] = []
    if arxiv_docs > 0:
        docs.extend(fetch_arxiv_docs(topic=topic, max_docs=arxiv_docs, timeout=timeout))
    if pubmed_docs > 0:
        docs.extend(fetch_pubmed_docs(topic=topic, max_docs=pubmed_docs, timeout=timeout))
    if not docs:
        raise RuntimeError("No documents were mined from ArXiv/PubMed for the given topic.")
    return docs
