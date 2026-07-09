"""Chroma vector store over the user's resume and parsed job descriptions.

Embeddings are chromadb's built-in ONNX MiniLM (all-MiniLM-L6-v2): fully local
and free (Groq exposes no embeddings API), ~80 MB one-time download, no torch.
Specialist agents call these functions as plain Python inside their nodes —
retrieval is deliberately not exposed as an LLM tool (keeps the kit flow at
one LLM call per agent).
"""
import re

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_DIR, RESUME_PATH

_client = None


def client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def resume_collection():
    return client().get_or_create_collection("resume")


def jobs_collection():
    return client().get_or_create_collection("jobs")


# --------------------------------------------------------------------- resume

def resume_full_text() -> str:
    """Raw resume text — used by the deterministic ATS scorer (never retrieval-dependent)."""
    return RESUME_PATH.read_text(encoding="utf-8")


def _split_resume(text: str) -> list[dict]:
    """Split on '## ' section headers first, then size-bound within sections."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    parts = re.split(r"(?m)^##\s+", text)
    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if i == 0:
            title, body = "Profile", part
        else:
            first, _, rest = part.partition("\n")
            title, body = first.strip(), rest.strip() or first
        for piece in splitter.split_text(body):
            chunks.append({"section": title, "text": f"[{title}] {piece}"})
    return chunks


def ingest_resume() -> int:
    """Idempotent rebuild of the resume collection (a resume is only ~15 chunks)."""
    chunks = _split_resume(resume_full_text())
    col = resume_collection()
    existing = col.get()
    if existing["ids"]:
        col.delete(ids=existing["ids"])
    col.add(
        ids=[f"resume-{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[{"section": c["section"]} for c in chunks],
    )
    return len(chunks)


def search_resume(query: str, k: int = 5) -> list[str]:
    col = resume_collection()
    n = col.count()
    if n == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(k, n))
    return res["documents"][0]


# ----------------------------------------------------------------------- jobs

def index_job(application_id: int, company: str, role: str, condensed: str) -> None:
    """One condensed doc per tracked job -> enables semantic queries across applications."""
    jobs_collection().upsert(
        ids=[f"job-{application_id}"],
        documents=[condensed],
        metadatas=[{"application_id": application_id, "company": company, "role": role}],
    )


def search_jobs(query: str, k: int = 3) -> list[dict]:
    col = jobs_collection()
    n = col.count()
    if n == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(k, n))
    return [
        {"document": doc, **meta}
        for doc, meta in zip(res["documents"][0], res["metadatas"][0])
    ]
