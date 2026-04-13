"""
vectorstore.py
ChromaDB vector store with HuggingFace sentence-transformer embeddings.
Manages two collections:
  - resume_achievements : user's project/achievement chunks
  - job_listings        : scraped job description chunks
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PERSIST_DIR       = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_RESUME = os.getenv("CHROMA_COLLECTION_RESUME", "resume_achievements")
COLLECTION_JOBS   = os.getenv("CHROMA_COLLECTION_JOBS", "job_listings")
EMBED_MODEL       = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class EmbeddingModel:
    """Singleton HuggingFace sentence-transformer embedder."""

    _instance: Optional["EmbeddingModel"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer(EMBED_MODEL)
        return cls._instance

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class VectorStore:
    """
    Wraps ChromaDB with a clean API for upserting and querying documents.
    Uses persistent storage so data survives restarts.
    """

    def __init__(self):
        os.makedirs(PERSIST_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedder = EmbeddingModel()

        # Create or load collections
        self.resume_col = self.client.get_or_create_collection(
            name=COLLECTION_RESUME,
            metadata={"hnsw:space": "cosine"},
        )
        self.jobs_col = self.client.get_or_create_collection(
            name=COLLECTION_JOBS,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_resume_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Insert resume / achievement chunks.
        Each chunk: {"text": str, "metadata": dict}
        Returns count of inserted docs.
        """
        if not chunks:
            return 0

        ids        = [str(uuid.uuid4()) for _ in chunks]
        texts      = [c["text"] for c in chunks]
        metadatas  = [c.get("metadata", {}) for c in chunks]
        embeddings = self.embedder.embed(texts)

        self.resume_col.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def upsert_job_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Insert scraped job listing chunks.
        Each chunk: {"text": str, "metadata": {"title", "company", "url", ...}}
        """
        if not chunks:
            return 0

        ids        = [str(uuid.uuid4()) for _ in chunks]
        texts      = [c["text"] for c in chunks]
        metadatas  = [c.get("metadata", {}) for c in chunks]
        embeddings = self.embedder.embed(texts)

        self.jobs_col.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query_resume(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top_k relevant achievement chunks for a query."""
        emb = self.embedder.embed_single(query)
        results = self.resume_col.query(
            query_embeddings=[emb],
            n_results=min(top_k, self.resume_col.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        return self._format_results(results)

    def query_jobs(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top_k relevant job chunks for a query."""
        emb = self.embedder.embed_single(query)
        count = self.jobs_col.count()
        if count == 0:
            return []
        results = self.jobs_col.query(
            query_embeddings=[emb],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        return self._format_results(results)

    def semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Return cosine similarity (0–1) between two texts."""
        vec_a = self.embedder.embed_single(text_a)
        vec_b = self.embedder.embed_single(text_b)
        return self.embedder.similarity(vec_a, vec_b)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_results(raw: Dict) -> List[Dict[str, Any]]:
        docs      = raw.get("documents", [[]])[0]
        metas     = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        results = []
        for doc, meta, dist in zip(docs, metas, distances):
            results.append({
                "text":       doc,
                "metadata":   meta,
                "similarity": round(1 - dist, 4),   # cosine distance → similarity
            })
        return results

    def clear_jobs(self):
        """Remove all job listings (call before each new scrape session)."""
        self.client.delete_collection(COLLECTION_JOBS)
        self.jobs_col = self.client.get_or_create_collection(
            name=COLLECTION_JOBS,
            metadata={"hnsw:space": "cosine"},
        )

    def stats(self) -> Dict[str, int]:
        return {
            "resume_chunks": self.resume_col.count(),
            "job_chunks":    self.jobs_col.count(),
        }