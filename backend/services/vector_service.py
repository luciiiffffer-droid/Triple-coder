"""
FAISS-backed vector store for knowledge base (RAG).
"""

import os
import json
import numpy as np
from typing import List, Optional
from config import settings
from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed — vector search disabled")

# In-memory stores (populated at startup / ingestion)
_index: Optional[object] = None
_documents: List[dict] = []
_dimension = 1536  # text-embedding-3-small dimension


def _ensure_dir():
    os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)


async def _get_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI."""
    import openai
    client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


async def ingest_document(title: str, content: str, category: str = "general") -> int:
    """Add a document to the vector index."""
    global _index, _documents

    if not FAISS_AVAILABLE:
        logger.warning("FAISS not available — skipping ingestion")
        return 0

    _ensure_dir()

    embedding = await _get_embedding(content)
    vec = np.array([embedding], dtype="float32")

    if _index is None:
        _index = faiss.IndexFlatL2(_dimension)

    _index.add(vec)
    _documents.append({"title": title, "content": content, "category": category})

    # Persist
    faiss.write_index(_index, os.path.join(settings.VECTOR_STORE_PATH, "index.faiss"))
    with open(os.path.join(settings.VECTOR_STORE_PATH, "docs.json"), "w") as f:
        json.dump(_documents, f)

    logger.info(f"Ingested document '{title}' — index size: {_index.ntotal}")
    return _index.ntotal


async def search(query: str, top_k: int = 3) -> str:
    """Search the knowledge base and return concatenated context."""
    global _index, _documents

    if not FAISS_AVAILABLE or _index is None or _index.ntotal == 0:
        return ""

    try:
        embedding = await _get_embedding(query)
        vec = np.array([embedding], dtype="float32")
        distances, indices = _index.search(vec, min(top_k, _index.ntotal))

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(_documents):
                doc = _documents[idx]
                results.append(f"[{doc['title']}]: {doc['content']}")

        return "\n\n".join(results)

    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return ""


def load_index():
    """Load persisted FAISS index from disk."""
    global _index, _documents

    if not FAISS_AVAILABLE:
        return

    _ensure_dir()
    index_path = os.path.join(settings.VECTOR_STORE_PATH, "index.faiss")
    docs_path = os.path.join(settings.VECTOR_STORE_PATH, "docs.json")

    if os.path.exists(index_path):
        _index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index with {_index.ntotal} vectors")

    if os.path.exists(docs_path):
        with open(docs_path) as f:
            _documents = json.load(f)
