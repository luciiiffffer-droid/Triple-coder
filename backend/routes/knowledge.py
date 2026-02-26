"""
Knowledge base ingestion endpoint.
"""

from fastapi import APIRouter, Depends
from models.entities import User
from models.schemas import KnowledgeIngestRequest, KnowledgeIngestResponse
from services.auth_service import require_admin
from services.vector_service import ingest_document

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.post("/ingest", response_model=KnowledgeIngestResponse)
async def ingest(
    req: KnowledgeIngestRequest,
    admin: User = Depends(require_admin),
):
    total = await ingest_document(req.title, req.content, req.category)
    return KnowledgeIngestResponse(
        status="ok",
        documents_indexed=total,
        message=f"Document '{req.title}' ingested successfully",
    )
