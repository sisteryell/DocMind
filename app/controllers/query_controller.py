from fastapi import APIRouter, HTTPException
import logging

from app.models.schemas import QueryRequest, QueryResponse
from app.core.dependencies import get_query_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system."""
    try:
        service = get_query_service()
        result = service.query(request.question, request.top_k)
        return QueryResponse(**result)
    except Exception as e:
        import traceback
        logger.error(f"Query error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
