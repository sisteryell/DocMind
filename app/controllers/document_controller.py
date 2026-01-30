from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from datetime import datetime
import logging

from app.models.schemas import DocumentUpload, DocumentInfo
from app.core.dependencies import get_document_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUpload)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF and TXT document."""
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    
    try:
        content = await file.read()
        service = get_document_service()
        result = service.upload_document(file.filename, content)
        
        return DocumentUpload(
            id=result["id"],
            filename=result["filename"],
            uploaded_at=datetime.fromisoformat(result["uploaded_at"]),
            chunks_count=result["chunks_count"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        logger.error(f"Upload error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.get("", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents."""
    service = get_document_service()
    docs = service.get_all_documents()
    return [DocumentInfo(**doc) for doc in docs]


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    service = get_document_service()
    if service.delete_document(doc_id):
        return {"message": "Document deleted successfully"}
    raise HTTPException(status_code=404, detail="Document not found")


@router.post("/reset")
async def reset_all_documents():
    """Delete all documents and clear the database."""
    service = get_document_service()
    if service.delete_all_documents():
        return {"message": "All documents deleted successfully"}
    raise HTTPException(status_code=500, detail="Failed to delete documents")
