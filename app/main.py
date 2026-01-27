from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.models import QueryRequest, QueryResponse, DocumentUpload, DocumentInfo, HealthResponse
from app.rag import get_rag_system
from typing import List
import logging
from langfuse import Langfuse
import os

# Configure logger
logger = logging.getLogger(__name__)

# Initialize Langfuse (only if keys are configured)
langfuse_client = None
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    langfuse_client = Langfuse()
    logger.info("Langfuse observability enabled")
else:
    logger.info("Langfuse observability disabled (no API keys found)")

app = FastAPI(
    title="Simple RAG System",
    description="A simple RAG system using Google Gemini and ChromaDB",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def langfuse_middleware(request: Request, call_next):
    """Middleware to trace all HTTP requests with Langfuse."""
    # Skip tracing for static files and health checks
    if request.url.path.startswith("/static") or request.url.path == "/health":
        return await call_next(request)
    
    if langfuse_client:
        trace = langfuse_client.trace(
            name=f"{request.method} {request.url.path}",
            metadata={
                "method": request.method,
                "path": request.url.path,
                "client_host": request.client.host if request.client else None
            },
            tags=["api", "fastapi"]
        )
        
        # Store trace in request state for access in endpoints
        request.state.langfuse_trace = trace
    
    response = await call_next(request)
    
    if langfuse_client and hasattr(request.state, "langfuse_trace"):
        request.state.langfuse_trace.update(
            metadata={"status_code": response.status_code}
        )
    
    return response


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    logger.info("Starting DocMind RAG System")
    if langfuse_client:
        logger.info("Langfuse tracing is active")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown."""
    logger.info("Shutting down DocMind RAG System")
    if langfuse_client:
        logger.info("Flushing Langfuse traces...")
        langfuse_client.flush()
        logger.info("Langfuse traces flushed successfully")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", message="RAG system is running")


@app.post("/documents/upload", response_model=DocumentUpload)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        content = await file.read()
        rag = get_rag_system()
        result = rag.add_document(file.filename, content)
        
        from datetime import datetime
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


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents"""
    rag = get_rag_system()
    docs = rag.get_all_documents()
    return [DocumentInfo(**doc) for doc in docs]


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document"""
    rag = get_rag_system()
    if rag.delete_document(doc_id):
        return {"message": "Document deleted successfully"}
    raise HTTPException(status_code=404, detail="Document not found")


@app.post("/documents/reset")
async def reset_all_documents():
    """Delete all documents and clear the database"""
    rag = get_rag_system()
    if rag.delete_all_documents():
        return {"message": "All documents deleted successfully"}
    raise HTTPException(status_code=500, detail="Failed to delete documents")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    try:
        rag = get_rag_system()
        result = rag.query(request.question, request.top_k)
        return QueryResponse(**result)
    except Exception as e:
        import traceback
        logger.error(f"Query error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
