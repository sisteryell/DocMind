from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DocumentUpload(BaseModel):
    """Response model for document upload."""
    id: str
    filename: str
    uploaded_at: datetime
    chunks_count: int


class QueryRequest(BaseModel):
    """Request model for querying documents."""
    question: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    """Response model for query results."""
    question: str
    answer: str
    sources: list[str]


class DocumentInfo(BaseModel):
    """Model for document information."""
    id: str
    filename: str
    uploaded_at: str
    chunks_count: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    message: str


class ErrorResponse(BaseModel):
    """Response model for errors."""
    detail: str


class SuccessResponse(BaseModel):
    """Response model for success messages."""
    message: str
