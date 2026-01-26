from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DocumentUpload(BaseModel):
    """Model for document upload response"""
    id: str
    filename: str
    uploaded_at: datetime
    chunks_count: int


class QueryRequest(BaseModel):
    """Model for query request"""
    question: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    """Model for query response"""
    question: str
    answer: str
    sources: list[str]


class DocumentInfo(BaseModel):
    """Model for document information"""
    id: str
    filename: str
    uploaded_at: str
    chunks_count: int


class HealthResponse(BaseModel):
    """Model for health check response"""
    status: str
    message: str
