"""Dependency injection for services."""

from app.repositories.document_repository import DocumentRepository
from app.services.document_service import DocumentService
from app.services.query_service import QueryService

# Singleton instances
_document_repository = None
_document_service = None
_query_service = None


def get_document_repository() -> DocumentRepository:
    global _document_repository
    if _document_repository is None:
        _document_repository = DocumentRepository()
    return _document_repository


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        repo = get_document_repository()
        _document_service = DocumentService(repo)
    return _document_service


def get_query_service() -> QueryService:
    global _query_service
    if _query_service is None:
        repo = get_document_repository()
        _query_service = QueryService(repo)
    return _query_service
