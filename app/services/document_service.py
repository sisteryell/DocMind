import io
import uuid
import logging
from datetime import datetime
from typing import List

from pypdf import PdfReader

from app.core.config import settings
from app.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentService:
    """Business logic for document management."""
    
    def __init__(self, repository: DocumentRepository):
        self.repository = repository
    
    def upload_document(self, filename: str, file_content: bytes) -> dict:
        """Process and store a PDF document."""
        doc_id = str(uuid.uuid4())
        
        logger.info(f"Processing file: {filename}, size: {len(file_content)} bytes")
        
        text = self._extract_text_from_pdf(file_content)
        logger.info(f"Extracted text length: {len(text)} characters")
        
        if not text.strip():
            raise ValueError("Could not extract text from PDF")
        
        chunks = self._chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks")
        
        self.repository.add_chunks(doc_id, filename, chunks)
        
        metadata = {
            "id": doc_id,
            "filename": filename,
            "uploaded_at": datetime.now().isoformat(),
            "chunks_count": len(chunks)
        }
        self.repository.save_document_metadata(doc_id, metadata)
        
        return metadata
    
    def get_all_documents(self) -> List[dict]:
        """Get all uploaded documents."""
        return self.repository.get_all_documents()
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a specific document."""
        return self.repository.delete_document(doc_id)
    
    def delete_all_documents(self) -> bool:
        """Delete all documents."""
        return self.repository.delete_all()
    
    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            reader = PdfReader(io.BytesIO(file_content))
            
            if reader.is_encrypted:
                raise ValueError("PDF is encrypted. Please provide an unencrypted PDF.")
            
            text = ""
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    text += page_text
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
            
            if not text.strip():
                raise ValueError(
                    "Could not extract any text from the PDF. "
                    "This might be a scanned/image-based PDF."
                )
            
            return text
        except Exception as e:
            if "encrypted" in str(e).lower():
                raise ValueError("PDF is encrypted. Please provide an unencrypted PDF.")
            raise
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        return chunks
