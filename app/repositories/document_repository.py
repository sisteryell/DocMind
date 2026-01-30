import os
import json
import logging
from typing import List, Optional
import chromadb
import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Data access layer for document storage and retrieval using ChromaDB."""
    
    def __init__(self):
        settings.validate()
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.embed_model = settings.EMBEDDING_MODEL_NAME
        
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.documents = {}
        self._load_metadata()
    
    def get_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generate embedding for text using Gemini."""
        result = genai.embed_content(
            model=self.embed_model,
            content=text,
            task_type=task_type
        )
        return result['embedding']
    
    def add_chunks(self, doc_id: str, filename: str, chunks: List[str]) -> int:
        """Store document chunks with embeddings in ChromaDB."""
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            embedding = self.get_embedding(chunk)
            
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"doc_id": doc_id, "filename": filename, "chunk_index": i}]
            )
        
        return len(chunks)
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> dict:
        """Search for relevant chunks using embedding similarity."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
    
    def save_document_metadata(self, doc_id: str, metadata: dict):
        """Save document metadata."""
        self.documents[doc_id] = metadata
        self._save_metadata()
    
    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get document metadata by ID."""
        return self.documents.get(doc_id)
    
    def get_all_documents(self) -> List[dict]:
        """Get all document metadata."""
        return list(self.documents.values())
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        if doc_id not in self.documents:
            return False
        
        results = self.collection.get(where={"doc_id": doc_id})
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
        
        del self.documents[doc_id]
        self._save_metadata()
        return True
    
    def delete_all(self) -> bool:
        """Delete all documents and reset the collection."""
        try:
            self.chroma_client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
            self.collection = self.chroma_client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            self.documents = {}
            self._save_metadata()
            return True
        except Exception as e:
            logger.error(f"Error deleting all documents: {e}")
            return False
    
    def _save_metadata(self):
        """Persist metadata to disk."""
        metadata_file = os.path.join(settings.CHROMA_DB_PATH, "metadata.json")
        os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(self.documents, f)
    
    def _load_metadata(self):
        """Load metadata from disk."""
        metadata_file = os.path.join(settings.CHROMA_DB_PATH, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    self.documents = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self.documents = {}
