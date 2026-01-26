import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from pypdf import PdfReader
from typing import List
import uuid
from jinja2 import Template

# Load environment variables from .env file
load_dotenv()


class RAGSystem:
    def __init__(self):
        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        
        # Get model configuration from environment
        model_name = os.getenv("MODEL_NAME")
        if not model_name:
            raise ValueError("MODEL_NAME not found. Please add it to your .env file")
        
        embedding_model = os.getenv("EMBEDDING_MODEL_NAME")
        if not embedding_model:
            raise ValueError("EMBEDDING_MODEL_NAME not found. Please add it to your .env file")
        
        self.model = genai.GenerativeModel(model_name)
        self.embed_model = embedding_model
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Store document metadata
        self.documents = {}
        self._load_metadata()
        
        # Load prompt template
        template_path = "prompts/base_prompt.txt"
        with open(template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = Template(f.read())
    
    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text content from a PDF file.
        
        Args:
            file_content: The PDF file content as bytes.
        
        Returns:
            str: The extracted text from all pages of the PDF.
        
        Raises:
            ValueError: If the PDF is encrypted, if no text can be extracted
                (e.g., image-based/scanned PDF), or if the PDF cannot be read.
        """
        import io
        try:
            reader = PdfReader(io.BytesIO(file_content))
            
            # Check if PDF is encrypted
            if reader.is_encrypted:
                raise ValueError("PDF is encrypted. Please provide an unencrypted PDF.")
            
            text = ""
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    text += page_text
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
            
            if not text.strip():
                raise ValueError(
                    "Could not extract any text from the PDF. "
                    "This might be a scanned/image-based PDF. "
                    "Please provide a text-based PDF document."
                )
            
            return text
        except Exception as e:
            if "encrypted" in str(e).lower():
                raise ValueError("PDF is encrypted. Please provide an unencrypted PDF.")
            raise
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for embedding generation.
        
        Args:
            text:
                The text to split into chunks.
            
            chunk_size:
                The maximum size of each chunk in characters. Defaults to 1000.
            
            overlap:
                The number of characters to overlap between consecutive chunks.
                Defaults to 200. This ensures context continuity across chunks.
        
        Returns:
            List[str]: A list of text chunks, each containing up to chunk_size
                characters with overlap between consecutive chunks.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        return chunks
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for document text using Gemini.
        
        Args:
            text:
                The text content to generate an embedding for.
        
        Returns:
            List[float]: A vector of floating-point numbers representing the
                semantic embedding of the input text, optimized for document
                retrieval tasks.
        """
        result = genai.embed_content(
            model=self.embed_model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    
    def _get_query_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a user query using Gemini.
        
        Args:
            text:
                The query text to generate an embedding for.
        
        Returns:
            List[float]: A vector of floating-point numbers representing the
                semantic embedding of the query, optimized for retrieval tasks.
        """
        result = genai.embed_content(
            model=self.embed_model,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    
    def add_document(self, filename: str, file_content: bytes) -> dict:
        """Add a PDF document to the RAG system for indexing and retrieval.
        
        Extracts text from the PDF, splits it into overlapping chunks, generates
        embeddings for each chunk, and stores them in ChromaDB for semantic search.
        
        Args:
            filename:
                The name of the PDF file being uploaded.
            
            file_content:
                The binary content of the PDF file.
        
        Returns:
            dict: A dictionary containing the document metadata with keys:
                - id (str): Unique identifier for the document
                - filename (str): Original filename
                - uploaded_at (str): ISO format timestamp of upload
                - chunks_count (int): Number of text chunks created
        
        Raises:
            ValueError: If text extraction fails or the PDF is encrypted.
        """
        doc_id = str(uuid.uuid4())
        
        print(f"Processing file: {filename}, size: {len(file_content)} bytes")
        
        # Extract text from PDF
        text = self._extract_text_from_pdf(file_content)
        
        print(f"Extracted text length: {len(text)} characters")
        
        if not text.strip():
            raise ValueError("Could not extract text from PDF")
        
        # Chunk the text
        chunks = self._chunk_text(text)
        
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings and store in ChromaDB
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_id = f"{doc_id}_{i}"
            embedding = self._get_embedding(chunk)
            
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"doc_id": doc_id, "filename": filename, "chunk_index": i}]
            )
        
        # Store document metadata
        from datetime import datetime
        self.documents[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "uploaded_at": datetime.now().isoformat(),
            "chunks_count": len(chunks)
        }
        self._save_metadata()
        
        return self.documents[doc_id]
    
    def query(self, question: str, top_k: int = 3) -> dict:
        """Query the RAG system to get answers from indexed documents.
        
        Generates an embedding for the question, retrieves the most relevant
        document chunks using semantic search, and uses Gemini to generate an
        answer based on the retrieved context.
        
        Args:
            question:
                The question to answer based on the indexed documents.
            
            top_k:
                The number of most relevant document chunks to retrieve.
                Defaults to 3.
        
        Returns:
            dict: A dictionary containing:
                - question (str): The original question
                - answer (str): The generated answer from the AI model
                - sources (List[str]): List of source document filenames used
        """
        # Get query embedding
        query_embedding = self._get_query_embedding(question)
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Get relevant chunks
        relevant_chunks = results['documents'][0] if results['documents'] else []
        sources = []
        if results['metadatas'] and results['metadatas'][0]:
            sources = list(set([m['filename'] for m in results['metadatas'][0]]))
        
        if not relevant_chunks:
            return {
                "question": question,
                "answer": "I don't have any documents to answer from. Please upload some PDFs first.",
                "sources": []
            }
        
        # Build context
        context = "\n\n---\n\n".join(relevant_chunks)
        
        # Render prompt using Jinja2 template
        full_prompt = self.prompt_template.render(
            context=context,
            question=question
        )
        
        response = self.model.generate_content(full_prompt)
        
        return {
            "question": question,
            "answer": response.text,
            "sources": sources
        }
    
    def get_all_documents(self) -> List[dict]:
        """Retrieve metadata for all documents in the RAG system.
        
        Returns:
            List[dict]: A list of dictionaries, each containing document
                metadata (id, filename, uploaded_at, chunks_count).
        """
        return list(self.documents.values())
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a specific document and all its associated chunks from the system.
        
        Removes the document's text chunks from ChromaDB and deletes its
        metadata from the system.
        
        Args:
            doc_id:
                The unique identifier of the document to delete.
        
        Returns:
            bool: True if the document was successfully deleted, False if the
                document was not found.
        """
        if doc_id not in self.documents:
            return False
        
        # Delete from ChromaDB - get all chunk IDs for this document
        results = self.collection.get(
            where={"doc_id": doc_id}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
        
        # Remove from metadata
        del self.documents[doc_id]
        self._save_metadata()
        return True
    
    def delete_all_documents(self) -> bool:
        """Delete all documents and clear the database"""
        try:
            # Delete the collection
            self.chroma_client.delete_collection(name="documents")
            # Recreate the collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            # Clear metadata
            self.documents = {}
            self._save_metadata()
            return True
        except Exception as e:
            print(f"Error deleting all documents: {e}")
            return False
    
    def _save_metadata(self):
        """Save document metadata to disk"""
        import json
        metadata_file = "./chroma_db/metadata.json"
        os.makedirs("./chroma_db", exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(self.documents, f)
    
    def _load_metadata(self):
        """Load document metadata from disk"""
        import json
        metadata_file = "./chroma_db/metadata.json"
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    self.documents = json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.documents = {}


# Singleton instance
rag_system = None

def get_rag_system() -> RAGSystem:
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem()
    return rag_system
