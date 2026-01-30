import logging
from typing import List

import google.generativeai as genai
from jinja2 import Template

from app.core.config import settings
from app.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class QueryService:
    """Business logic for querying documents."""
    
    def __init__(self, repository: DocumentRepository):
        self.repository = repository
        self.model = genai.GenerativeModel(settings.MODEL_NAME)
        self.prompt_template = self._load_prompt_template()
    
    def query(self, question: str, top_k: int = 3) -> dict:
        """Query documents and generate an answer."""
        query_embedding = self.repository.get_embedding(question, task_type="retrieval_query")
        
        results = self.repository.search(query_embedding, top_k)
        
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
        
        context = "\n\n---\n\n".join(relevant_chunks)
        
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
    
    def _load_prompt_template(self) -> Template:
        """Load the prompt template from file."""
        with open(settings.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            return Template(f.read())
