import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "")
    
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    PROMPT_TEMPLATE_PATH: str = "prompts/base_prompt.txt"
    
    @classmethod
    def validate(cls):
        """Validate that required settings are present."""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        if not cls.MODEL_NAME:
            raise ValueError("MODEL_NAME not found. Please add it to your .env file")
        if not cls.EMBEDDING_MODEL_NAME:
            raise ValueError("EMBEDDING_MODEL_NAME not found. Please add it to your .env file")


settings = Settings()
