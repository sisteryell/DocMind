from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.controllers.document_controller import router as document_router
from app.controllers.query_controller import router as query_router
from app.controllers.health_controller import router as health_router

app = FastAPI(
    title="DocMind RAG System",
    description="A RAG system using Google Gemini and ChromaDB",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Register routers
app.include_router(health_router)
app.include_router(document_router)
app.include_router(query_router)
