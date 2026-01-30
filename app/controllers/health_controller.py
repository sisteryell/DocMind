from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", message="RAG system is running")
