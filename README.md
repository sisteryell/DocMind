# Simple RAG System

A simple PDF Question-Answering system using Google Gemini and ChromaDB, with full observability powered by Langfuse.

## Features

- 📤 Upload PDF documents
- 🔍 Ask questions about your documents
- 🤖 Get AI-powered answers using Google Gemini
- 💾 Persistent storage with ChromaDB
- 🎨 Clean web interface
- 📊 **Full observability with Langfuse** (track performance, tokens, costs)

## Quick Start

### 1. Get Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

### 2. (Optional) Get Langfuse Keys for Observability

See [LANGFUSE_QUICKSTART.md](LANGFUSE_QUICKSTART.md) for 2-minute setup to enable:
- Performance monitoring
- Token usage tracking
- Error debugging
- Cost analysis

### 3. Run with Docker

```bash

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_actual_api_key

# Build and run
docker-compose up --build
```

### 3. Run without Docker

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable
# Windows:
set GEMINI_API_KEY=your_api_key

# Linux/Mac:
export GEMINI_API_KEY=your_api_key

# Run the app
uvicorn app.main:app --reload
```

## Usage

1. Open http://localhost:8000 in your browser
2. Upload PDF documents
3. Ask questions about your documents
4. Get AI-powered answers!

## API Endpoints

- `GET /` - Web UI
- `GET /health` - Health check
- `POST /documents/upload` - Upload a PDF
- `GET /documents` - List all documents
- `DELETE /documents/{id}` - Delete a document
- `POST /query` - Ask a question

## Observability with Langfuse

This project includes comprehensive tracing with Langfuse:

- **📊 Performance Monitoring**: Track latency for uploads, queries, embeddings, and LLM calls
- **💰 Cost Tracking**: Monitor token usage for embeddings and generation
- **🐛 Error Debugging**: Detailed error traces with context
- **📈 Analytics**: Understand user behavior and common queries

**Quick Setup**: See [LANGFUSE_QUICKSTART.md](LANGFUSE_QUICKSTART.md)  
**Full Guide**: See [LANGFUSE_INTEGRATION.md](LANGFUSE_INTEGRATION.md)

**Optional**: The app works perfectly without Langfuse. Simply don't set the Langfuse environment variables.

## Project Structure

```
DocMind/
├── app/
│   ├── __init__.py
│   ├── main.py      # FastAPI application
│   ├── models.py    # Pydantic models
│   └── rag.py       # RAG logic (Gemini + ChromaDB)
├── static/
│   └── index.html   # Web UI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```
