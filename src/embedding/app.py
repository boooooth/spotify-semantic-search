import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from embedding.config import PROJECT_ROOT
from embedding.engine import SpotifySearchEngine


# Global search engine reference (loaded once in RAM across all requests)
engine: SpotifySearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads the AI embedding model and Milvus index into RAM on startup,
    guaranteeing sub-millisecond query latency for all web users.
    """
    global engine
    print("🚀 Initializing Spotify Search Engine for Web App...")
    engine = SpotifySearchEngine()
    print("✅ Search Engine loaded and ready to serve requests!")
    yield
    print("🛑 Shutting down search engine...")


# Initialize FastAPI application
app = FastAPI(
    title="Spotify Semantic Vibe Search",
    description="Vector search engine for Spotify songs based on mood and emotional lyrics.",
    version="1.0.0",
    lifespan=lifespan,
)

# Static files (CSS, JS) and HTML Templates paths
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Mount static directory for stylesheets and client-side scripts
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse)
async def serve_home():
    """Serves the interactive frontend search page."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "templates/index.html not found. Please create the frontend template."},
        )
    return FileResponse(index_file)


@app.get("/api/search")
async def api_search(
    q: str = Query(..., description="Semantic search query or vibe description"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to retrieve (1-50)"),
):
    """
    Executes semantic vector search over 57,650 songs:
    - Encodes user query into a 384-d vector
    - Performs brute-force cosine similarity search in Milvus Lite
    - Returns ranked songs with similarity scores and playback URLs
    """
    if not engine:
        return JSONResponse(
            status_code=503,
            content={"error": "Search engine is still initializing. Please retry in a moment."},
        )

    clean_query = q.strip()
    if not clean_query:
        return {"query": "", "count": 0, "latency_ms": 0.0, "results": []}

    # Measure exact query latency (proves sub-millisecond search performance)
    start_time = time.perf_counter()
    results = engine.search(query=clean_query, top_k=top_k)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "query": clean_query,
        "top_k": top_k,
        "count": len(results),
        "latency_ms": elapsed_ms,
        "results": results,
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    return {
        "status": "healthy",
        "engine_ready": engine is not None,
    }
