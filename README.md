# Spotify Semantic Search Engine

A semantic search engine built over the Spotify Million Song Dataset (~57,650 tracks) to retrieve songs based on descriptive "vibe", mood, or emotional queries (e.g., *"melancholic acoustic ballad for a rainy morning"*).

## Tech Stack
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Vector Database:** Milvus Lite (local file-based vector store)
- **Similarity Metric:** Cosine Similarity (`COSINE`)
- **Index Type:** `FLAT` (brute-force exact search with 100% recall)
- **Backend:** FastAPI, Uvicorn
- **Package Manager:** `uv`

## Quick Start

### 1. Install Dependencies
```bash
uv sync
uv add milvus-lite
```

### 2. Build the Vector Index
```bash
# Build test sample (1,000 songs):
uv run python -m embedding.build_index --sample-size 1000

# Or build full index (all 57,650 songs):
uv run python -m embedding.build_index
```

### 3. Test in Terminal
```bash
uv run python -m embedding.engine
```

### 4. Run the Web Application
```bash
uv run uvicorn embedding.app:app --reload
```
