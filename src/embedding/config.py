from pathlib import Path

# 1. PATHS
# Resolves the project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Ensure the artifacts directory exists for storing our Milvus database
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_PATH = DATA_DIR / "spotify_millsongdata.csv"
MILVUS_DB_PATH = ARTIFACTS_DIR / "spotify_songs.db"

# 2. EMBEDDING MODEL CONFIGURATION
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384     # vector dimension for all-MiniLM-L6-v2
MAX_LYRIC_WORDS = 200   # keep input under the 256-token limit ()
BATCH_SIZE = 128        # batch size

# 3. MILVUS COLLECTION & INDEX SETTINGS
COLLECTION_NAME = "spotify_songs"
METRIC_TYPE = "COSINE"  # cosine similarity
INDEX_TYPE = "FLAT"     # brute-force search

# Field max lengths (these info below got from the check_word_lengths.py script)
MAX_ARTIST_LENGTH = 128
MAX_SONG_LENGTH = 128
MAX_LINK_LENGTH = 256
MAX_SNIPPET_LENGTH = 512