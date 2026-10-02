"""
Search engine query logic for Spotify Semantic Search.
Encodes user queries, searches the Milvus Lite vector database,
and formats ranked results with Spotify playback URLs and match percentages.
"""

import urllib.parse
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

from embedding.config import (
    MILVUS_DB_PATH,
    COLLECTION_NAME,
    MODEL_NAME,
)


def generate_spotify_url(artist: str, song: str) -> str:
    """
    Generates a direct Spotify search URL for instant playback:
    'https://open.spotify.com/search/<artist>+<song>'
    """
    query = f"{artist} {song}"
    encoded = urllib.parse.quote_plus(query)
    return f"https://open.spotify.com/search/{encoded}"


class SpotifySearchEngine:
    # Loads once on startup for sub-millisecond query responses.

    def __init__(self):
        print(f"Loading search engine model: '{MODEL_NAME}' ...")
        self.model = SentenceTransformer(MODEL_NAME)

        print(f"Connecting to Milvus Lite at: {MILVUS_DB_PATH} ...")
        self.client = MilvusClient(uri=str(MILVUS_DB_PATH))
        self.client.load_collection(collection_name=COLLECTION_NAME)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Executes semantic search over indexed song lyrics:
        1. Encodes query into a 384-d normalized vector
        2. Queries Milvus Lite for top_k nearest neighbors
        3. Formats results with match percentages and Spotify URLs
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Encode query using the exact same normalization as the index
        query_vector = self.model.encode(
            clean_query,
            normalize_embeddings=True
        ).tolist()

        # 2. Search Milvus Lite
        search_results = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=top_k,
            output_fields=["artist", "song", "link", "lyrics_snippet"],
        )

        # 3. Format ranked results
        formatted_results = []
        if not search_results or len(search_results) == 0:
            return formatted_results

        # search_results[0] holds the matches for our query vector
        for rank, hit in enumerate(search_results[0], start=1):
            entity = hit.get("entity", {})
            distance = hit.get("distance", 0.0)  # Cosine similarity score (-1.0 to 1.0)

            # Clamp and convert cosine score to a clean 0% - 100% match percentage
            clamped_score = max(0.0, min(1.0, distance))
            similarity_pct = round(clamped_score * 100, 1)

            artist = entity.get("artist", "Unknown Artist")
            song = entity.get("song", "Unknown Title")

            formatted_results.append({
                "rank": rank,
                "artist": artist,
                "song": song,
                "similarity_score": round(distance, 4),
                "match_percentage": f"{similarity_pct}%",
                "lyrics_snippet": entity.get("lyrics_snippet", ""),
                "dataset_link": entity.get("link", ""),
                "spotify_url": generate_spotify_url(artist, song),
            })

        return formatted_results


if __name__ == "__main__":
    # Quick terminal test of the search engine
    engine = SpotifySearchEngine()
    test_query = "sad kill kill bomb breakup"
    print(f"\nSearching for vibe: '{test_query}' ...\n")
    results = engine.search(test_query, top_k=5)

    for r in results:
        print(f"#{r['rank']} [{r['match_percentage']}] {r['artist']} - {r['song']}")
        print(f"   Lyrics: {r['lyrics_snippet'][:100]}...")
        print(f"   Listen: {r['spotify_url']}\n")
