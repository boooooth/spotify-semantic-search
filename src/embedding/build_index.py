import argparse
import time
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType

from embedding.config import (
    DATASET_PATH,
    MILVUS_DB_PATH,
    COLLECTION_NAME,
    MODEL_NAME,
    EMBEDDING_DIM,
    METRIC_TYPE,
    INDEX_TYPE,
    BATCH_SIZE,
    MAX_ARTIST_LENGTH,
    MAX_SONG_LENGTH,
    MAX_LINK_LENGTH,
    MAX_SNIPPET_LENGTH,
)
from embedding.preprocess import prepare_dataset


def init_milvus_collection(client: MilvusClient, drop_existing: bool = True):

    # Initializes the Milvus collection schema and index parameters.
    # If collection already exists, drop it so we start fresh
    if client.has_collection(collection_name=COLLECTION_NAME):
        if drop_existing:
            print(f"Dropping existing collection: '{COLLECTION_NAME}'...")
            client.drop_collection(collection_name=COLLECTION_NAME)
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
            return

    print(f"Creating collection '{COLLECTION_NAME}' with {INDEX_TYPE} index...")

    # 1. Define the schema (columns)
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field(field_name="artist", datatype=DataType.VARCHAR, max_length=MAX_ARTIST_LENGTH)
    schema.add_field(field_name="song", datatype=DataType.VARCHAR, max_length=MAX_SONG_LENGTH)
    schema.add_field(field_name="link", datatype=DataType.VARCHAR, max_length=MAX_LINK_LENGTH)
    schema.add_field(field_name="lyrics_snippet", datatype=DataType.VARCHAR, max_length=MAX_SNIPPET_LENGTH)

    # 2. Define the index (FLAT + COSINE)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type=METRIC_TYPE,
        index_type=INDEX_TYPE,
    )

    # 3. Create the collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully.")


def build_index(sample_size: int | None = None, batch_size: int = BATCH_SIZE):

    # data loading, embedding generation, and Milvus insertion.
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please check data folder.")

    print(f"Loading dataset from: {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)
    total_tracks = len(df)
    print(f"Total tracks in CSV: {total_tracks:,}")

    # Optional: sample mode for rapid testing
    if sample_size and sample_size < total_tracks:
        print(f"Running in sample mode: indexing first {sample_size:,} tracks...")
        df = df.head(sample_size).copy()
    else:
        print(f"Indexing all {total_tracks:,} tracks...")

    # Clean and build search_text + lyrics_snippet
    print("Preprocessing lyrics...")
    df = prepare_dataset(df)

    # Load SentenceTransformers model
    print(f"Loading embedding model: '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)

    # Connect to Milvus Lite (local file)
    print(f"Connecting to Milvus Lite at: {MILVUS_DB_PATH} ...")
    client = MilvusClient(uri=str(MILVUS_DB_PATH))
    init_milvus_collection(client, drop_existing=True)

    # Batch embedding and ingestion
    print(f"Generating embeddings and inserting in batches (batch_size={batch_size}) ...")
    start_time = time.time()
    num_records = len(df)

    for i in tqdm(range(0, num_records, batch_size), desc="Embedding & Ingesting"):
        batch_df = df.iloc[i : i + batch_size]
        batch_texts = batch_df["search_text"].tolist()

        # Generate L2-normalized 384-d vectors
        embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Prepare records defensively sliced to avoid any schema overflow
        records = []
        for emb, (_, row) in zip(embeddings, batch_df.iterrows()):
            records.append({
                "vector": emb.tolist(),
                "artist": str(row["artist"])[:MAX_ARTIST_LENGTH],
                "song": str(row["song"])[:MAX_SONG_LENGTH],
                "link": str(row["link"])[:MAX_LINK_LENGTH],
                "lyrics_snippet": str(row["lyrics_snippet"])[:MAX_SNIPPET_LENGTH],
            })

        # Insert batch into Milvus
        client.insert(collection_name=COLLECTION_NAME, data=records)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)!")

    # Verify indexed rows
    stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
    print(f"Milvus Collection row count: {stats.get('row_count', num_records):,}")
    print(f"Database saved to: {MILVUS_DB_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Build Milvus vector index for Spotify Semantic Search")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of songs to index (e.g. 1000 for quick test). Omit for full dataset.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process the entire ~57k dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE}).",
    )
    args = parser.parse_args()

    sample = None if args.full else args.sample_size
    build_index(sample_size=sample, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
