"""
Uploads embedded chunks to Qdrant Cloud vector database.

Creates a collection if it doesn't exist, then uploads chunks in batches.
Each point in Qdrant has:
  - id: UUID (the chunk_id)
  - vector: 384-dim embedding
  - payload: full chunk metadata (paper_id, page, text, etc.)
"""

import json
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

from backend.app.core.config import settings


def get_client() -> QdrantClient:
    """Create authenticated Qdrant client."""
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )


def create_collection_if_not_exists(client: QdrantClient) -> None:
    """Create the scholarmind_papers collection with proper config."""
    existing = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION_NAME in existing:
        info = client.get_collection(settings.QDRANT_COLLECTION_NAME)
        print(f"✅ Collection '{settings.QDRANT_COLLECTION_NAME}' already exists")
        print(f"   Points: {info.points_count}")
        return

    print(f"🆕 Creating collection: {settings.QDRANT_COLLECTION_NAME}")
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=settings.EMBEDDING_DIMENSION,
            distance=qmodels.Distance.COSINE,
        ),
    )
    print(f"✅ Collection created")


def upload_chunks(client: QdrantClient, chunks: List[dict], batch_size: int = 64) -> None:
    """Upload chunks to Qdrant in batches."""
    total = len(chunks)
    print(f"\n📤 Uploading {total} chunks in batches of {batch_size}...")

    uploaded = 0
    for i in tqdm(range(0, total, batch_size), desc="Uploading batches"):
        batch = chunks[i : i + batch_size]

        points = [
            qmodels.PointStruct(
                id=chunk["chunk_id"],  # UUID string
                vector=chunk["embedding"],
                payload={
                    "paper_id": chunk["paper_id"],
                    "page": chunk["page"],
                    "chunk_index_on_page": chunk["chunk_index_on_page"],
                    "text": chunk["text"],
                    "token_count": chunk["token_count"],
                },
            )
            for chunk in batch
        ]

        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points,
            wait=True,  # ensures data is persisted before returning
        )
        uploaded += len(batch)

    print(f"✅ Uploaded {uploaded} chunks")


def index_all_chunks() -> None:
    """End-to-end: load embedded chunks, create collection, upload."""
    chunks_path = settings.PAPERS_PROCESSED_DIR / "chunks_with_embeddings.json"

    if not chunks_path.exists():
        print(f"⚠ Embeddings file not found: {chunks_path}")
        print(f"   Run embedder first: python -m backend.app.ingestion.embedder")
        return

    print(f"📂 Loading embedded chunks from {chunks_path.name}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   Loaded {len(chunks)} chunks")

    client = get_client()
    create_collection_if_not_exists(client)
    upload_chunks(client, chunks)

    # Final verification
    final_info = client.get_collection(settings.QDRANT_COLLECTION_NAME)
    print(f"\n{'='*60}")
    print(f"📊 Qdrant collection status:")
    print(f"   Name:    {settings.QDRANT_COLLECTION_NAME}")
    print(f"   Points:  {final_info.points_count}")
    print(f"   Status:  {final_info.status}")


if __name__ == "__main__":
    index_all_chunks()