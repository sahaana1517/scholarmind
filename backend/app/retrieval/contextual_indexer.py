"""
Upload contextualized embeddings to a separate Qdrant collection.

Strategy:
  - Baseline collection (scholarmind_papers) untouched
  - New collection (scholarmind_papers_contextual) holds the 200 chunks
    that were embedded with context prepended
  - Payload includes BOTH the contextualized text used for embedding AND
    the original chunk text (so generation still cites the original wording)
"""

import json

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

from backend.app.core.config import settings


CONTEXTUAL_COLLECTION = "scholarmind_papers_contextual"
INPUT_PATH = settings.PAPERS_PROCESSED_DIR / "chunks_with_context_embeddings.json"


def get_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=60,
    )


def create_collection_if_not_exists(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if CONTEXTUAL_COLLECTION in existing:
        info = client.get_collection(CONTEXTUAL_COLLECTION)
        print(f"✅ Collection '{CONTEXTUAL_COLLECTION}' already exists")
        print(f"   Points: {info.points_count}")
        # Wipe it so a re-run starts clean
        print(f"🧹 Clearing existing points for re-upload...")
        client.delete_collection(CONTEXTUAL_COLLECTION)

    print(f"🆕 Creating collection: {CONTEXTUAL_COLLECTION}")
    client.create_collection(
        collection_name=CONTEXTUAL_COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=settings.EMBEDDING_DIMENSION,
            distance=qmodels.Distance.COSINE,
        ),
    )
    print(f"✅ Collection created")


def index_contextual_chunks() -> None:
    if not INPUT_PATH.exists():
        print(f"⚠ {INPUT_PATH} not found. Run contextual_embedder first.")
        return

    print(f"📂 Loading from {INPUT_PATH.name}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   Loaded {len(chunks)} contextualized embedded chunks")

    client = get_client()
    create_collection_if_not_exists(client)

    print(f"\n📤 Uploading {len(chunks)} points...")
    batch_size = 64
    for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading"):
        batch = chunks[i : i + batch_size]
        points = [
            qmodels.PointStruct(
                id=c["chunk_id"],
                vector=c["embedding"],
                payload={
                    "paper_id": c["paper_id"],
                    "page": c["page"],
                    "chunk_index_on_page": c.get("chunk_index_on_page", 0),
                    "text": c["text"],  # original chunk for display/citation
                    "context": c["context"],  # generated context (for inspection)
                    "token_count": c.get("token_count", 0),
                },
            )
            for c in batch
        ]
        client.upsert(collection_name=CONTEXTUAL_COLLECTION, points=points, wait=True)

    final_info = client.get_collection(CONTEXTUAL_COLLECTION)
    print(f"\n{'='*60}")
    print(f"📊 Contextual collection status:")
    print(f"   Name:   {CONTEXTUAL_COLLECTION}")
    print(f"   Points: {final_info.points_count}")
    print(f"   Status: {final_info.status}")


if __name__ == "__main__":
    index_contextual_chunks()