from qdrant_client import QdrantClient, models
import uuid
import logging
from typing import List, Optional
import numpy as np

from app.core.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME,
    TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM,
    AUDIO_EMBEDDING_DIM, VIDEO_EMBEDDING_DIM,
    TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME, AUDIO_VECTOR_NAME, VIDEO_VECTOR_NAME
)
from app.models.encoders import encoder

logger = logging.getLogger(__name__)
# Initialize Qdrant client with host and port from config
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_collection_if_not_exists():
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        logger.info(f"Collection '{QDRANT_COLLECTION_NAME}' already exists.")
    except Exception:
        logger.info(f"Collection '{QDRANT_COLLECTION_NAME}' not found. Creating...")
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={
                TEXT_VECTOR_NAME: models.VectorParams(size=TEXT_EMBEDDING_DIM, distance=models.Distance.COSINE),
                IMAGE_VECTOR_NAME: models.VectorParams(size=IMAGE_EMBEDDING_DIM, distance=models.Distance.COSINE),
                VIDEO_VECTOR_NAME: models.VectorParams(size=VIDEO_EMBEDDING_DIM, distance=models.Distance.COSINE),
                AUDIO_VECTOR_NAME: models.VectorParams(size=AUDIO_EMBEDDING_DIM, distance=models.Distance.COSINE),
            },
        )
        logger.info("Collection created successfully.")


def upsert_chunk(doc_id: str, chunk_metadata: dict, text_chunk: str = None, image_chunk_bytes: bytes = None):
    point_id = str(uuid.uuid4())
    vectors = {}
    payload = {"doc_id": doc_id, **chunk_metadata}

    text_to_encode = text_chunk
    if isinstance(text_chunk, dict):
        text_to_encode = text_chunk.get('text')

    if text_to_encode and isinstance(text_to_encode, str) and text_to_encode.strip():
        vectors[TEXT_VECTOR_NAME] = encoder.encode_text(text_to_encode)
        payload['text'] = text_to_encode

    if image_chunk_bytes:
        image_vector = encoder.encode_image(image_chunk_bytes)
        if image_vector:
            vectors[IMAGE_VECTOR_NAME] = image_vector
            payload['type'] = 'image'

    if not vectors:
        logger.warning(f"Skipping upsert for doc_id {doc_id} as no vector was generated.")
        return

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[models.PointStruct(id=point_id, vector=vectors, payload=payload)],
        wait=False
    )
    logger.debug(f"Upserted chunk for doc_id {doc_id} with point_id {point_id}")


def upsert_video_audio_embeddings(filename: str, video_vector: list, audio_vector: list | None = None):
    point_id = str(uuid.uuid4())
    payload = {"filename": filename, "type": "video"}

    vectors = {VIDEO_VECTOR_NAME: video_vector}
    if audio_vector:
        vectors[AUDIO_VECTOR_NAME] = audio_vector

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[models.PointStruct(id=point_id, vector=vectors, payload=payload)],
        wait=False
    )
    logger.info(f"Video/audio embeddings upserted for file {filename}")


def get_points_by_ids(point_ids: List[str]) -> List[models.Record]:
    """Retrieve full point data for a list of point IDs."""
    if not point_ids:
        return []
    try:
        records, _ = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.HasIdCondition(has_id=point_ids)]
            ),
            limit=len(point_ids),
            with_vectors=True,
            with_payload=True
        )
        return records
    except Exception as e:
        logger.error(f"Failed to retrieve points by IDs: {e}")
        return []

def search_similar_content(
    vector: np.ndarray,
    vector_name: str,
    limit: int = 10,
    exclude_ids: Optional[List[str]] = None
) -> List[models.ScoredPoint]:
    """Performs a similarity search in Qdrant."""
    
    # Exclude previously interacted items from recommendations
    search_filter = None
    if exclude_ids:
        search_filter = models.Filter(
            must_not=[models.HasIdCondition(has_id=exclude_ids)]
        )

    try:
        hits = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=(vector_name, vector.tolist()),
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )
        return hits
    except Exception as e:
        logger.error(f"Error during Qdrant search for vector '{vector_name}': {e}")
        return []