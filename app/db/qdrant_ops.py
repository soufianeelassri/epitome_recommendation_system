from qdrant_client import QdrantClient, models
import uuid
import logging
from app.core.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME,
    TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM,
    AUDIO_EMBEDDING_DIM, VIDEO_EMBEDDING_DIM,
    AUDIO_VECTOR_NAME, VIDEO_VECTOR_NAME
)

from app.models.encoders import encoder

logger = logging.getLogger(__name__)
# Initialize Qdrant client with host and port from config
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_collection_if_not_exists():
    try:
        # Try to get the collection
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        logger.info(f"Collection '{QDRANT_COLLECTION_NAME}' already exists.")
    except Exception:
        # If collection does not exist, create it with text and image vector configs
        logger.info(f"Collection '{QDRANT_COLLECTION_NAME}' not found. Creating...")
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={
                "text": models.VectorParams(size=TEXT_EMBEDDING_DIM, distance=models.Distance.COSINE),
                "image": models.VectorParams(size=IMAGE_EMBEDDING_DIM, distance=models.Distance.COSINE),
                VIDEO_VECTOR_NAME: models.VectorParams(size=VIDEO_EMBEDDING_DIM, distance=models.Distance.COSINE),
                AUDIO_VECTOR_NAME: models.VectorParams(size=AUDIO_EMBEDDING_DIM, distance=models.Distance.COSINE),
            },
        )

        logger.info("Collection created successfully.")


def upsert_chunk(doc_id: str, chunk_metadata: dict, text_chunk: str = None, image_chunk_bytes: bytes = None):
    point_id = str(uuid.uuid4())  # Unique identifier for the Qdrant point
    vectors = {}  # Dictionary to hold vector data for text/image
    payload = {"doc_id": doc_id, **chunk_metadata}  # Metadata payload for Qdrant

    # Handle text_chunk being either a string or a dict (e.g., from table elements)
    text = None
    if isinstance(text_chunk, str):
        text = text_chunk
    elif isinstance(text_chunk, dict):
        text = text_chunk.get('text', None)

    # If a non-empty text chunk is provided, encode and add to vectors/payload
    if text and isinstance(text, str) and text.strip():
        vectors['text'] = encoder.encode_text(text)
        payload['text'] = text
    
    # If image bytes are provided, encode and add to vectors/payload
    if image_chunk_bytes:
        image_vector = encoder.encode_image(image_chunk_bytes)
        if image_vector:
            vectors['image'] = image_vector
            payload['type'] = 'image'  # Mark payload as image type

    if not vectors:
        logger.warning(f"Skipping upsert for doc_id {doc_id} as no vector was generated.")
        return

    try:
        # Upsert the point into Qdrant collection
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload=payload,
                )
            ],
            wait=False  # Do not wait for operation to complete
        )
        logger.debug(f"Upserted chunk for doc_id {doc_id} with point_id {point_id}")
    except Exception as e:
        logger.error(f"Failed to upsert chunk for doc_id {doc_id}: {e}")


def find_similar_images(image_bytes, top_k=1):
    """
    Given image bytes, encode and search for similar images in Qdrant.
    Returns a list of Qdrant search results (payloads and scores).
    """
    image_vector = encoder.encode_image(image_bytes)
    if image_vector is None:
        return []

    results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector={"image": image_vector},
        limit=top_k,
        with_payload=True,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value="image")
                )
            ]
        )
    )
    return results
def upsert_video_audio_embeddings(filename: str, video_vector: list, audio_vector: list | None = None):
    point_id = str(uuid.uuid4())
    payload = {"filename": filename, "type": "video"}

    vectors = {VIDEO_VECTOR_NAME: video_vector}
    if audio_vector:
        vectors[AUDIO_VECTOR_NAME] = audio_vector

    try:
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload=payload,
                )
            ],
            wait=False
        )
        logger.info(f"Video/audio embeddings upserted for file {filename}")
    except Exception as e:
        logger.error(f"Failed to upsert video/audio embeddings for file {filename}: {e}")
