from qdrant_client import QdrantClient, models
import uuid
import logging

from app.core.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME,
    TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM,
    AUDIO_EMBEDDING_DIM, VIDEO_EMBEDDING_DIM,
    TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME, AUDIO_VECTOR_NAME, VIDEO_VECTOR_NAME
)
from app.models.encoders import encoder

logger = logging.getLogger(__name__)
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


def upsert_chunk(doc_id, chunk_metadata, text_chunk = None, image_chunk_bytes = None):
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


def upsert_video_audio_embeddings(doc_id, metadata, video_vector, audio_vector = None):
    point_id = str(uuid.uuid4())
    payload = {"doc_id": doc_id, **metadata}

    vectors = {VIDEO_VECTOR_NAME: video_vector}
    if audio_vector:
        vectors[AUDIO_VECTOR_NAME] = audio_vector

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[models.PointStruct(id=point_id, vector=vectors, payload=payload)],
        wait=False
    )
    logger.info(f"Video/audio embeddings upserted for doc_id {doc_id}")


def get_points_by_ids(point_ids):
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

def search_similar_content(vector, vector_name, limit = 10, exclude_ids = None):
    search_filter = None
    if exclude_ids:
        search_filter = models.Filter(
            must_not=[models.HasIdCondition(has_id=exclude_ids)]
        )

    try:
        vector_list = vector.tolist() if hasattr(vector, 'tolist') else vector
            
        hits = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=(vector_name, vector_list),
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )
        return hits
    except Exception as e:
        logger.error(f"Error during Qdrant search for vector '{vector_name}': {e}")
        return []


def insert_temporary_point(point_id, vector, vector_name, payload):
    try:
        if hasattr(vector, 'tolist'):
            vector_list = vector.tolist()
        else:
            vector_list = vector
            
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[models.PointStruct(id=point_id, vector={vector_name: vector_list}, payload=payload)],
            wait=True 
        )
        logger.debug(f"Temporary point {point_id} inserted for keyword search")
        return True
    except Exception as e:
        logger.error(f"Failed to insert temporary point {point_id}: {e}")
        return False


def delete_point(point_id):
    try:
        qdrant_client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=models.PointIdsList(
                points=[point_id]
            )
        )
        logger.debug(f"Point {point_id} deleted from Qdrant")
        return True
    except Exception as e:
        logger.error(f"Failed to delete point {point_id}: {e}")
        return False

def search_similar_to_point(point_id, vector_name, limit = 10, exclude_ids = None):
    exclude_list = [point_id]
    if exclude_ids:
        exclude_list.extend(exclude_ids)
    
    search_filter = models.Filter(
        must_not=[models.HasIdCondition(has_id=exclude_list)]
    )

    try:
        points = qdrant_client.retrieve(
            collection_name=QDRANT_COLLECTION_NAME,
            ids=[point_id],
            with_vectors=True
        )
        
        if not points:
            logger.error(f"Point {point_id} not found in Qdrant")
            return []
        
        point = points[0]
        if vector_name not in point.vector:
            logger.error(f"Vector '{vector_name}' not found in point {point_id}")
            return []
        
        vector = point.vector[vector_name]
        
        hits = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=(vector_name, vector),
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )
        return hits
    except Exception as e:
        logger.error(f"Error during Qdrant search for point '{point_id}': {e}")
        return []