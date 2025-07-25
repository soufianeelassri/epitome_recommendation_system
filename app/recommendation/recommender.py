import logging
import numpy as np
from typing import List, Dict
from collections import defaultdict

from app.db.qdrant_ops import get_points_by_ids, search_similar_content
from app.core.config import TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME, VIDEO_VECTOR_NAME, AUDIO_VECTOR_NAME

logger = logging.getLogger(__name__)

VALID_VECTOR_NAMES = {TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME, VIDEO_VECTOR_NAME, AUDIO_VECTOR_NAME}

def build_user_profile_vector(point_ids: List[str]) -> Dict[str, np.ndarray]:
    """
    Builds a user's profile by averaging the vectors of content they've interacted with.
    """
    if not point_ids:
        return {}

    points = get_points_by_ids(point_ids)
    if not points:
        return {}

    vector_aggs = defaultdict(list)
    for point in points:
        # Check if point.vector is a dictionary
        if isinstance(point.vector, dict):
            for vec_name, vector in point.vector.items():
                if vec_name in VALID_VECTOR_NAMES and vector:
                    vector_aggs[vec_name].append(np.array(vector))
        # Handle cases where point.vector might be a list (older format, for robustness)
        elif isinstance(point.vector, list):
             vector_aggs[TEXT_VECTOR_NAME].append(np.array(point.vector))


    profile_vectors = {}
    for vec_name, vectors in vector_aggs.items():
        if vectors:
            profile_vectors[vec_name] = np.mean(vectors, axis=0)
            logger.info(f"Generated profile vector for modality '{vec_name}' with shape {profile_vectors[vec_name].shape}")
            
    return profile_vectors


def get_recommendations_for_user(
    user_id: str,
    interaction_history: List[str],
    limit: int = 10
) -> List[dict]:
    """
    Generates content recommendations for a given user, ensuring each source document
    is recommended only once.
    """
    profile_vectors = build_user_profile_vector(interaction_history)

    if not profile_vectors:
        logger.warning(f"Could not generate profile vector for user '{user_id}'. No recommendations possible.")
        return []

    # This dictionary will store the best recommendation for each source document (doc_id or filename).
    recommended_docs = {}

    for vec_name, profile_vector in profile_vectors.items():
        per_modality_limit = limit * 2  # Fetch more to ensure we have enough after deduplication
        
        hits = search_similar_content(
            vector=profile_vector,
            vector_name=vec_name,
            limit=per_modality_limit,
            exclude_ids=interaction_history
        )
        
        for hit in hits:
            payload = hit.payload
            # Use doc_id for documents, and filename as a fallback for videos
            source_key = payload.get('doc_id') or payload.get('filename')
            
            if not source_key:
                continue # Skip chunks that don't have a source identifier

            # If we haven't seen this document OR this new chunk has a better score, store it.
            if source_key not in recommended_docs or hit.score > recommended_docs[source_key]['similarity_score']:
                
                # This is the new, clean response format
                response_item = {
                    'doc_id': payload.get('doc_id'),
                    'filename': payload.get('original_filename') or payload.get('filename'),
                    'type': 'video' if payload.get('type') == 'video' else 'document',
                    'similarity_score': hit.score,
                    'best_matching_chunk_payload': payload # Include the chunk payload for context
                }
                recommended_docs[source_key] = response_item

    # Sort the unique document recommendations by their highest score
    sorted_recommendations = sorted(recommended_docs.values(), key=lambda x: x['similarity_score'], reverse=True)

    logger.info(f"Generated {len(sorted_recommendations)} unique document recommendations for user '{user_id}'")
    
    return sorted_recommendations[:limit]