import logging
import numpy as np
from typing import List, Dict
from collections import defaultdict
import uuid

from app.db.qdrant_ops import get_points_by_ids, search_similar_content, insert_temporary_point, delete_point, search_similar_to_point
from app.core.config import TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME, VIDEO_VECTOR_NAME, AUDIO_VECTOR_NAME
from app.models.encoders import encoder

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


def get_recommendations_for_keywords(
    keywords: List[str],
    per_keyword_limit: int = 5,
    final_limit: int = 5
) -> List[dict]:
    """
    Generates content recommendations for a list of keywords.
    For each keyword, temporarily stores it in Qdrant, searches for similar content,
    then removes the temporary point.
    """
    if not keywords:
        logger.warning("No keywords provided for recommendation search.")
        return []

    all_recommendations = []
    temporary_point_ids = []
    
    try:
        for keyword in keywords:
            logger.info(f"Searching for content similar to keyword: '{keyword}'")
            
            # Encode the keyword to get its vector representation
            keyword_vector = encoder.encode_text(keyword)
            
            if keyword_vector is None:
                logger.warning(f"Could not encode keyword '{keyword}', skipping.")
                continue
            
            # Create a temporary point ID using UUID
            temp_point_id = str(uuid.uuid4())
            
            # Insert the keyword as a temporary point in Qdrant
            temp_payload = {
                "type": "temporary_keyword",
                "keyword": keyword,
                "text": keyword
            }
            
            success = insert_temporary_point(
                point_id=temp_point_id,
                vector=keyword_vector,
                vector_name=TEXT_VECTOR_NAME,
                payload=temp_payload
            )
            
            if not success:
                logger.error(f"Failed to insert temporary point for keyword '{keyword}', skipping.")
                continue
            
            # Add to the list of temporary points only if insertion was successful
            temporary_point_ids.append(temp_point_id)
            
            # Search for similar content using the temporary point
            hits = search_similar_to_point(
                point_id=temp_point_id,
                vector_name=TEXT_VECTOR_NAME,
                limit=per_keyword_limit * 2,  # Get more to ensure we have enough after deduplication
                exclude_ids=temporary_point_ids  # Exclude other temporary points
            )
            
            # Process hits for this keyword
            keyword_recommendations = {}
            for hit in hits:
                payload = hit.payload
                # Use doc_id for documents, and filename as a fallback for videos
                source_key = payload.get('doc_id') or payload.get('filename')
                
                if not source_key:
                    continue  # Skip chunks that don't have a source identifier

                # If we haven't seen this document OR this new chunk has a better score, store it.
                if source_key not in keyword_recommendations or hit.score > keyword_recommendations[source_key]['similarity_score']:
                    response_item = {
                        'doc_id': payload.get('doc_id'),
                        'filename': payload.get('original_filename') or payload.get('filename'),
                        'type': 'video' if payload.get('type') == 'video' else 'document',
                        'similarity_score': hit.score,
                        'keyword': keyword,
                        'best_matching_chunk_payload': payload
                    }
                    keyword_recommendations[source_key] = response_item
            
            # Add the top recommendations for this keyword
            sorted_keyword_recs = sorted(keyword_recommendations.values(), key=lambda x: x['similarity_score'], reverse=True)
            all_recommendations.extend(sorted_keyword_recs[:per_keyword_limit])
            
            logger.info(f"Found {len(sorted_keyword_recs[:per_keyword_limit])} recommendations for keyword '{keyword}'")
    
    finally:
        # Clean up: delete all temporary points
        for temp_point_id in temporary_point_ids:
            try:
                delete_point(temp_point_id)
                logger.debug(f"Cleaned up temporary point: {temp_point_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary point {temp_point_id}: {e}")

    # Sort all recommendations by similarity score and return the top results
    final_recommendations = sorted(all_recommendations, key=lambda x: x['similarity_score'], reverse=True)
    
    logger.info(f"Generated {len(final_recommendations)} total recommendations from {len(keywords)} keywords")
    
    return final_recommendations[:final_limit]