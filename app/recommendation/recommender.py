import logging
import numpy as np
from collections import defaultdict
import uuid
from typing import List, Dict, Any

from app.db import qdrant_ops
from app.core.config import TEXT_VECTOR_NAME, VIDEO_VECTOR_NAME, AUDIO_VECTOR_NAME
from app.models.encoders import encoder
from app.recommendation import user_service
from qdrant_client import models

logger = logging.getLogger(__name__)

VALID_VECTOR_NAMES = {TEXT_VECTOR_NAME, VIDEO_VECTOR_NAME, AUDIO_VECTOR_NAME}

def build_user_profile_vector(point_ids: List[str]) -> Dict[str, np.ndarray]:
    """Builds a user's profile by averaging the vectors of content they've interacted with."""
    if not point_ids:
        return {}

    points = qdrant_ops.get_points_by_ids(point_ids)
    if not points:
        return {}

    vector_aggs = defaultdict(list)
    for point in points:
        if isinstance(point.vector, dict):
            for vec_name, vector in point.vector.items():
                if vec_name in VALID_VECTOR_NAMES and vector:
                    vector_aggs[vec_name].append(np.array(vector))

    profile_vectors = {}
    for vec_name, vectors in vector_aggs.items():
        if vectors:
            profile_vectors[vec_name] = np.mean(vectors, axis=0)
            logger.info(f"Generated profile vector for '{vec_name}'")
            
    return profile_vectors

def get_recommendations_for_user(user_id: str, interaction_history: List[str], limit: int) -> List[Dict[str, Any]]:
    """Generates hybrid recommendations using a filter-first or vector-search strategy."""
    
    user_prefs = user_service.get_user_preferences(user_id)
    recommended_docs = {}
    
    if not interaction_history:
        logger.info(f"Cold start for user '{user_id}'. Using filter-first strategy.")
        
        filter_conditions = []
        if user_prefs.areas_of_interest:
            filter_conditions.append(models.FieldCondition(
                key="category",
                match=models.MatchAny(any=user_prefs.areas_of_interest)
            ))
        if user_prefs.preferred_content_types:
            types_to_match = []
            if "Video" in user_prefs.preferred_content_types:
                types_to_match.extend(['video_summary', 'video_chunk'])
            if "Document" in user_prefs.preferred_content_types:
                 types_to_match.extend(['document', 'metadata'])
            
            if types_to_match:
                filter_conditions.append(models.FieldCondition(
                    key="content_type",
                    match=models.MatchAny(any=types_to_match)
                ))

        if not filter_conditions:
            logger.warning(f"User '{user_id}' has no preferences to filter by. Cannot recommend.")
            return []

        candidate_points, _ = qdrant_ops.qdrant_client.scroll(
            collection_name=qdrant_ops.QDRANT_COLLECTION_NAME,
            scroll_filter=models.Filter(should=filter_conditions),
            limit=200,
            with_payload=True
        )
        
        for point in candidate_points:
            payload = point.payload
            score = 1.0
            if user_prefs.preferred_content_types and any(t in payload.get('content_type', '') for t in types_to_match):
                score += 0.5
            if user_prefs.areas_of_interest and payload.get('category') in user_prefs.areas_of_interest:
                score += 0.5

            source_key = payload.get('doc_id')
            if source_key and (source_key not in recommended_docs or score > recommended_docs[source_key]['similarity_score']):
                 recommended_docs[source_key] = {
                    'doc_id': source_key,
                    'title': payload.get('title', 'Title not available'),
                    'filename': payload.get('original_filename'),
                    'type': 'video' if 'video' in payload.get('content_type', '') else 'document',
                    'similarity_score': score,
                    'best_matching_chunk_payload': payload,
                    'start_time': payload.get('start_time'),
                    'end_time': payload.get('end_time'),
                }

    else:
        logger.info(f"Warm start for user '{user_id}'. Using interaction history.")
        profile_vectors = build_user_profile_vector(interaction_history)
        if not profile_vectors: return []

        all_hits = []
        for vec_name, profile_vector in profile_vectors.items():
            all_hits.extend(qdrant_ops.search_similar_content(
                vector=profile_vector,
                vector_name=vec_name,
                limit=limit * 2,
                exclude_ids=interaction_history
            ))

        for hit in all_hits:
            payload = hit.payload
            source_key = payload.get('doc_id')
            if not source_key: continue

            boost = 0.0
            if user_prefs.preferred_content_types and "Video" in user_prefs.preferred_content_types and 'video' in payload.get('content_type', ''):
                boost += 0.15
            if user_prefs.areas_of_interest and payload.get('category') in user_prefs.areas_of_interest:
                 boost += 0.1
            
            final_score = hit.score + boost

            if source_key not in recommended_docs or final_score > recommended_docs[source_key]['similarity_score']:
                recommended_docs[source_key] = {
                    'doc_id': source_key,
                    'title': payload.get('title', 'Title not available'),
                    'filename': payload.get('original_filename'),
                    'type': 'video' if 'video' in payload.get('content_type', '') else 'document',
                    'similarity_score': final_score,
                    'best_matching_chunk_payload': payload,
                    'start_time': payload.get('start_time'),
                    'end_time': payload.get('end_time'),
                }

    sorted_recommendations = sorted(recommended_docs.values(), key=lambda x: x['similarity_score'], reverse=True)
    logger.info(f"Generated {len(sorted_recommendations)} unique recommendations for user '{user_id}'")
    
    return sorted_recommendations[:limit]

def get_recommendations_for_keywords(keywords: List[str], per_keyword_limit: int, final_limit: int) -> List[Dict[str, Any]]:
    """Generates content recommendations based on a list of keywords."""
    if not keywords:
        logger.warning("No keywords provided for recommendation search.")
        return []

    all_recommendations = []
    temporary_point_ids = []
    
    try:
        for keyword in keywords:
            logger.info(f"Searching for content similar to keyword: '{keyword}'")
            
            keyword_vector = encoder.encode_text(keyword)
            if keyword_vector is None:
                logger.warning(f"Could not encode keyword '{keyword}', skipping.")
                continue
            
            temp_point_id = str(uuid.uuid4())
            temp_payload = {"type": "temporary_keyword", "keyword": keyword, "text": keyword}
            
            # --- CORRECTED: Added qdrant_ops. prefix ---
            success = qdrant_ops.insert_temporary_point(
                point_id=temp_point_id,
                vector=np.array(keyword_vector),
                vector_name=TEXT_VECTOR_NAME,
                payload=temp_payload
            )
            
            if not success:
                logger.error(f"Failed to insert temporary point for keyword '{keyword}', skipping.")
                continue
            
            temporary_point_ids.append(temp_point_id)
            all_hits = []
            
            for vector_name in [TEXT_VECTOR_NAME, VIDEO_VECTOR_NAME, AUDIO_VECTOR_NAME]:
                # --- CORRECTED: Added qdrant_ops. prefix ---
                hits = qdrant_ops.search_similar_to_point(
                    point_id=temp_point_id,
                    vector_name=vector_name,
                    limit=per_keyword_limit,
                    exclude_ids=temporary_point_ids
                )
                all_hits.extend(hits)
            
            hits = sorted(all_hits, key=lambda x: x.score, reverse=True)
            
            # This logic remains the same
            for hit in hits:
                payload = hit.payload
                source_key = payload.get('doc_id') or payload.get('filename')
                if not source_key: continue 

                if source_key not in all_recommendations or hit.score > next((r['similarity_score'] for r in all_recommendations if r.get('doc_id') == source_key), -1):
                    all_recommendations = [r for r in all_recommendations if r.get('doc_id') != source_key]
                    response_item = {
                        'doc_id': payload.get('doc_id'),
                        'filename': payload.get('original_filename') or payload.get('filename'),
                        'type': 'video' if 'video' in payload.get('type', '') else 'document',
                        'similarity_score': hit.score,
                        'keyword': keyword,
                        'best_matching_chunk_payload': payload
                    }
                    all_recommendations.append(response_item)
    finally:
        for temp_point_id in temporary_point_ids:
            try:
                qdrant_ops.delete_point(temp_point_id)
                logger.debug(f"Cleaned up temporary point: {temp_point_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary point {temp_point_id}: {e}")

    sorted_recommendations = sorted(all_recommendations, key=lambda x: x['similarity_score'], reverse=True)
    return sorted_recommendations[:final_limit]