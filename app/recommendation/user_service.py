import logging
from collections import defaultdict
from typing import List, Dict

logger = logging.getLogger(__name__)

# In-memory data store (for MVP purposes)
# Structure: {"user_id_1": {"point_id_1", "point_id_2"}, "user_id_2": {"point_id_3"}}
user_interactions: Dict[str, set] = defaultdict(set)

def record_interaction(user_id: str, point_id: str) -> bool:
    """Records a user's interaction with a content point."""
    logger.info(f"Recording interaction for user '{user_id}' with point '{point_id}'")
    user_interactions[user_id].add(point_id)
    return True

def get_user_interactions(user_id: str) -> List[str]:
    """Retrieves all content point IDs a user has interacted with."""
    interactions = list(user_interactions.get(user_id, []))
    logger.info(f"Retrieved {len(interactions)} interactions for user '{user_id}'")
    return interactions

def get_user_profile(user_id: str) -> dict:
    """Constructs a user profile response."""
    history = get_user_interactions(user_id)
    return {
        "user_id": user_id,
        "interaction_history": history,
        "learning_path": [],  # Placeholder for future development
        "preferences": {},    # Placeholder for future development
    }