import logging
from collections import defaultdict
from typing import Dict
from app.models.schemas import UserPreferences

logger = logging.getLogger(__name__)

user_interactions: Dict[str, set] = defaultdict(set)
user_preferences: Dict[str, UserPreferences] = {} 

def record_interaction(user_id, point_id):
    logger.info(f"Recording interaction for user '{user_id}' with point '{point_id}'")
    user_interactions[user_id].add(point_id)
    return True

def get_user_interactions(user_id):
    interactions = list(user_interactions.get(user_id, []))
    logger.info(f"Retrieved {len(interactions)} interactions for user '{user_id}'")
    return interactions

def save_user_preferences(user_id, preferences: UserPreferences):
    logger.info(f"Saving preferences for user '{user_id}'")
    user_preferences[user_id] = preferences
    return True

def get_user_preferences(user_id):
    default_prefs = UserPreferences(domaines_interets=[], types_contenus=[], objectifs_apprentissage=[])
    prefs = user_preferences.get(user_id, default_prefs)
    logger.info(f"Retrieved preferences for user '{user_id}'")
    return prefs

def get_user_profile(user_id):
    history = get_user_interactions(user_id)
    prefs = get_user_preferences(user_id)
    return {
        "user_id": user_id,
        "preferences": prefs,
        "interaction_history": history,
    }