import uuid
from collections import defaultdict
from typing import Dict, List
from langchain_core.messages import AIMessage, HumanMessage

conversation_histories: Dict[str, List] = defaultdict(list)

def get_or_create_conversation_id(conversation_id: str = None) -> str:
    return conversation_id if conversation_id else str(uuid.uuid4())

def get_history(conversation_id: str) -> List:
    return conversation_histories.get(conversation_id, [])

def update_history(conversation_id: str, question: str, answer: str):
    conversation_histories[conversation_id].extend([
        HumanMessage(content=question),
        AIMessage(content=answer)
    ])