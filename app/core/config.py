import os
from pathlib import Path

# Qdrant vector database host
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
# Qdrant vector database port
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
# Name of the Qdrant collection used for storing content vectors
QDRANT_COLLECTION_NAME = "epitome_academy_content"

# --- Vector Configuration ---
# Keys used in the collection to differentiate vector types
TEXT_VECTOR_NAME = "text"
IMAGE_VECTOR_NAME = "image"
VIDEO_VECTOR_NAME = "video"
AUDIO_VECTOR_NAME = "audio"

# --- Model & Dimension Configuration ---
# Text Model
TEXT_EMBEDDING_MODEL = 'app/embedding_service/sentence_transformer/'
TEXT_EMBEDDING_DIM = 384

# Image Model (CLIP)
IMAGE_EMBEDDING_MODEL = 'app/embedding_service/clip/'
IMAGE_EMBEDDING_DIM = 512

# Video Model (uses Image Model)
VIDEO_EMBEDDING_DIM = 512

# Audio Model (Wav2Vec2)
AUDIO_EMBEDDING_MODEL = 'app/embedding_service/wav2vec2/'
AUDIO_EMBEDDING_DIM = 768

# --- File System ---
TEMP_FILES_DIR = Path("temp_files")