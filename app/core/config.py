import os
from pathlib import Path

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = "epitome_academy_content"

TEXT_VECTOR_NAME = "text"
IMAGE_VECTOR_NAME = "image"
VIDEO_VECTOR_NAME = "video"
AUDIO_VECTOR_NAME = "audio"

TEXT_EMBEDDING_MODEL = 'app/embedding_service/sentence_transformer/'
TEXT_EMBEDDING_DIM = 384

IMAGE_EMBEDDING_MODEL = 'app/embedding_service/clip/'
IMAGE_EMBEDDING_DIM = 512

VIDEO_EMBEDDING_DIM = 512

AUDIO_EMBEDDING_MODEL = 'app/embedding_service/wav2vec2/'
AUDIO_EMBEDDING_DIM = 768

TEMP_FILES_DIR = Path("temp_files")