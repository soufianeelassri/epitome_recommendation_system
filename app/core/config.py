import os

# Qdrant vector database host
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
# Qdrant vector database port
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
# Name of the Qdrant collection used for storing content vectors
QDRANT_COLLECTION_NAME = "epitome_academy_content"

# Model name for text embeddings
TEXT_EMBEDDING_MODEL = 'app/embedding_service/sentence_transformer/'
# Dimensionality of the text embedding vectors
TEXT_EMBEDDING_DIM = 384

# Model name for image embeddings
IMAGE_EMBEDDING_MODEL = 'app/embedding_service/clip/'
# Dimensionality of the image embedding vectors
IMAGE_EMBEDDING_DIM = 512

from pathlib import Path

TEMP_FILES_DIR = Path("temp_files")
# Clés utilisées dans la collection pour différencier les types de vecteurs
VIDEO_VECTOR_NAME = "video"
AUDIO_VECTOR_NAME = "audio"

# Dimensions spécifiques pour chaque vecteur dans la même collection
VIDEO_EMBEDDING_DIM = 512     # CLIP
AUDIO_EMBEDDING_MODEL = 'app/embedding_service/wav2vec2/'
AUDIO_EMBEDDING_DIM = 768     # Wav2Vec2