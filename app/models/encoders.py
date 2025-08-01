from sentence_transformers import SentenceTransformer
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import io
import base64
import logging
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
from app.core.config import AUDIO_EMBEDDING_MODEL, VIDEO_EMBEDDING_DIM, AUDIO_EMBEDDING_DIM


logger = logging.getLogger(__name__)

class MultimodalEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"MultimodalEncoder initialized, will use device: {self.device}")
        
        self.text_model = None
        self.image_model = None
        self.image_preprocess = None
        self.audio_model = None
        self.audio_processor = None
        self.video_model = None
        self.video_processor = None

    def _load_text_model(self):
        if self.text_model is None:
            logger.info("Loading text embedding model...")
            try:
                self.text_model = SentenceTransformer("app/embedding_service/sentence_transformer/", device=self.device)
                logger.info("Text embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading text model: {e}")
                raise

    def _load_image_model(self):
        if self.image_model is None:
            logger.info("Loading image embedding model...")
            try:
                self.image_model = CLIPModel.from_pretrained("app/embedding_service/clip/").to(self.device)
                self.image_preprocess = CLIPProcessor.from_pretrained("app/embedding_service/clip/")
                logger.info("Image embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading image model: {e}")
                raise
    def _load_audio_model(self):
        if self.audio_model is None or self.audio_processor is None:
            logger.info("Loading audio embedding model...")
            try:
                self.audio_model = Wav2Vec2Model.from_pretrained(AUDIO_EMBEDDING_MODEL).to(self.device)
                self.audio_processor = Wav2Vec2Processor.from_pretrained(AUDIO_EMBEDDING_MODEL)
                logger.info("Audio embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading audio model: {e}")
                raise

    def encode_text(self, text):
        if not text or not isinstance(text, str):
            return []
        
        self._load_text_model()
        
        embedding = self.text_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def encode_image(self, image_data):
        try:
            self._load_image_model()
            
            if isinstance(image_data, str):
                if image_data.startswith('data:'):
                    image_data = image_data.split(',', 1)[1]
                
                try:
                    image_bytes = base64.b64decode(image_data)
                except Exception as e:
                    logger.error(f"Error decoding base64 image data: {e}")
                    return None
                
            elif isinstance(image_data, bytes):
                image_bytes = image_data
                
            else:
                logger.error(f"Unsupported image data type: {type(image_data)}")
                return None
            
            if not image_bytes:
                logger.error("Empty image data")
                return None
            
            try:
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                logger.error(f"Error opening image from bytes: {e}")
                return None
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            try:
                processed = self.image_preprocess(images=image, return_tensors="pt")
                image_input = processed["pixel_values"].to(self.device)
            except Exception as e:
                logger.error(f"Error preprocessing image: {e}")
                return None
            
            with torch.no_grad():
                try:
                    image_features = self.image_model.get_image_features(pixel_values=image_input)
                except Exception as e:
                    logger.error(f"Error encoding image with CLIP: {e}")
                    return None
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().squeeze().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Unexpected error encoding image: {e}", exc_info=True)
            return None

    def encode_audio(self, audio_path: str):
        self._load_audio_model()

        try:
            wav, sr = sf.read(audio_path)
        except Exception as e:
            logger.error(f"Error reading audio file: {e}")
            return None

        try:
            inputs = self.audio_processor(wav, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.audio_model(inputs["input_values"]).last_hidden_state
                embedding = features.mean(dim=1).squeeze(0).cpu().numpy().tolist()
                return embedding
        except Exception as e:
            logger.error(f"Error generating audio embedding: {e}")
            return None

    def encode_video_from_frames(self, frame_paths: list[str]):
        self._load_image_model()

        try:
            images = [Image.open(p).convert("RGB") for p in frame_paths[:16]]
            inputs = self.image_preprocess(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.image_model.get_image_features(**inputs)
                video_embedding = image_features.mean(dim=0).cpu().numpy().tolist()
                return video_embedding
        except Exception as e:
            logger.error(f"Error generating video embedding from frames: {e}")
            return None

encoder = MultimodalEncoder()