import os
import shutil
import uuid
import logging
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Body
from prometheus_fastapi_instrumentator import Instrumentator
from pathlib import Path
from typing import List, Optional

from app.processing.video_processor import extract_frames, extract_audio
from app.models.encoders import encoder
from app.db.qdrant_ops import upsert_video_audio_embeddings
from app.core.config import TEMP_FILES_DIR
from app.db.qdrant_ops import create_collection_if_not_exists, qdrant_client, upsert_chunk
from app.processing.document_processor import process_pdf
from app.recommendation import user_service, recommender
from app.models import schemas

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Epitome Academy Recommendation System API",
    description="API for content processing and recommendation at Epitome Academy.",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)

@app.on_event("startup")
def startup_event():
    logger.info("Application startup...")
    create_collection_if_not_exists()
    logger.info("Application startup complete.")

def process_and_embed_pdf(temp_path, doc_id, original_filename, course_metadata):
    logger.info(f"BG Task: Starting PDF processing for doc_id: {doc_id}")
    try:
        extracted_elements = process_pdf(temp_path)
        for element in extracted_elements:
            final_metadata = {
                **course_metadata, 
                **element['metadata'], 
                'original_filename': original_filename,
                'content_type': 'document'
            }
            
            content = element['content']
            element_type = element['type']
            
            if element_type in ['text', 'table']:
                upsert_chunk(doc_id=doc_id, text_chunk=content, chunk_metadata=final_metadata)
            elif element_type == 'image':
                upsert_chunk(doc_id=doc_id, image_chunk_bytes=content, chunk_metadata=final_metadata)

        logger.info(f"BG Task: Successfully processed PDF for doc_id: {doc_id}")
    except Exception as e:
        logger.error(f"BG Task: Error processing doc_id {doc_id}: {e}", exc_info=True)
    finally:
        os.remove(temp_path)
        logger.info(f"BG Task: Cleaned up temp PDF: {temp_path}")

def process_and_embed_video(temp_path, original_filename, doc_id, course_metadata):
    temp_folder = Path(TEMP_FILES_DIR) / Path(temp_path).stem
    try:
        logger.info(f"BG Task: Starting video processing for {original_filename}")
        video_metadata = {
            **course_metadata,
            'original_filename': original_filename,
            'content_type': 'video'
        }
        
        frames = extract_frames(temp_path, str(temp_folder / "frames"))
        wav_path = extract_audio(temp_path, str(temp_folder / "audio.wav"))

        if not frames:
            logger.error(f"BG Task: No frames extracted from {original_filename}")
            return

        v_emb = encoder.encode_video_from_frames(frames)
        a_emb = encoder.encode_audio(wav_path) if wav_path else None

        upsert_video_audio_embeddings(
            doc_id=doc_id,
            metadata=video_metadata,
            video_vector=v_emb,
            audio_vector=a_emb
        )
        logger.info(f"BG Task: Video processed for {original_filename}")

    except Exception as e:
        logger.error(f"BG Task: Error processing video {original_filename}: {e}", exc_info=True)
    finally:
        try:
            os.remove(temp_path)
            shutil.rmtree(temp_folder, ignore_errors=True)
            logger.info(f"BG Task: Cleaned up temp video files for {original_filename}")
        except Exception as e:
            logger.warning(f"BG Task: Failed to clean up temp files for {original_filename}: {e}")

def get_course_metadata(metadata_str = Form(...)) -> schemas.CourseMetadata:
    try:
        metadata_dict = json.loads(metadata_str)
        return schemas.CourseMetadata(**metadata_dict)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid metadata format. Must be a valid JSON string.")

@app.post("/ai/recommendations", response_model=List[schemas.RecommendationResponse], tags=["Recommendations"])
def get_recommendations(request: schemas.RecommendationRequest):
    interaction_history = user_service.get_user_interactions(request.user_id)
    recommendations = recommender.get_recommendations_for_user(
        user_id=request.user_id,
        interaction_history=interaction_history,
        limit=request.limit
    )
    return recommendations

@app.post("/ai/recommendations/keywords", response_model=List[schemas.KeywordsRecommendationResponse], tags=["Recommendations"])
def get_recommendations_by_keywords(request: schemas.KeywordsRecommendationRequest):
    if not request.keywords:
        raise HTTPException(status_code=400, detail="At least one keyword must be provided.")
    return recommender.get_recommendations_for_keywords(
        keywords=request.keywords,
        per_keyword_limit=request.per_keyword_limit,
        final_limit=request.final_limit
    )

@app.get("/ai/user/{user_id}/profile", response_model=schemas.UserProfileResponse, tags=["User Management"])
def get_user_profile(user_id: str):
    profile = user_service.get_user_profile(user_id)
    return profile

@app.post("/ai/user/preferences", status_code=200, tags=["User Management"])
def set_user_preferences(user_id: str, preferences: schemas.UserPreferences):
    success = user_service.save_user_preferences(user_id, preferences)
    if success:
        return {"message": "Preferences saved successfully."}
    raise HTTPException(status_code=500, detail="Failed to save preferences.")

@app.post("/ai/user/interact", status_code=200, tags=["User Management"])
def record_user_interaction(interaction: schemas.UserInteraction):
    success = user_service.record_interaction(interaction.user_id, interaction.point_id)
    if success:
        return {"message": "Interaction recorded successfully."}
    raise HTTPException(status_code=500, detail="Failed to record interaction.")

@app.post("/ai/upload/course", status_code=202, tags=["Content Ingestion"])
async def upload_course(
    background_tasks: BackgroundTasks,
    metadata: schemas.CourseMetadata = Depends(get_course_metadata),
    documents: List[UploadFile] = File(None, description="Liste des documents du cours (PDF, etc.)."),
    video: Optional[UploadFile] = File(None, description="Fichier vidéo principal du cours.")
):
    doc_id = str(uuid.uuid4())
    base_metadata = metadata.model_dump()

    if not documents and not video:
        raise HTTPException(status_code=400, detail="At least one document or a video must be provided.")

    if documents:
        for doc_file in documents:
            valid_ext = (".pdf", ".docx", ".pptx")
            if not doc_file.filename.lower().endswith(valid_ext):
                logger.warning(f"Invalid file type skipped: {doc_file.filename}")
                continue

            os.makedirs(TEMP_FILES_DIR, exist_ok=True)
            temp_path = TEMP_FILES_DIR / f"{uuid.uuid4()}-{doc_file.filename}"

            try:
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(doc_file.file, buffer)
            finally:
                doc_file.file.close()
            
            background_tasks.add_task(process_and_embed_pdf, str(temp_path), doc_id, doc_file.filename, base_metadata)

    if video:
        valid_ext = (".mp4", ".mov", ".mkv")
        if not video.filename.lower().endswith(valid_ext):
            raise HTTPException(status_code=400, detail=f"Invalid video file type: {video.filename}")

        os.makedirs(TEMP_FILES_DIR, exist_ok=True)
        temp_path = TEMP_FILES_DIR / f"{uuid.uuid4()}-{video.filename}"

        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
        finally:
            video.file.close()

        background_tasks.add_task(process_and_embed_video, str(temp_path), video.filename, doc_id, base_metadata)

    return {"message": "Course content accepted for processing.", "course_id": doc_id, "title": metadata.title}

@app.get("/ai/health", response_model=schemas.HealthCheckResponse, tags=["Monitoring"])
def health_check():
    try:
        qdrant_client.get_collections()
        return {"api_status": "ok", "qdrant_status": "ok"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail={"api_status": "ok", "qdrant_status": "error", "error": str(e)})