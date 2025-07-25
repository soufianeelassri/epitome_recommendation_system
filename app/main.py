import os
import shutil
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Body
from prometheus_fastapi_instrumentator import Instrumentator
from pathlib import Path

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

# Initialize FastAPI app
app = FastAPI(
    title="Epitome Academy Recommendation System API",
    description="API for content processing and recommendation at Epitome Academy.",
    version="1.0.0",
)

# Instrument Prometheus metrics before startup
Instrumentator().instrument(app).expose(app)

# Startup event: Initialize Qdrant collection and log startup
@app.on_event("startup")
def startup_event():
    logger.info("Application startup...")
    create_collection_if_not_exists()
    logger.info("Application startup complete.")

# Background task: Process and embed PDF content into Qdrant
def process_and_embed_pdf(temp_path: str, doc_id: str, original_filename: str):
    logger.info(f"BG Task: Starting processing for doc_id: {doc_id}")
    try:
        extracted_elements = process_pdf(temp_path)
        for element in extracted_elements:
            element_type = element['type']
            content = element['content']
            metadata = {**element['metadata'], 'original_filename': original_filename}
            
            if element_type == 'text' or element_type == 'table':
                upsert_chunk(doc_id=doc_id, text_chunk=content, chunk_metadata=metadata)
            elif element_type == 'image':
                upsert_chunk(doc_id=doc_id, image_chunk_bytes=content, chunk_metadata=metadata)

        logger.info(f"BG Task: Successfully processed and initiated upsert for doc_id: {doc_id}")
    except Exception as e:
        logger.error(f"BG Task: Error processing doc_id {doc_id}: {e}", exc_info=True)
    finally:
        os.remove(temp_path)
        logger.info(f"BG Task: Cleaned up temporary file: {temp_path}")

# Recommendation endpoints
@app.post("/ai/recommendations", 
          response_model=list[schemas.RecommendationResponse], 
          tags=["Recommendations"])
def get_recommendations(request: schemas.RecommendationRequest):
    interaction_history = user_service.get_user_interactions(request.user_id)
    if not interaction_history:
        raise HTTPException(
            status_code=404,
            detail=f"No interaction history found for user '{request.user_id}'. Cannot generate recommendations."
        )

    recommendations = recommender.get_recommendations_for_user(
        user_id=request.user_id,
        interaction_history=interaction_history,
        limit=request.limit
    )
    return recommendations

@app.get("/ai/user/{user_id}/profile", 
         response_model=schemas.UserProfileResponse, 
         tags=["User Management"])
def get_user_profile(user_id: str):
    profile = user_service.get_user_profile(user_id)
    if not profile["interaction_history"]:
        logger.warning(f"No profile found for user_id: {user_id}")
    return profile

@app.post("/ai/user/interact", status_code=200, tags=["User Management"])
def record_user_interaction(interaction: schemas.UserInteraction):
    success = user_service.record_interaction(
        user_id=interaction.user_id,
        point_id=interaction.point_id
    )
    if success:
        return {"message": "Interaction recorded successfully."}
    # This simple service always returns true, but this structure allows for future failure handling
    raise HTTPException(status_code=500, detail="Failed to record interaction.")


# Core service endpoints
@app.get("ai/health", response_model=schemas.HealthCheckResponse, tags=["Monitoring"])
def health_check():
    try:
        qdrant_client.get_collections()
        return {"api_status": "ok", "qdrant_status": "ok"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail={"api_status": "ok", "qdrant_status": "error", "error": str(e)})

@app.post("ai/upload/document", status_code=202, tags=["Content Ingestion"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="A document to be processed (PDF, Word, PPT).")
):
    valid_ext = (".pdf", ".docx", ".pptx")
    if not file.filename.lower().endswith(valid_ext):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF, Word, and PowerPoint are accepted.")

    os.makedirs(TEMP_FILES_DIR, exist_ok=True)
    temp_filename = f"{uuid.uuid4()}-{file.filename}"
    temp_path = os.path.join(TEMP_FILES_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    doc_id = str(uuid.uuid4())
    background_tasks.add_task(process_and_embed_pdf, temp_path, doc_id, file.filename)
    return {"message": "Document accepted for processing.", "doc_id": doc_id, "filename": file.filename}

@app.post("ai/upload/video", status_code=202, tags=["Content Ingestion"])
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="A video file to be processed (.mp4, .mov, .mkv).")
):
    valid_ext = (".mp4", ".mov", ".mkv")
    if not file.filename.lower().endswith(valid_ext):
        raise HTTPException(status_code=400, detail="Invalid file type. Only video files are accepted.")

    os.makedirs(TEMP_FILES_DIR, exist_ok=True)
    temp_filename = f"{uuid.uuid4()}-{file.filename}"
    temp_path = os.path.join(TEMP_FILES_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    # Use a background task for video processing as well
    background_tasks.add_task(process_and_embed_video, temp_path, file.filename)
    return {"message": "Video accepted for processing.", "filename": file.filename}

def process_and_embed_video(temp_path: str, original_filename: str):
    temp_folder = Path(TEMP_FILES_DIR) / Path(temp_path).stem
    try:
        logger.info(f"BG Task: Starting video processing for {original_filename}")
        frames = extract_frames(temp_path, str(temp_folder / "frames"))
        wav_path = extract_audio(temp_path, str(temp_folder / "audio.wav"))

        if not frames:
            logger.error(f"BG Task: No frames could be extracted from video {original_filename}")
            return

        v_emb = encoder.encode_video_from_frames(frames)
        a_emb = encoder.encode_audio(wav_path) if wav_path else None

        upsert_video_audio_embeddings(
            filename=original_filename,
            video_vector=v_emb,
            audio_vector=a_emb
        )
        logger.info(f"BG Task: Video processed and embeddings stored for {original_filename}")

    except Exception as e:
        logger.error(f"BG Task: Error processing video {original_filename}: {e}", exc_info=True)
    finally:
        try:
            os.remove(temp_path)
            shutil.rmtree(temp_folder, ignore_errors=True)
            logger.info(f"BG Task: Cleaned up temp files for {original_filename}")
        except Exception as e:
            logger.warning(f"BG Task: Failed to clean up temp files for {original_filename}: {e}")