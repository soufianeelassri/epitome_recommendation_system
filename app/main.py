import os
import shutil
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from prometheus_fastapi_instrumentator import Instrumentator
from pathlib import Path
from app.processing.video_processor import extract_frames, extract_audio
from app.models.encoders import encoder
from app.db.qdrant_ops import upsert_video_audio_embeddings
from app.core.config import TEMP_FILES_DIR
from app.db.qdrant_ops import create_collection_if_not_exists, qdrant_client, upsert_chunk
from app.processing.document_processor import process_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Epitome Academy Recommendation System API",
    description="EPITOME ACADEMY",
    version="0.1.0",
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
            metadata = element['metadata']
            
            if element_type == 'text' or element_type == 'table':
                upsert_chunk(
                    doc_id=doc_id,
                    text_chunk=content,
                    chunk_metadata=metadata
                )
            elif element_type == 'image':
                upsert_chunk(
                    doc_id=doc_id,
                    image_chunk_bytes=content,
                    chunk_metadata=metadata
                )

        logger.info(f"BG Task: Successfully processed and initiated upsert for {len(extracted_elements)} elements from doc_id: {doc_id}")

    except Exception as e:
        logger.error(f"BG Task: Error processing doc_id {doc_id}: {e}", exc_info=True)
    finally:
        os.remove(temp_path)
        logger.info(f"BG Task: Cleaned up temporary file: {temp_path}")

# Health check endpoint: Verifies API and Qdrant status
@app.get("/ai/health", tags=["Monitoring"])
def health_check():
    try:
        qdrant_client.get_collections()
        return {"api_status": "ok", "qdrant_status": "ok"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail={"api_status": "ok", "qdrant_status": "error", "error": str(e)})

# Document upload endpoint
@app.post("/ai/upload/document", status_code=202, tags=["Document Processing"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="A PDF document to be processed.")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are accepted.")

    os.makedirs(TEMP_FILES_DIR, exist_ok=True)
    temp_filename = f"{uuid.uuid4()}-{file.filename}"
    temp_path = os.path.join(TEMP_FILES_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Could not save uploaded file.")
    finally:
        file.file.close()

    doc_id = str(uuid.uuid4())
    
    background_tasks.add_task(process_and_embed_pdf, temp_path, doc_id, file.filename)

    return {"message": "PDF document accepted for processing.", "doc_id": doc_id, "filename": file.filename}

@app.post("/ai/upload/video", status_code=202, tags=["Video Processing"])
async def upload_video(
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
    except Exception as e:
        logger.error(f"Failed to save uploaded video {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Could not save uploaded video.")
    finally:
        file.file.close()

    try:
        video_name = Path(file.filename).name
        temp_folder = Path(TEMP_FILES_DIR) / Path(temp_filename).stem
        frames = extract_frames(temp_path, str(temp_folder / "frames"))
        wav_path = extract_audio(temp_path, str(temp_folder / "audio.wav"))

        if not frames:
            raise HTTPException(status_code=500, detail="No frames could be extracted from video.")

        v_emb = encoder.encode_video_from_frames(frames)
        a_emb = encoder.encode_audio(wav_path) if wav_path else None

        upsert_video_audio_embeddings(
            filename=video_name,
            video_vector=v_emb,
            audio_vector=a_emb
        )

        logger.info(f"Video processed and embeddings stored: {video_name}")
        return {"message": "Video processed and embedded successfully", "filename": video_name}

    except Exception as e:
        logger.error(f"Error processing video {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Video processing failed.")

    finally:
        try:
            os.remove(temp_path)
            shutil.rmtree(temp_folder, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to clean up temp files for {file.filename}: {e}")
