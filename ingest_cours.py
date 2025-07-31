import os
import uuid
from pathlib import Path

from app.db.qdrant_ops import upsert_chunk, upsert_video_audio_embeddings
from app.processing.document_processor import process_pdf
from app.processing.video_processor import extract_audio, extract_frames
from app.models.encoders import encoder

COURS_DIR = Path("cours")
SUPPORTED_PDF = [".pdf"]
SUPPORTED_VIDEO = [".mp4", ".mov", ".mkv"]

def ingest_all_courses():
    if not COURS_DIR.exists():
        print(f"❌ Le dossier {COURS_DIR} n'existe pas.")
        return

    files = list(COURS_DIR.glob("*"))
    if not files:
        print("❗ Aucun fichier trouvé dans le dossier 'cours'.")
        return

    for file_path in files:
        file_ext = file_path.suffix.lower()
        doc_id = str(uuid.uuid4())
        print(f"\n🟦 Traitement de : {file_path.name}")

        if file_ext in SUPPORTED_PDF:
            print("📄 Fichier PDF détecté. Extraction du contenu en cours...")
            try:
                elements = process_pdf(str(file_path))
                for element in elements:
                    if element["type"] == "text":
                        print("📝 Texte détecté. Encodage...")
                        upsert_chunk(doc_id=doc_id, text_chunk=element["content"], chunk_metadata=element["metadata"])
                    elif element["type"] == "image":
                        print("🖼️ Image détectée. Encodage...")
                        upsert_chunk(doc_id=doc_id, image_chunk_bytes=element["content"], chunk_metadata=element["metadata"])
                print("✅ PDF traité avec succès.")
            except Exception as e:
                print(f"❌ Erreur lors du traitement PDF : {e}")

        elif file_ext in SUPPORTED_VIDEO:
            print("🎥 Fichier vidéo détecté. Extraction en cours...")
            try:
                temp_dir = file_path.parent / f"temp_{file_path.stem}"
                temp_dir.mkdir(exist_ok=True)

                print("🧩 Extraction des frames...")
                frames = extract_frames(str(file_path), str(temp_dir / "frames"))

                print("🔊 Extraction de l’audio...")
                wav_path = extract_audio(str(file_path), str(temp_dir / "audio.wav"))

                if not frames:
                    print("⚠️ Aucune frame extraite. Vidéo ignorée.")
                    continue

                print("📷 Encodage des frames vidéo...")
                v_emb = encoder.encode_video_from_frames(frames)

                print("🎧 Encodage de l’audio...")
                a_emb = encoder.encode_audio(wav_path) if wav_path else None

                print("📤 Insertion dans Qdrant...")
                upsert_video_audio_embeddings(
                    filename=file_path.name,
                    video_vector=v_emb,
                    audio_vector=a_emb
                )
                print("✅ Vidéo traitée et insérée avec succès.")
            except Exception as e:
                print(f"❌ Erreur lors du traitement vidéo : {e}")
            finally:
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass

        else:
            print(f"⚠️ Format non supporté : {file_ext}. Ignoré.")

    print("\n🎉 Tous les fichiers ont été traités.")

# Exécution directe
if __name__ == "__main__":
    ingest_all_courses()
