import subprocess
import json
import glob
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def extract_frames(video_path, out_dir, fps = 1):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        f"{out_dir}/frame_%04d.jpg",
        "-loglevel", "error"
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract frames from {video_path}: {e}")
        return []

    return sorted(glob.glob(f"{out_dir}/frame_*.jpg"))


def has_audio(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "json", video_path
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
        return bool(json.loads(output).get("streams"))
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to check audio presence in {video_path}: {e}")
        return False


def extract_audio(video_path, audio_out, sr = 16000):
    if not has_audio(video_path):
        logger.info(f"No audio stream found in {video_path} → skipping extraction.")
        return None

    Path(audio_out).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path, "-vn",
        "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le",
        audio_out, "-loglevel", "error", "-y"
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract audio from {video_path}: {e}")
        return None

    return audio_out
