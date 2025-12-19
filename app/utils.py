import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from PySide6.QtCore import QMutex

DOWNLOAD_DIR = Path("downloads").absolute()
FFMPEG_PATH = r"C:\ffmpeg\bin"

CACHE_MUTEX = QMutex()
AUDIO_ANALYSIS_CACHE: Dict[str, dict] = {}

DEBUG_ENABLED = False


def sanitize(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 _-]", "", text)


def metadata_path_for_audio(audio_path: Path) -> Path:
    return audio_path.with_suffix(".json")


def read_metadata(audio_path: Path) -> dict:
    try:
        mp = metadata_path_for_audio(audio_path)
        if mp.exists():
            with mp.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def write_metadata(audio_path: Path, meta: dict):
    try:
        mp = metadata_path_for_audio(audio_path)
        with mp.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        debug_log("error", f"Metadata write failed: {e}")


def debug_log(category, message, *args):
    if not DEBUG_ENABLED:
        return
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    full_message = f"[{timestamp}] [{category.upper()}] {message}"
    if args:
        full_message += " " + " ".join(str(arg) for arg in args)
    print(full_message)


def event_log(message):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    full_message = f"[{timestamp}] [EVENT] {message}"
    print(full_message)

