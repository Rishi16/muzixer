from pathlib import Path
import subprocess
import sys

from PySide6.QtCore import QThread, Signal

from ytmusicapi import YTMusic

from app.audio import analyze_audio_batch, pick_informative_segment, infer_key_mode_from_chroma, key_to_camelot
import librosa
from librosa import feature as _librosa_feature


class SearchWorker(QThread):
    results_ready = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, ytmusic: YTMusic, query: str):
        super().__init__()
        self.ytmusic = ytmusic
        self.query = query

    def run(self):
        try:
            results = self.ytmusic.search(self.query, filter="songs", limit=20)
            self.results_ready.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))


class RecommendationWorker(QThread):
    recommendations_ready = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, ytmusic: YTMusic, video_id: str):
        super().__init__()
        self.ytmusic = ytmusic
        self.video_id = video_id

    def run(self):
        try:
            watch = self.ytmusic.get_watch_playlist(self.video_id)
            tracks = watch.get("tracks", [])[:15]
            self.recommendations_ready.emit(tracks)
        except Exception as e:
            self.error_occurred.emit(str(e))


class DownloadWorker(QThread):
    done = Signal(str, str, str)  # path, video_id, artist
    error = Signal(str)

    def __init__(self, video_id: str, title: str, artist: str):
        super().__init__()
        self.video_id = video_id
        self.title = title
        self.artist = artist or ""

    def run(self):
        try:
            from app.utils import sanitize, DOWNLOAD_DIR, FFMPEG_PATH
            DOWNLOAD_DIR.mkdir(exist_ok=True)
            safe = sanitize(self.title)
            output_tpl = str(DOWNLOAD_DIR / f"{safe}.%(ext)s")
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "-x", "--audio-format", "mp3", "--audio-quality", "0",
                "--no-playlist",
                "-o", output_tpl,
                f"https://www.youtube.com/watch?v={self.video_id}"
            ]
            if FFMPEG_PATH:
                cmd += ["--ffmpeg-location", FFMPEG_PATH]
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, universal_newlines=True
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                self.error.emit(f"Download failed: {stderr.strip()}")
                return
            for file_path in DOWNLOAD_DIR.glob(f"{safe}*.mp3"):
                self.done.emit(str(file_path), self.video_id, self.artist)
                return
            self.error.emit("Downloaded file not found")
        except Exception as e:
            self.error.emit(f"Download error: {str(e)}")


class AudioAnalysisWorker(QThread):
    analysis_complete = Signal(str, int, str)
    error_occurred = Signal(str)
    progress_update = Signal(int, str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            self.progress_update.emit(5, "Loading audio…")
            bpm, key_name, mode = analyze_audio_batch(self.file_path)
            self.progress_update.emit(85, "Detecting key…")
            if key_name is None or mode is None:
                try:
                    y_full, sr = librosa.load(self.file_path, mono=True, duration=90, sr=22050)
                    y = pick_informative_segment(y_full, sr, target_seconds=30)
                    chroma = _librosa_feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
                    key_name, mode = infer_key_mode_from_chroma(chroma)
                except Exception:
                    pass
            camelot = key_to_camelot(key_name, mode) if key_name else "UNK"
            self.progress_update.emit(100, "Done")
            self.analysis_complete.emit(self.file_path, bpm or 0, camelot)
        except Exception as e:
            self.error_occurred.emit(str(e))
