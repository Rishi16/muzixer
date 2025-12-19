import sys
import re
import json
import subprocess
from pathlib import Path
import time
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QListWidget, QSplitter,
    QTableWidget, QTableWidgetItem, QProgressBar, QTextEdit,
    QCheckBox, QHeaderView, QAbstractItemView, QDialog, QTextBrowser
)
from PySide6.QtCore import QThread, Signal, QTimer, QMutex, Qt, QObject
from PySide6.QtGui import QFont, QKeySequence, QShortcut

from ytmusicapi import YTMusic
import librosa
from librosa import feature as _librosa_feature

# Try optional mutagen for ID3 tagging
try:
    from mutagen.easyid3 import EasyID3  # type: ignore
    ID3_AVAILABLE = True
except Exception:
    ID3_AVAILABLE = False

# ================= CONFIG =================
DOWNLOAD_DIR = Path("downloads").absolute()
FFMPEG_PATH = r"C:\ffmpeg\bin"   # change if needed (or set to "" to use PATH)
MAX_SEARCH_RESULTS = 20
MAX_RECOMMENDATIONS = 15
AUDIO_ANALYSIS_CACHE = {}  # Cache for BPM/key analysis
CACHE_MUTEX = QMutex()

# Global debug settings
DEBUG_ENABLED = False  # toggle via checkbox at bottom

KEY_MAP = {
    "C": ("8B", "5A"), "C#": ("3B", "12A"), "D": ("10B", "7A"),
    "D#": ("5B", "2A"), "E": ("12B", "9A"), "F": ("7B", "4A"),
    "F#": ("2B", "11A"), "G": ("9B", "6A"), "G#": ("4B", "1A"),
    "A": ("11B", "8A"), "A#": ("6B", "3A"), "B": ("1B", "10A")
}

# Cache for file existence checks
_file_cache = {}
_cache_timestamp = 0

# ================= THREAD-SAFE GUI LOGGER =================
class GuiLogger(QObject):
    log = Signal(str)

GUI_LOGGER = GuiLogger()  # thread-safe signal emitter to append logs in the UI

class MutexGuard:
    """Context manager to safely lock/unlock a QMutex."""
    def __init__(self, mutex: QMutex):
        self.mutex = mutex
    def __enter__(self):
        self.mutex.lock()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.mutex.unlock()

# ================= LOGGING =================
def debug_log(category, message, *args):
    """Thread-safe logging via signal to the GUI console (respects DEBUG_ENABLED)."""
    if not DEBUG_ENABLED:
        return
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    full_message = f"[{timestamp}] [{category.upper()}] {message}"
    if args:
        full_message += " " + " ".join(str(arg) for arg in args)
    GUI_LOGGER.log.emit(full_message)

def event_log(message):
    """Always-visible major event logs regardless of DEBUG flag. (No print to avoid duplicates)"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    full_message = f"[{timestamp}] [EVENT] {message}"
    GUI_LOGGER.log.emit(full_message)

# ================= MIX COMPATIBILITY (used for library recommendations) =================
def parse_camelot(c):
    """Parse Camelot notation into number and mode."""
    if not c or c == "UNK":
        return None, None
    try:
        num = int(c[:-1])
        mode = c[-1]
        return num, mode
    except (ValueError, IndexError):
        return None, None

def camelot_compatible(a, b):
    """Check harmonic compatibility on Camelot wheel."""
    if not a or not b or "UNK" in (a, b):
        return False
    n1, m1 = parse_camelot(a)
    n2, m2 = parse_camelot(b)
    if n1 is None or n2 is None:
        return False
    # Same key
    if n1 == n2 and m1 == m2:
        return True
    # Adjacent number, same mode (wraparound)
    if m1 == m2 and (abs(n1 - n2) == 1 or abs(n1 - n2) == 11):
        return True
    # Same number, opposite mode
    if n1 == n2 and m1 != m2:
        return True
    return False

def key_proximity_score(a, b):
    """Extended harmonic proximity beyond strict compatibility (0-100)."""
    if not a or not b or "UNK" in (a, b):
        return 0
    n1, m1 = parse_camelot(a)
    n2, m2 = parse_camelot(b)
    if n1 is None or n2 is None:
        return 0
    # Exact/standard compat
    if n1 == n2 and m1 == m2:
        return 100
    if n1 == n2 and m1 != m2:
        return 85
    dist = abs(n1 - n2)
    dist = min(dist, 12 - dist)  # wrap-around distance on the wheel
    if m1 == m2 and dist == 1:
        return 90
    if m1 == m2 and dist == 2:
        return 65
    if m1 == m2 and dist == 3:
        return 45
    return 0

def bpm_tier_score(a, b):
    """Tiered BPM scoring including half/double tempo with tier labels."""
    if not a or not b:
        return 0, "unknown"
    diff = abs(a - b)
    # Tight match
    if diff <= 2:
        return 100, "tight"
    # Close
    if diff <= 5:
        return 85, "close"
    # Half/double ratio within tolerance
    if abs(a - b * 2) <= 3 or abs(a * 2 - b) <= 3:
        return 70, "ratio"
    # Loose but possibly acceptable
    if diff <= 8:
        return 50, "loose"
    return 0, "far"

def mix_score(cur_bpm, cur_key, bpm, key):
    """Aggregate compatibility score using weighted BPM and key proximity (0-100)."""
    bpm_score_val, _ = bpm_tier_score(cur_bpm, bpm)
    key_score_val = key_proximity_score(cur_key, key)
    # Weight BPM slightly higher than key for on-deck transitions
    return int(round(0.6 * bpm_score_val + 0.4 * key_score_val))

# ================= FILE CACHE =================
def invalidate_file_cache():
    """Invalidate file cache when directory changes."""
    global _file_cache, _cache_timestamp
    debug_log("cache", "Invalidating file cache")
    _file_cache.clear()
    _cache_timestamp = time.time()

def get_cached_files():
    """Get cached directory listing with automatic refresh."""
    global _file_cache, _cache_timestamp
    current_time = time.time()
    if not _file_cache or (current_time - _cache_timestamp) > 5:
        debug_log("cache", f"Refreshing file cache. Age: {current_time - _cache_timestamp:.2f}s")
        if DOWNLOAD_DIR.exists():
            _file_cache = {f.name.lower(): f for f in DOWNLOAD_DIR.iterdir() if f.suffix.lower() == '.mp3'}
            debug_log("cache", f"Cached {len(_file_cache)} mp3 files")
        else:
            _file_cache = {}
            debug_log("cache", "Download directory does not exist")
        _cache_timestamp = current_time
    else:
        debug_log("cache", f"Using cached files ({len(_file_cache)} items)")
    return _file_cache

# ================= AUDIO ANALYSIS =================
# Krumhansl-Schmuckler key profiles (normalized) for major/minor
_KS_PROFILE_MAJOR = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88
]
_KS_PROFILE_MINOR = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17
]
_KEYS_12 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def _rotate(lst, n):
    n %= len(lst)
    return lst[n:] + lst[:n]

def infer_key_mode_from_chroma(chroma):
    """Infer pitch class (C..B) and mode ('major'/'minor') using K-S profiles.
    chroma: (12, frames) array. Returns (key_name, mode).
    """
    import numpy as _np  # local import to avoid global aliasing
    if chroma is None or chroma.size == 0:
        return None, None
    chroma_mean = chroma.mean(axis=1)
    # Normalize
    if chroma_mean.max() > 0:
        chroma_mean = chroma_mean / (chroma_mean.max() + 1e-9)
    best_score = -1.0
    best_key = None
    best_mode = None
    for i in range(12):
        prof_maj = _np.array(_rotate(_KS_PROFILE_MAJOR, i))
        prof_min = _np.array(_rotate(_KS_PROFILE_MINOR, i))
        # Normalize profiles
        prof_maj = prof_maj / (prof_maj.max() + 1e-9)
        prof_min = prof_min / (prof_min.max() + 1e-9)
        s_maj = float(_np.dot(chroma_mean, prof_maj))
        s_min = float(_np.dot(chroma_mean, prof_min))
        if s_maj > best_score:
            best_score = s_maj; best_key = _KEYS_12[i]; best_mode = "major"
        if s_min > best_score:
            best_score = s_min; best_key = _KEYS_12[i]; best_mode = "minor"
    return best_key, best_mode

def pick_informative_segment(y, sr, target_seconds=30):
    """Pick a high-energy contiguous segment to avoid long intros/silence.
    Returns a mono array segment of ~target_seconds length (or shorter if not enough audio).
    """
    import numpy as _np
    if y is None or len(y) == 0:
        return y
    seg_len = int(target_seconds * sr)
    if len(y) <= seg_len:
        return y
    # Window step of 5 seconds
    step = int(5 * sr)
    best_start = 0
    best_energy = -1.0
    for start in range(0, max(1, len(y) - seg_len + 1), step):
        end = start + seg_len
        window = y[start:end]
        # Energy: sum of squares (robust for beats)
        e = float(_np.sum(window.astype(_np.float32) ** 2))
        if e > best_energy:
            best_energy = e
            best_start = start
    return y[best_start: best_start + seg_len]

def detect_key(path):
    """Detect musical key using chromagram analysis with caching (mode-aware)."""
    path_str = str(path)
    with MutexGuard(CACHE_MUTEX):
        if path_str in AUDIO_ANALYSIS_CACHE and 'key' in AUDIO_ANALYSIS_CACHE[path_str]:
            # If we cached (key, mode), return just key for backward compat
            cached = AUDIO_ANALYSIS_CACHE[path_str]['key']
            if isinstance(cached, dict):
                return cached.get('name')
            return cached
    try:
        # Load up to 90 seconds then pick a 30s high-energy window
        y_full, sr = librosa.load(path_str, mono=True, duration=90)
        y = pick_informative_segment(y_full, sr, target_seconds=30)
        chroma = _librosa_feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
        key_name, mode = infer_key_mode_from_chroma(chroma)
        with MutexGuard(CACHE_MUTEX):
            AUDIO_ANALYSIS_CACHE.setdefault(path_str, {})['key'] = {"name": key_name, "mode": mode}
        return key_name
    except Exception as e:
        debug_log("analysis", f"detect_key failed: {e}")
        return None

def key_to_camelot(key, mode="minor"):
    """Convert musical key to Camelot notation. Mode is 'major' or 'minor'."""
    if not key:
        return None
    major, minor = KEY_MAP.get(key, (None, None))
    if mode not in ("major", "minor"):
        mode = "minor"
    camelot = minor if mode == "minor" else major
    return camelot

def analyze_bpm(path):
    """Analyze BPM of audio file using librosa with caching and optimization."""
    path_str = str(path)
    with MutexGuard(CACHE_MUTEX):
        if path_str in AUDIO_ANALYSIS_CACHE and 'bpm' in AUDIO_ANALYSIS_CACHE[path_str]:
            return AUDIO_ANALYSIS_CACHE[path_str]['bpm']
    try:
        y_full, sr = librosa.load(path_str, mono=True, duration=90, sr=22050)
        y = pick_informative_segment(y_full, sr, target_seconds=30)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=1024)
        result = int(round(float(tempo))) if tempo is not None else None
        with MutexGuard(CACHE_MUTEX):
            AUDIO_ANALYSIS_CACHE.setdefault(path_str, {})['bpm'] = result
        return result
    except Exception as e:
        debug_log("analysis", f"analyze_bpm failed: {e}")
        return None

def analyze_audio_batch(path):
    """Analyze both BPM and key (with mode) in a single pass for efficiency."""
    path_str = str(path)
    with MutexGuard(CACHE_MUTEX):
        entry = AUDIO_ANALYSIS_CACHE.get(path_str, {})
        if 'bpm' in entry and 'key' in entry:
            k = entry['key']
            if isinstance(k, dict):
                return entry['bpm'], k.get('name'), k.get('mode')
            return entry['bpm'], k, None
    try:
        y_full, sr = librosa.load(path_str, mono=True, duration=90, sr=22050)
        y = pick_informative_segment(y_full, sr, target_seconds=30)
        # Emit no progress here (pure function) — progress will be emitted from worker
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=1024)
        bpm = int(round(float(tempo))) if tempo is not None else None
        chroma = _librosa_feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
        key_name, mode = infer_key_mode_from_chroma(chroma)
        with MutexGuard(CACHE_MUTEX):
            AUDIO_ANALYSIS_CACHE.setdefault(path_str, {}).update({'bpm': bpm, 'key': {"name": key_name, "mode": mode}})
        return bpm, key_name, mode
    except Exception as e:
        debug_log("analysis", f"analyze_audio_batch failed: {e}")
        return None, None, None

# ================= HELPERS =================
def sanitize(text):
    """Remove special characters for safe filenames."""
    return re.sub(r"[^a-zA-Z0-9 _-]", "", text)

def find_existing_track(title):
    """Check if track already exists in download directory using cached listing."""
    safe = sanitize(title).lower()
    cached_files = get_cached_files()
    for filename_lower, filepath in cached_files.items():
        if safe in filename_lower:
            return filepath
    return None

def extract_bpm_key(filename):
    """Extract BPM and key from filename. Expected '124 8A - Track.mp3'"""
    try:
        base_name = Path(filename).name
        if " - " not in base_name:
            return None, None
        head = base_name.split(" - ", 1)[0]
        parts = head.split()
        if len(parts) < 2 or not parts[0].isdigit():
            return None, None
        key_part = parts[1]
        if len(key_part) < 2 or not key_part[:-1].isdigit() or key_part[-1] not in "AB":
            return None, None
        return int(parts[0]), key_part
    except Exception:
        return None, None

def extract_title(filename):
    """Extract clean title from filename 'BPM KEY - Title.mp3' -> 'Title'."""
    name = Path(filename).name
    title = name.split(" - ", 1)[1] if " - " in name else name
    return re.sub(r"\.mp3$", "", title, flags=re.IGNORECASE)

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

# Embed ID3 tags if mutagen is available
def embed_id3_tags(audio_path: Path, meta: dict):
    if not ID3_AVAILABLE:
        debug_log("id3", "mutagen not available, skipping ID3 tags")
        return
    try:
        tags = EasyID3(str(audio_path))
    except Exception:
        try:
            # Create tags if file lacks ID3 header
            from mutagen.id3 import ID3, ID3NoHeaderError  # type: ignore
            try:
                ID3(str(audio_path))
            except ID3NoHeaderError:
                from mutagen.id3 import ID3 as _ID3  # type: ignore
                _id3 = _ID3()
                _id3.save(str(audio_path))
            tags = EasyID3(str(audio_path))
        except Exception as e:
            debug_log("id3", f"Failed to initialize ID3: {e}")
            return
    try:
        if meta.get("title"):
            tags["title"] = [meta["title"]]
        if meta.get("artist"):
            tags["artist"] = [meta["artist"]]
        if meta.get("bpm"):
            tags["bpm"] = [str(meta["bpm"])]
        # Use Camelot code in initial key for DJ compatibility
        if meta.get("camelot"):
            tags["initialkey"] = [meta["camelot"]]
        tags.save()
        debug_log("id3", f"ID3 tags written for {audio_path.name}")
    except Exception as e:
        debug_log("id3", f"Failed to write ID3: {e}")

# ================= CUSTOM LIST/TABLE WIDGETS =================
class NavigableList(QListWidget):
    """QListWidget with custom keyboard navigation: Left/Right to switch, Space to act.
       Arrow keys move the current row only; selection changes ONLY on Space.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.on_left = None
        self.on_right = None
        self.on_space = None

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Left and self.on_left:
            self.on_left()
            return
        if key == Qt.Key_Right and self.on_right:
            self.on_right()
            return
        if key == Qt.Key_Space and self.on_space:
            self.on_space()
            return
        super().keyPressEvent(event)

    def select_first(self):
        if self.count() > 0:
            self.setCurrentRow(0)

    def toggle_current_selection(self):
        """Toggle selection state of the current item (space)."""
        row = self.currentRow()
        if row < 0:
            if self.count() > 0:
                row = 0
                self.setCurrentRow(row)
            else:
                return
        item = self.item(row)
        if item:
            item.setSelected(not item.isSelected())

class LibraryTable(QTableWidget):
    """QTableWidget ensuring Up/Down arrows move whole-row selection and trigger updates."""
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up, Qt.Key_Down):
            row_count = self.rowCount()
            if row_count == 0:
                return
            current_row = self.currentRow()
            if current_row < 0:
                new_row = 0
            elif event.key() == Qt.Key_Up:
                new_row = max(0, current_row - 1)
            else:
                new_row = min(row_count - 1, current_row + 1)
            self.selectRow(new_row)
            event.accept()
            return
        super().keyPressEvent(event)

# ================= CONSOLE WIDGET =================
class DebugConsole(QTextEdit):
    """Debug console widget with copy functionality."""
    def __init__(self):
        super().__init__()
        self.setMaximumHeight(220)
        self.setMinimumHeight(120)
        self.setReadOnly(True)
        font = QFont("Consolas", 11)
        if not font.exactMatch():
            font = QFont("Courier New", 11)
        self.setFont(font)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e3e;
                selection-background-color: #264f78;
            }
        """)
        self.setPlaceholderText("Console - Major events are always shown. Enable debug logging for detailed logs.")

    def append_log(self, message):
        self.append(message)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def clear_console(self):
        self.clear()
        debug_log("console", "Console cleared")

# ================= THREADS =================
class SearchWorker(QThread):
    results_ready = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, ytmusic, query):
        super().__init__()
        self.ytmusic = ytmusic
        self.query = query

    def run(self):
        try:
            results = self.ytmusic.search(self.query, filter="songs", limit=MAX_SEARCH_RESULTS)
            self.results_ready.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))

class RecommendationWorker(QThread):
    recommendations_ready = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, ytmusic, video_id):
        super().__init__()
        self.ytmusic = ytmusic
        self.video_id = video_id

    def run(self):
        try:
            watch = self.ytmusic.get_watch_playlist(self.video_id)
            tracks = watch.get("tracks", [])[:MAX_RECOMMENDATIONS]
            self.recommendations_ready.emit(tracks)
        except Exception as e:
            self.error_occurred.emit(str(e))

class DownloadWorker(QThread):
    done = Signal(str, str, str)  # path, video_id, artist
    error = Signal(str)

    def __init__(self, video_id, title, artist):
        super().__init__()
        self.video_id = video_id
        self.title = title
        self.artist = artist or ""

    def run(self):
        event_log(f"Downloading selected: {self.title}")
        try:
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
                event_log(f"Downloaded: {Path(file_path).name}")
                self.done.emit(str(file_path), self.video_id, self.artist)
                invalidate_file_cache()
                return

            self.error.emit("Downloaded file not found")
        except Exception as e:
            self.error.emit(f"Download error: {str(e)}")

class AudioAnalysisWorker(QThread):
    analysis_complete = Signal(str, int, str)  # path, bpm, camelot
    error_occurred = Signal(str)
    progress_update = Signal(int, str)  # percent, stage text

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            self.progress_update.emit(5, "Loading audio…")
            # Do batch analysis and emit staged progress
            bpm, key_name, mode = analyze_audio_batch(self.file_path)
            self.progress_update.emit(65, "Estimating tempo…")
            # If BPM missing (edge), retry on a later offset quick pass
            if not bpm:
                try:
                    y_full, sr = librosa.load(self.file_path, mono=True, duration=120, sr=22050)
                    y = pick_informative_segment(y_full, sr, target_seconds=30)
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=1024)
                    bpm = int(round(float(tempo))) if tempo is not None else None
                except Exception:
                    pass
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
            event_log(f"Analyzed: {Path(self.file_path).name} (BPM: {bpm if bpm else 'UNK'}, Key: {camelot})")
            self.progress_update.emit(100, "Done")
            self.analysis_complete.emit(self.file_path, bpm or 0, camelot)
        except Exception as e:
            self.error_occurred.emit(str(e))

# ================= MAIN WINDOW =================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Muzixer – YouTube DJ Lab")
        self.resize(1280, 920)

        # Increase overall UI font
        app_font = QFont("Segoe UI", 11)
        app_instance = QApplication.instance()
        if app_instance:
            app_instance.setFont(app_font)

        # Initialize YouTube Music API
        try:
            self.ytmusic = YTMusic()
        except Exception:
            self.ytmusic = None

        self.search_results = []
        self.reco_tracks = []
        self.current_bpm = None
        self.current_key = None

        # Library index: list of dicts with path, bpm, key, title, artist, camelot parsed
        self.library_index = []
        # Fast BPM index for low-latency recommendations with large libraries
        self.bpm_index = {}  # {int_bpm: [item_dict,...]}
        # Keep pending metadata for last download (video_id, artist) by original path
        self.pending_meta_by_path = {}

        # Timers
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.perform_search)

        self.library_timer = QTimer()
        self.library_timer.setSingleShot(True)
        self.library_timer.timeout.connect(self.refresh_library_delayed)

        # Filter debounce timer
        self.filter_timer = QTimer()
        self.filter_timer.setSingleShot(True)
        self.filter_timer.timeout.connect(self.filter_library)

        # Debounced recommendations loader for search selections
        self.reco_timer = QTimer()
        self.reco_timer.setSingleShot(True)
        self.reco_timer.timeout.connect(self.load_recommendations)

        # Track progress bar usage
        self.network_progress_active = False
        self.analysis_progress_active = False
        # Track active download operations to prevent spam
        self.download_in_progress = 0

        self.build_ui()
        self.apply_fonts()
        self.setup_shortcuts()
        self.refresh_library()

    def build_ui(self):
        # Top (main) and bottom (console)
        self.main_splitter = QSplitter(Qt.Vertical)

        # Top content
        top_widget = QWidget()
        main_layout = QVBoxLayout(top_widget)

        # Search
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search YouTube Music")
        self.search_box.returnPressed.connect(self.perform_search)
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.perform_search)

        search_header = QWidget()
        search_header_layout = QVBoxLayout(search_header)
        search_header_layout.setContentsMargins(0, 0, 0, 0)
        search_header_layout.addWidget(QLabel("Search YouTube Music:"))
        search_header_layout.addWidget(self.search_box)
        search_header_layout.addWidget(self.search_btn)
        main_layout.addWidget(search_header)
        main_layout.setStretchFactor(search_header, 0)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("%p%")
        main_layout.addWidget(self.progress_bar)
        main_layout.setStretchFactor(self.progress_bar, 0)

        # Search results + recommendations (YouTube)
        lists_container = QWidget()
        lists_layout = QHBoxLayout(lists_container)
        lists_layout.setContentsMargins(0, 0, 0, 0)
        self.search_list = NavigableList()
        self.reco_list = NavigableList()
        self.reco_list.setSelectionMode(QListWidget.MultiSelection)

        # Auto-load recommendations when a search result is selected (debounced)
        self.search_list.itemSelectionChanged.connect(self.queue_recommendations_for_current)
        # Double-click to load immediately and focus recos
        self.search_list.itemDoubleClicked.connect(lambda _: self.load_recommendations_and_focus())

        # Wire navigation behavior
        self.search_list.on_right = lambda: self.focus_recommendations()
        self.search_list.on_left = None
        self.search_list.on_space = self.load_recommendations_and_focus

        self.reco_list.on_left = lambda: self.focus_search_results()
        self.reco_list.on_right = None
        self.reco_list.on_space = self.reco_list.toggle_current_selection

        lists_layout.addWidget(self.search_list)
        lists_layout.addWidget(self.reco_list)
        lists_header = QLabel("Search Results → YouTube Recommendations:")
        main_layout.addWidget(lists_header)
        main_layout.addWidget(lists_container)
        # Give lists a smaller vertical share vs library
        main_layout.setStretchFactor(lists_container, 1)

        # Download
        self.download_btn = QPushButton("Download Selected")
        self.download_btn.clicked.connect(self.download_selected)
        main_layout.addWidget(self.download_btn)
        main_layout.setStretchFactor(self.download_btn, 0)

        # Library (split into two panes)
        main_layout.addWidget(QLabel("Downloaded Library (Left) → Next Up From Library (Right):"))

        lib_splitter = QSplitter(Qt.Horizontal)
        # Left: Library table
        lib_left_widget = QWidget()
        lib_left_layout = QVBoxLayout(lib_left_widget)

        self.lib_search = QLineEdit()
        self.lib_search.setPlaceholderText("Search downloaded songs (title, BPM, key, artist)")
        self.lib_search.returnPressed.connect(self.filter_library)
        self.lib_search.textChanged.connect(self.filter_library_debounced)
        lib_left_layout.addWidget(self.lib_search)

        self.library = LibraryTable()
        self.library.setColumnCount(4)
        self.library.setHorizontalHeaderLabels(["BPM", "Key", "Song", "Artist"])
        self.library.setSortingEnabled(True)
        self.library.setSelectionBehavior(QTableWidget.SelectRows)
        self.library.setSelectionMode(QTableWidget.SingleSelection)
        self.library.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.library.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.library.itemSelectionChanged.connect(self.update_next_up_from_library)
        lib_left_layout.addWidget(self.library)

        # Right: "Next Up" list from library
        lib_right_widget = QWidget()
        lib_right_layout = QVBoxLayout(lib_right_widget)
        self.next_up_list = QListWidget()
        self.next_up_list.setToolTip("Recommended tracks from your library to play next")
        lib_right_layout.addWidget(self.next_up_list)

        lib_splitter.addWidget(lib_left_widget)
        lib_splitter.addWidget(lib_right_widget)
        lib_splitter.setStretchFactor(0, 3)
        lib_splitter.setStretchFactor(1, 2)

        main_layout.addWidget(lib_splitter)
        # Make the library section take more space vertically than search
        main_layout.setStretchFactor(lib_splitter, 3)

        # Console bottom with controls
        console_container = QWidget()
        console_container_layout = QVBoxLayout(console_container)

        # Console controls at bottom
        console_controls = QHBoxLayout()
        self.show_console_cb = QCheckBox("Show Console")
        self.show_console_cb.setChecked(False)  # hidden by default
        self.show_console_cb.toggled.connect(self.toggle_console_visibility)

        self.debug_checkbox = QCheckBox("Enable Debug Logging")
        self.debug_checkbox.setChecked(DEBUG_ENABLED)
        self.debug_checkbox.toggled.connect(self.toggle_debug)

        clear_btn = QPushButton("Clear Console")
        clear_btn.clicked.connect(self.clear_console)

        shortcuts_btn = QPushButton("Keyboard Shortcuts")
        shortcuts_btn.clicked.connect(self.show_shortcuts)

        console_controls.addWidget(self.show_console_cb)
        console_controls.addWidget(self.debug_checkbox)
        console_controls.addWidget(clear_btn)
        console_controls.addWidget(shortcuts_btn)
        console_controls.addStretch()
        console_container_layout.addLayout(console_controls)

        # Console widget
        self.console_widget = QWidget()
        console_layout = QVBoxLayout(self.console_widget)
        self.console = DebugConsole()
        if not self.logger_connected:
            GUI_LOGGER.log.connect(self.console.append_log)
            self.logger_connected = True
        console_layout.addWidget(self.console)
        # Hidden by default
        self.console_widget.setVisible(False)

        console_container_layout.addWidget(self.console_widget)

        # Splitter
        self.main_splitter.addWidget(top_widget)
        self.main_splitter.addWidget(console_container)
        self.main_splitter.setStretchFactor(0, 7)
        self.main_splitter.setStretchFactor(1, 2)

        # Set final layout
        final_layout = QVBoxLayout()
        final_layout.addWidget(self.main_splitter)
        self.setLayout(final_layout)

    def apply_fonts(self):
        """Increase fonts across major widgets."""
        list_font = QFont("Segoe UI", 12)
        table_font = QFont("Segoe UI", 12)
        edit_font = QFont("Segoe UI", 12)
        btn_font = QFont("Segoe UI", 12)

        for w in [self.search_list, self.reco_list, self.next_up_list]:
            w.setFont(list_font)
        self.library.setFont(table_font)
        self.search_box.setFont(edit_font)
        self.lib_search.setFont(edit_font)
        self.search_btn.setFont(btn_font)
        self.download_btn.setFont(btn_font)

    def setup_shortcuts(self):
        """Keyboard shortcuts for fast, keyboard-driven control."""
        sc = QShortcut(QKeySequence("Ctrl+F"), self)
        sc.activated.connect(lambda: self.search_box.setFocus())
        sc = QShortcut(QKeySequence("Ctrl+L"), self)
        sc.activated.connect(lambda: self.lib_search.setFocus())
        # Show YouTube recommendations for current selection (any section)
        sc = QShortcut(QKeySequence("Ctrl+R"), self)
        sc.activated.connect(self.show_youtube_recommendations_for_selection)

        # Move focus between lists
        sc = QShortcut(QKeySequence("Right"), self)
        sc.activated.connect(self.focus_recommendations)
        sc = QShortcut(QKeySequence("Left"), self)
        sc.activated.connect(self.focus_search_results)

        # Space action fallback
        sc = QShortcut(QKeySequence("Space"), self)
        sc.activated.connect(self.space_action_global)

        # Download selected
        sc = QShortcut(QKeySequence("Ctrl+D"), self)
        sc.activated.connect(self.download_selected)
        sc = QShortcut(QKeySequence("Ctrl+Return"), self)
        sc.activated.connect(self.download_selected)
        sc = QShortcut(QKeySequence("Ctrl+Enter"), self)
        sc.activated.connect(self.download_selected)

        # Toggle console visibility
        sc = QShortcut(QKeySequence("Ctrl+`"), self)
        sc.activated.connect(lambda: self.show_console_cb.toggle())
        # Toggle debug logging
        sc = QShortcut(QKeySequence("Ctrl+B"), self)
        sc.activated.connect(lambda: self.debug_checkbox.toggle())

        # Shortcuts help
        sc = QShortcut(QKeySequence("F1"), self)
        sc.activated.connect(self.show_shortcuts)
        sc = QShortcut(QKeySequence("Ctrl+/"), self)
        sc.activated.connect(self.show_shortcuts)

        # Refresh library
        sc = QShortcut(QKeySequence("F5"), self)
        sc.activated.connect(self.refresh_library)

        # Library navigate J/K optional, Up/Down handled by LibraryTable
        sc = QShortcut(QKeySequence("J"), self)
        sc.activated.connect(self.lib_select_next)
        sc = QShortcut(QKeySequence("K"), self)
        sc.activated.connect(self.lib_select_prev)

    def show_shortcuts(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        layout = QVBoxLayout(dlg)
        tb = QTextBrowser(dlg)
        tb.setReadOnly(True)
        tb.setFont(QFont("Segoe UI", 11))
        tb.setHtml("""
        <h3>Keyboard Shortcuts</h3>
        <ul>
          <li><b>Ctrl+F</b> – Focus Search box</li>
          <li><b>Enter</b> / <b>Ctrl+S</b> – Run search</li>
          <li><b>Right / Left</b> – Switch focus between Search Results and YouTube Recommendations</li>
          <li><b>Up / Down</b> – Move within focused list/table</li>
          <li><b>Space</b> – Toggle selection in the YouTube Recommendations list</li>
          <li><b>Ctrl+R</b> – Show YouTube recommendations for the currently selected song (search results, recommendations, library, or next-up)</li>
          <li><b>Ctrl+D</b> / <b>Ctrl+Enter</b> – Download selected recommendations</li>
          <li><b>Ctrl+L</b> – Focus Library search</li>
          <li><b>Ctrl+`</b> – Toggle Console visibility</li>
          <li><b>Ctrl+B</b> – Toggle Debug logging</li>
          <li><b>J / K</b> – Move down/up in Library table (optional)</li>
          <li><b>F1</b> / <b>Ctrl+/</b> – Show this shortcuts dialog</li>
        </ul>
        """)
        layout.addWidget(tb)
        close_btn = QPushButton("Close", dlg)
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.resize(520, 480)
        dlg.exec()

    def space_action_global(self):
        if self.search_list.hasFocus():
            self.load_recommendations_and_focus()
        elif self.reco_list.hasFocus():
            self.reco_list.toggle_current_selection()

    def focus_search_results(self):
        self.search_list.setFocus()
        if self.search_list.count() > 0 and self.search_list.currentRow() < 0:
            self.search_list.setCurrentRow(0)

    def focus_recommendations(self):
        self.reco_list.setFocus()
        if self.reco_list.count() > 0 and self.reco_list.currentRow() < 0:
            self.reco_list.setCurrentRow(0)

    def lib_select_next(self):
        """Move selection down one row in the library table."""
        if self.library.rowCount() == 0:
            return
        row = self.library.currentRow()
        new_row = min((row if row >= 0 else -1) + 1, self.library.rowCount() - 1)
        self.library.selectRow(new_row)

    def lib_select_prev(self):
        """Move selection up one row in the library table."""
        if self.library.rowCount() == 0:
            return
        row = self.library.currentRow()
        new_row = max((row if row >= 0 else 1) - 1, 0)
        self.library.selectRow(new_row)

    # ---------- Console/Debug ----------
    def toggle_console_visibility(self, visible: bool):
        self.console_widget.setVisible(visible)

    def toggle_debug(self, enabled):
        global DEBUG_ENABLED
        DEBUG_ENABLED = enabled
        event_log(f"Debug {'enabled' if enabled else 'disabled'}")

    def clear_console(self):
        if hasattr(self, "console") and self.console:
            self.console.clear_console()

    # ---------- Search ----------
    def perform_search(self):
        if not self.ytmusic:
            event_log("YouTube Music API not available")
            return
        query = self.search_box.text().strip()
        if len(query) < 3:
            return
        self.search_list.clear()
        self.reco_list.clear()
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Busy indicator during network calls
        self.network_progress_active = True
        self.update_progress_visibility()
        event_log(f"Searching: {query}")
        worker = SearchWorker(self.ytmusic, query)
        worker.results_ready.connect(self.handle_search_results)
        worker.error_occurred.connect(self.handle_search_error)
        worker.finished.connect(lambda: self.cleanup_worker(worker))
        def _end_net():
            self.network_progress_active = False
            self.progress_bar.setRange(0, 100)
            self.update_progress_visibility()
        worker.finished.connect(_end_net)
        self.active_workers.append(worker)
        worker.start()

    def handle_search_results(self, results):
        self.search_results = results
        items = []
        for r in results:
            artists = ", ".join(a["name"] for a in r.get("artists", []))
            items.append(f'{r.get("title", "Unknown")} — {artists}')
        self.search_list.addItems(items)
        if self.search_list.count() > 0:
            self.search_list.setCurrentRow(0)
            self.search_list.setFocus()
        event_log(f"Search results: {len(results)}")

    def handle_search_error(self, error):
        GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Search error: {error}")

    # ---------- Recommendations (YouTube) ----------
    def load_recommendations_and_focus(self):
        if self.search_list.count() == 0:
            return
        if self.search_list.currentRow() < 0:
            self.search_list.setCurrentRow(0)
        self.load_recommendations()
        self.focus_recommendations()

    def load_recommendations(self):
        if not self.ytmusic:
            return
        idx = self.search_list.currentRow()
        if idx < 0 or idx >= len(self.search_results):
            return
        result = self.search_results[idx]
        video_id = result.get("videoId")
        if not video_id:
            # Fallback: search by title + artist
            title = result.get("title") or ""
            artists = ", ".join(a.get("name", "") for a in result.get("artists", []) if a.get("name"))
            query = f"{title} {artists}".strip()
            if query:
                event_log(f"No direct videoId for selection; searching for: {query}")
                self.search_and_start_recos_by_title(query)
            return
        self.start_youtube_recos(video_id)

    def start_youtube_recos(self, video_id: str):
        """Start a RecommendationWorker and show results in reco_list."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Busy indicator during network calls
        self.network_progress_active = True
        self.update_progress_visibility()
        event_log("Loading YouTube recommendations...")
        worker = RecommendationWorker(self.ytmusic, video_id)
        worker.recommendations_ready.connect(self.handle_recommendations)
        worker.error_occurred.connect(self.handle_recommendation_error)
        worker.finished.connect(lambda: self.cleanup_worker(worker))
        def _end_net():
            self.network_progress_active = False
            self.progress_bar.setRange(0, 100)
            self.update_progress_visibility()
        worker.finished.connect(_end_net)
        self.active_workers.append(worker)
        worker.start()

    def handle_recommendations(self, tracks):
        self.reco_tracks = tracks
        self.reco_list.clear()
        self.reco_list.addItems([t.get("title", "Unknown") for t in tracks])
        if self.reco_list.count() > 0:
            self.reco_list.setCurrentRow(0)
            self.reco_list.setFocus()
        event_log(f"YouTube recommendations loaded: {len(tracks)}")

    def handle_recommendation_error(self, error):
        GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Recommendation error: {error}")

    def show_youtube_recommendations_for_selection(self):
        """Ctrl+R: From any section, fetch YouTube recommendations for the selected song."""
        if not self.ytmusic:
            return

        # Case 1: search results
        if self.search_list.hasFocus() and self.search_list.currentRow() >= 0:
            idx = self.search_list.currentRow()
            if 0 <= idx < len(self.search_results):
                vid = self.search_results[idx].get("videoId")
                if vid:
                    self.start_youtube_recos(vid)
                    return

        # Case 2: recommendations list (use current video's id if available)
        if self.reco_list.hasFocus() and self.reco_list.currentRow() >= 0:
            row = self.reco_list.currentRow()
            if 0 <= row < len(self.reco_tracks):
                vid = self.reco_tracks[row].get("videoId")
                if vid:
                    self.start_youtube_recos(vid)
                    return
            # fallback: query by title
            title = self.reco_list.item(row).text()
            self.search_and_start_recos_by_title(title)
            return

        # Case 3: library table
        selected_rows = self.library.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self.library_index):
                item = self.library_index[row]
                # Try metadata video id first
                audio_path = item["path"]
                meta = read_metadata(audio_path)
                vid = meta.get("video_id")
                if vid:
                    self.start_youtube_recos(vid)
                    return
                # fallback: query using title + artist
                title = item["title"]
                artist = item.get("artist") or ""
                query = f"{title} {artist}".strip()
                self.search_and_start_recos_by_title(query)
                return

        # Case 4: next_up_list
        if self.next_up_list.hasFocus() and self.next_up_list.currentRow() >= 0:
            text = self.next_up_list.item(self.next_up_list.currentRow()).text()
            # Text format: "Title (BPM BPM, KEY)" → extract title
            title = text.split(" (", 1)[0].strip()
            # Find in library index to get artist if possible
            artist = ""
            for it in self.library_index:
                if it["title"].lower() == title.lower():
                    artist = it.get("artist") or ""
                    break
            query = f"{title} {artist}".strip()
            self.search_and_start_recos_by_title(query)
            return

    def search_and_start_recos_by_title(self, query: str):
        """Search YouTube for a query and start recommendations for the top result."""
        try:
            results = self.ytmusic.search(query, filter="songs", limit=1)
            if results and results[0].get("videoId"):
                self.start_youtube_recos(results[0]["videoId"])
            else:
                event_log(f"No YouTube song result for: {query}")
        except Exception as e:
            GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Search for recos failed: {e}")

    # ---------- Download ----------
    def download_selected(self):
        # Debounce: if a download is already in progress, ignore rapid presses
        if self.download_in_progress > 0:
            event_log("Download already in progress; ignoring duplicate request")
            return
        selected_items = self.reco_list.selectedItems()
        if not selected_items:
            return
        start_count = 0
        # Disable button to prevent repeated Ctrl+D spamming
        self.download_btn.setEnabled(False)
        for item in selected_items:
            row = self.reco_list.row(item)
            if row < 0 or row >= len(self.reco_tracks):
                continue
            track = self.reco_tracks[row]
            title = track.get("title", "Unknown")
            video_id = track.get("videoId")
            artists_list = track.get("artists", [])
            artist_str = ", ".join(a.get("name", "") for a in artists_list) if artists_list else ""
            if not video_id:
                continue
            if find_existing_track(title):
                continue
            worker = DownloadWorker(video_id, title, artist_str)
            worker.done.connect(self.download_finished)
            worker.error.connect(self.download_error)
            worker.finished.connect(lambda w=worker: self.cleanup_worker(w))
            # When each worker finishes, decrement counter and re-enable btn if none remain
            def _on_worker_finished():
                self.download_in_progress = max(0, self.download_in_progress - 1)
                if self.download_in_progress == 0:
                    self.download_btn.setEnabled(True)
            worker.finished.connect(_on_worker_finished)
            self.active_workers.append(worker)
            start_count += 1
            self.download_in_progress += 1
            worker.start()
        if start_count > 0:
            event_log(f"Downloading selected: {start_count} track(s)")
        else:
            # Nothing started; re-enable the button
            self.download_btn.setEnabled(True)

    def download_error(self, error_msg):
        GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Download error: {error_msg}")

    def download_finished(self, path, video_id, artist):
        # Keep pending meta for when analysis completes and file is renamed
        base_title = Path(path).stem
        self.pending_meta_by_path[path] = {"video_id": video_id, "artist": artist or "", "title": base_title}
        if not path or not Path(path).exists():
            GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Downloaded file not found")
            return
        worker = AudioAnalysisWorker(path)
        worker.analysis_complete.connect(self.handle_audio_analysis)
        worker.error_occurred.connect(self.handle_analysis_error)
        # Analysis progress plumbing
        worker.progress_update.connect(self.handle_analysis_progress)
        self.analysis_progress_active = True
        self.update_progress_visibility()
        self.active_workers.append(worker)
        worker.start()

    # ---------- Artist enrichment ----------
    def _enrich_artist(self, video_id: str | None, title_guess: str) -> str:
        """Try to fetch artist via YTMusic using video_id or title.
        Returns artist string or empty if not found."""
        if not self.ytmusic:
            return ""
        # Prefer title search; 'get_song' may not return structured artists reliably
        try:
            q = title_guess.strip()
            if q:
                results = self.ytmusic.search(q, filter="songs", limit=1)
                if results:
                    artists_list = results[0].get("artists", [])
                    artist_str = ", ".join(a.get("name", "") for a in artists_list if a.get("name"))
                    return artist_str
        except Exception:
            pass
        # Fallback: try watch playlist to infer first track
        try:
            if video_id:
                watch = self.ytmusic.get_watch_playlist(video_id)
                tracks = watch.get("tracks", [])
                if tracks:
                    artists_list = tracks[0].get("artists", [])
                    artist_str = ", ".join(a.get("name", "") for a in artists_list if a.get("name"))
                    return artist_str
        except Exception:
            pass
        return ""

    # ---------- Post-processing ----------
    def handle_audio_analysis(self, original_path, bpm, camelot):
        try:
            path = Path(original_path)
            base = path.stem
            safe_title = sanitize(base)
            bpm_txt = str(bpm) if bpm else "UNK"
            final_name = f"{bpm_txt} {camelot} - {safe_title}.mp3"
            final_path = path.parent / final_name
            if final_path.exists():
                path.unlink(missing_ok=True)
                event_log(f"Already exists: {final_name}")
                invalidate_file_cache()
                return
            path.rename(final_path)
            event_log(f"Done: {final_name}")
            invalidate_file_cache()

            # Write metadata sidecar with artist and video_id
            meta_src = self.pending_meta_by_path.pop(str(original_path), {})
            # Use extracted title from filename for display; more consistent than raw base
            display_title = extract_title(final_path.name)
            artist_val = meta_src.get("artist", "")
            if not artist_val:
                # Try to enrich artist if missing
                artist_val = self._enrich_artist(meta_src.get("video_id"), display_title) or ""
            meta = {
                "title": display_title,
                "artist": artist_val,
                "video_id": meta_src.get("video_id", ""),
                "bpm": bpm,
                "camelot": camelot
            }
            write_metadata(final_path, meta)
            # Also embed ID3 tags if possible
            embed_id3_tags(final_path, meta)

            if self.current_bpm is None and bpm:
                self.current_bpm = bpm
                self.current_key = camelot

            self.library_timer.stop()
            self.library_timer.start(300)
        except Exception as e:
            GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Post-process error: {e}")
        finally:
            # End analysis progress display if no other analysis active
            self.analysis_progress_active = False
            self.update_progress_visibility()

    def handle_analysis_error(self, error):
        GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Analysis error: {error}")
        self.analysis_progress_active = False
        self.update_progress_visibility()

    def handle_analysis_progress(self, percent: int, stage: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(max(0, min(100, percent)))
        self.progress_bar.setFormat(f"Analyzing… {percent}% – {stage}")
        self.progress_bar.setVisible(True)

    # ---------- Library ----------
    def refresh_library_delayed(self):
        self.refresh_library()
        invalidate_file_cache()

    def refresh_library(self):
        self.library_index = []
        self.bpm_index = {}
        self.library.setRowCount(0)
        if not DOWNLOAD_DIR.exists():
            return
        files = list(DOWNLOAD_DIR.glob("*.mp3"))
        if not files:
            return

        def sort_key(p: Path):
            bpm, _ = extract_bpm_key(p.name)
            return bpm if bpm is not None else 999

        files.sort(key=sort_key)
        self.library.setRowCount(len(files))
        for row, p in enumerate(files):
            bpm, key = extract_bpm_key(p.name)
            title = extract_title(p.name)
            meta = read_metadata(p)
            artist = meta.get("artist", "")

            camelot_num, camelot_mode = parse_camelot(key)
            item = {
                "path": p,
                "bpm": bpm,
                "key": key,
                "title": title,
                "artist": artist,
                "camelot_num": camelot_num,
                "camelot_mode": camelot_mode
            }
            self.library_index.append(item)
            if bpm is not None:
                self.bpm_index.setdefault(bpm, []).append(item)

            bpm_txt = str(bpm) if bpm is not None else "UNK"
            key_txt = key if key is not None else "UNK"
            self.library.setItem(row, 0, QTableWidgetItem(bpm_txt))
            self.library.setItem(row, 1, QTableWidgetItem(key_txt))
            self.library.setItem(row, 2, QTableWidgetItem(title))
            self.library.setItem(row, 3, QTableWidgetItem(artist))

    def filter_library_debounced(self):
        self.filter_timer.stop()
        self.filter_timer.start(150)

    def filter_library(self):
        text = self.lib_search.text().lower()
        if not text:
            for r in range(self.library.rowCount()):
                self.library.setRowHidden(r, False)
            return
        for r in range(self.library.rowCount()):
            bpm_item = self.library.item(r, 0)
            key_item = self.library.item(r, 1)
            name_item = self.library.item(r, 2)
            artist_item = self.library.item(r, 3)
            if not bpm_item or not key_item or not name_item or not artist_item:
                self.library.setRowHidden(r, True)
                continue
            bpm_v = bpm_item.text().lower()
            key_v = key_item.text().lower()
            name_v = name_item.text().lower()
            artist_v = artist_item.text().lower()
            show = (text in bpm_v) or (text in key_v) or (text in name_v) or (text in artist_v)
            self.library.setRowHidden(r, not show)

    def gather_candidates_by_bpm(self, cur_bpm):
        """Low-latency candidate gathering using BPM index and tolerance ranges."""
        if cur_bpm is None or not self.bpm_index:
            return self.library_index[:200]
        seen = set()
        candidates = []
        def add_bucket(b):
            items = self.bpm_index.get(b, [])
            for it in items:
                pid = it["path"]
                if pid not in seen:
                    seen.add(pid)
                    candidates.append(it)
        for delta in range(-5, 6):
            add_bucket(cur_bpm + delta)
        half = round(cur_bpm / 2) if cur_bpm else None
        double = cur_bpm * 2 if cur_bpm else None
        for base in [half, double]:
            if base and base > 0:
                for delta in range(-3, 4):
                    add_bucket(base + delta)
        if len(candidates) < 50:
            for delta in range(-8, 9):
                add_bucket(cur_bpm + delta)
        return candidates

    def update_next_up_from_library(self):
        self.next_up_list.clear()
        selected_rows = self.library.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        if row < 0 or row >= len(self.library_index):
            return
        cur = self.library_index[row]
        cur_bpm = cur["bpm"]
        cur_key = cur["key"]
        cur_title = cur["title"]
        candidates = self.gather_candidates_by_bpm(cur_bpm)
        scored = []
        for item in candidates:
            if item["path"] == cur["path"]:
                continue
            total = mix_score(cur_bpm, cur_key, item["bpm"], item["key"])
            if total > 0:
                scored.append((total, item))
        scored.sort(key=lambda x: (x[0], x[1]["title"]), reverse=True)
        best = scored[:10]
        for total, item in best:
            bpm_txt = str(item["bpm"]) if item["bpm"] is not None else "UNK"
            key_txt = item["key"] if item["key"] is not None else "UNK"
            self.next_up_list.addItem(f'{total}% - {item["title"]} ({bpm_txt} BPM, {key_txt})')
        event_log(f"Next up (top {len(best)}) for: {cur_title}")

    # ---------- Worker cleanup ----------
    def cleanup_worker(self, worker):
        try:
            self.active_workers.remove(worker)
        except ValueError:
            pass
        worker.deleteLater()

    # ---------- Close ----------
    def closeEvent(self, event):
        for worker in self.active_workers[:]:
            if worker.isRunning():
                worker.quit()
                worker.wait(2000)
        event.accept()

    def update_progress_visibility(self):
        # Show progress bar if either network or analysis work is active
        visible = self.network_progress_active or self.analysis_progress_active
        self.progress_bar.setVisible(visible)
        if not visible:
            # Reset format for next time
            self.progress_bar.setFormat("%p%")
            self.progress_bar.setValue(0)

    def queue_recommendations_for_current(self):
        """Debounce auto-loading of YouTube recommendations when selection changes."""
        # Avoid triggering during empty state
        if self.search_list.currentRow() < 0 or not self.search_results:
            return
        # Start a short timer; if user navigates quickly, only last selection triggers
        self.reco_timer.stop()
        self.reco_timer.start(350)

