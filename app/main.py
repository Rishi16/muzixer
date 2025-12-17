import sys
import os
import subprocess
import re
import shlex
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QListWidget, QTableWidget,
    QTableWidgetItem
)
from PySide6.QtCore import QThread, Signal
from ytmusicapi import YTMusic
import librosa

# ================= CONFIG =================
DOWNLOAD_DIR = os.path.abspath("downloads")
FFMPEG_PATH = r"C:\ffmpeg\bin"   # adjust if needed

# ================= UTILS =================
def log(msg):
    print(msg)

def sanitize_title(title):
    return re.sub(r"[^a-zA-Z0-9 _-]", "", title)

def analyze_bpm(path):
    try:
        y, sr = librosa.load(path, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return int(round(float(tempo)))
    except Exception as e:
        log(f"BPM failed for {path}: {e}")
        return None

def bpm_compatible(base, other, tol=4):
    if not base or not other:
        return False
    return (
        abs(base - other) <= tol or
        abs(base - (other * 2)) <= tol or
        abs(base - (other / 2)) <= tol
    )

# ================= DOWNLOAD THREAD =================
class DownloadWorker(QThread):
    status = Signal(str)
    done = Signal(str)

    def __init__(self, video_id, title):
        super().__init__()
        self.video_id = video_id
        self.title = title

    def run(self):
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

        safe_title = sanitize_title(self.title)
        output_tpl = os.path.join(DOWNLOAD_DIR, f"{safe_title}.%(ext)s")

        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--ffmpeg-location", FFMPEG_PATH,
            "-x", "--audio-format", "mp3",
            "--audio-quality", "0",
            "-o", output_tpl,
            f"https://www.youtube.com/watch?v={self.video_id}"
        ]

        self.status.emit(f"Downloading: {self.title}")
        log("CMD:", " ".join(cmd))

        result = subprocess.run(
            " ".join(shlex.quote(c) for c in cmd),
            shell=True, capture_output=True, text=True
        )

        log(result.stdout)
        log(result.stderr)

        for f in os.listdir(DOWNLOAD_DIR):
            if safe_title.lower() in f.lower() and f.endswith(".mp3"):
                self.done.emit(os.path.join(DOWNLOAD_DIR, f))
                return

        self.done.emit(None)

# ================= MAIN UI =================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muzixer – YouTube DJ Lab")
        self.resize(1200, 750)

        self.ytmusic = YTMusic()
        self.search_results = []
        self.reco_tracks = []
        self.current_bpm = None
        self.track_bpms = {}

        self.build_ui()
        self.refresh_library()

    # ---------- UI ----------
    def build_ui(self):
        main = QVBoxLayout()
        lists = QHBoxLayout()

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search YouTube Music")
        self.search_btn = QPushButton("Search")

        self.search_list = QListWidget()
        self.reco_list = QListWidget()
        self.reco_list.setSelectionMode(QListWidget.MultiSelection)

        lists.addWidget(self.search_list)
        lists.addWidget(self.reco_list)

        self.download_btn = QPushButton("Download Selected")
        self.status = QLabel("Idle")

        # ---- Library ----
        self.lib_search = QLineEdit()
        self.lib_search.setPlaceholderText("Search downloaded songs (name or BPM)")

        self.library = QTableWidget()
        self.library.setColumnCount(2)
        self.library.setHorizontalHeaderLabels(["BPM", "Song"])
        self.library.setSortingEnabled(True)

        main.addWidget(self.search_box)
        main.addWidget(self.search_btn)
        main.addLayout(lists)
        main.addWidget(self.download_btn)
        main.addWidget(self.status)
        main.addWidget(QLabel("Downloaded Library"))
        main.addWidget(self.lib_search)
        main.addWidget(self.library)

        self.setLayout(main)

        # ---- Signals ----
        self.search_btn.clicked.connect(self.search)
        self.search_list.itemClicked.connect(self.load_recommendations)
        self.download_btn.clicked.connect(self.download_selected)
        self.lib_search.textChanged.connect(self.filter_library)

    # ---------- SEARCH ----------
    def search(self):
        q = self.search_box.text().strip()
        if not q:
            return

        self.search_list.clear()
        self.reco_list.clear()
        self.search_results = self.ytmusic.search(q, filter="songs", limit=10)

        for r in self.search_results:
            title = r["title"]
            artists = ", ".join(a["name"] for a in r.get("artists", []))
            self.search_list.addItem(f"{title} — {artists}")

    # ---------- RECOMMEND ----------
    def load_recommendations(self):
        idx = self.search_list.currentRow()
        video_id = self.search_results[idx]["videoId"]

        self.reco_list.clear()
        watch = self.ytmusic.get_watch_playlist(video_id)
        self.reco_tracks = watch["tracks"][:10]

        for t in self.reco_tracks:
            self.reco_list.addItem(t["title"])

    # ---------- DOWNLOAD ----------
    def download_selected(self):
        for idx in self.reco_list.selectedIndexes():
            track = self.reco_tracks[idx.row()]
            worker = DownloadWorker(track["videoId"], track["title"])
            worker.status.connect(self.update_status)
            worker.done.connect(self.download_finished)
            worker.start()

    def update_status(self, msg):
        self.status.setText(msg)

    def download_finished(self, path):
        if not path:
            self.status.setText("Download failed")
            return

        bpm = analyze_bpm(path)
        bpm_txt = str(bpm) if bpm else "UNK"

        new_name = f"{bpm_txt} - {os.path.basename(path)}"
        new_path = os.path.join(DOWNLOAD_DIR, new_name)
        os.rename(path, new_path)

        self.track_bpms[new_path] = bpm
        if not self.current_bpm and bpm:
            self.current_bpm = bpm

        self.status.setText(f"Downloaded: {new_name}")
        self.refresh_library()
        self.update_reco_bpm()

    # ---------- LIBRARY ----------
    def refresh_library(self):
        self.library.setRowCount(0)

        for f in sorted(os.listdir(DOWNLOAD_DIR)):
            if not f.endswith(".mp3"):
                continue
            bpm = f.split(" - ")[0]
            row = self.library.rowCount()
            self.library.insertRow(row)
            self.library.setItem(row, 0, QTableWidgetItem(bpm))
            self.library.setItem(row, 1, QTableWidgetItem(f))

    def filter_library(self, text):
        text = text.lower()
        for row in range(self.library.rowCount()):
            bpm = self.library.item(row, 0).text().lower()
            name = self.library.item(row, 1).text().lower()
            self.library.setRowHidden(row, not (text in bpm or text in name))

    # ---------- BPM UI ----------
    def update_reco_bpm(self):
        for i, track in enumerate(self.reco_tracks):
            title = track["title"]
            safe = sanitize_title(title)
            for f in os.listdir(DOWNLOAD_DIR):
                if safe.lower() in f.lower():
                    bpm = int(f.split(" - ")[0]) if f.split(" - ")[0].isdigit() else None
                    symbol = "✔" if bpm_compatible(self.current_bpm, bpm) else "✖"
                    self.reco_list.item(i).setText(f"{symbol} {title} ({bpm})")

# ================= MAIN =================
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
