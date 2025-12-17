import sys
import os
import subprocess
import librosa
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QLineEdit, QPushButton, QListWidget, QHBoxLayout
)
from PySide6.QtCore import QThread, Signal
from ytmusicapi import YTMusic


DOWNLOAD_DIR = os.path.abspath("downloads")


def analyze_bpm(file_path):
    y, sr = librosa.load(file_path, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return round(float(tempo))


def bpm_compatible(current, candidate, tolerance=4):
    if current is None or candidate is None:
        return False

    return (
        abs(current - candidate) <= tolerance or
        abs(current - (candidate * 2)) <= tolerance or
        abs(current - (candidate / 2)) <= tolerance
    )


class DownloadWorker(QThread):
    status = Signal(str)
    done = Signal(object, str)

    def __init__(self, video_id, title):
        super().__init__()
        self.video_id = video_id
        self.title = title
        self.file_path = None

    def run(self):
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        output_template = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

        for f in os.listdir(DOWNLOAD_DIR):
            if self.title.lower() in f.lower():
                self.file_path = os.path.join(DOWNLOAD_DIR, f)
                self.status.emit(f"Skipped (exists): {self.title}")
                self.done.emit(self, self.file_path)
                return

        self.status.emit(f"Downloading: {self.title}")

        subprocess.run(
            [
                "yt-dlp",
                "--ffmpeg-location", "C:\\ffmpeg\\bin",
                "-x",
                "--audio-format", "mp3",
                "-o", output_template,
                f"https://music.youtube.com/watch?v={self.video_id}"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        for f in os.listdir(DOWNLOAD_DIR):
            if self.title.lower() in f.lower():
                self.file_path = os.path.join(DOWNLOAD_DIR, f)
                break

        self.status.emit(f"Downloaded: {self.title}")
        self.done.emit(self, self.file_path)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YouTube DJ Lab")
        self.resize(1100, 650)

        self.ytmusic = YTMusic()
        self.search_results = []
        self.upnext_tracks = []
        self.download_workers = []
        self.track_bpms = {}
        self.current_bpm = None

        # Layouts
        main_layout = QVBoxLayout()
        lists_layout = QHBoxLayout()

        # Search
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search YouTube Music")
        self.search_button = QPushButton("Search")

        # Lists
        self.search_list = QListWidget()
        self.upnext_list = QListWidget()
        self.upnext_list.setSelectionMode(QListWidget.MultiSelection)

        # Download button
        self.download_button = QPushButton("Download Selected Recommendation")
        self.status_label = QLabel("Status: Idle")

        lists_layout.addWidget(self.search_list)
        lists_layout.addWidget(self.upnext_list)

        main_layout.addWidget(self.search_box)
        main_layout.addWidget(self.search_button)
        main_layout.addLayout(lists_layout)
        main_layout.addWidget(self.download_button)
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

        # Signals
        self.search_button.clicked.connect(self.perform_search)
        self.search_list.itemClicked.connect(self.fetch_up_next)
        self.download_button.clicked.connect(self.download_selected)

    def perform_search(self):
        query = self.search_box.text().strip()
        if not query:
            return

        self.search_list.clear()
        self.upnext_list.clear()

        self.search_results = self.ytmusic.search(query, filter="songs", limit=10)

        for item in self.search_results:
            title = item.get("title", "Unknown")
            artists = ", ".join(a["name"] for a in item.get("artists", []))
            self.search_list.addItem(f"{title} — {artists}")

    def fetch_up_next(self):
        index = self.search_list.currentRow()
        selected = self.search_results[index]

        video_id = selected.get("videoId")
        if not video_id:
            return

        watch = self.ytmusic.get_watch_playlist(video_id)
        self.upnext_tracks = watch.get("tracks", [])[:10]

        self.upnext_list.clear()
        for track in self.upnext_tracks:
            title = track.get("title", "Unknown")
            artists = ", ".join(a["name"] for a in track.get("artists", []))
            self.upnext_list.addItem(f"{title} — {artists}")

    def download_selected(self):
        selected_items = self.upnext_list.selectedIndexes()
        if not selected_items:
            return

        for index_obj in selected_items:
            index = index_obj.row()
            track = self.upnext_tracks[index]

            title = track.get("title", "Unknown")
            video_id = track.get("videoId")

            worker = DownloadWorker(video_id, title)
            worker.status.connect(self.update_status)
            worker.done.connect(self.cleanup_worker)

            self.download_workers.append(worker)
            worker.start()

    def update_status(self, text):
        self.status_label.setText(f"Status: {text}")

    def cleanup_worker(self, worker, file_path):
        if worker in self.download_workers:
            self.download_workers.remove(worker)

        if file_path and os.path.exists(file_path):
            bpm = analyze_bpm(file_path)
            self.track_bpms[file_path] = bpm

            if self.current_bpm is None:
                self.current_bpm = bpm
                self.status_label.setText(f"Current BPM set to {bpm}")
            else:
                compatible = bpm_compatible(self.current_bpm, bpm)
                symbol = "✔" if compatible else "✖"
                self.status_label.setText(
                    f"{symbol} {os.path.basename(file_path)} – BPM {bpm}"
                )

        worker.quit()
        worker.wait()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
