from __future__ import annotations

import sys
import json
import re
import time
from pathlib import Path
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

from app.workers import SearchWorker, RecommendationWorker, DownloadWorker, AudioAnalysisWorker
from app.audio import key_to_camelot
from app.utils import (
    DOWNLOAD_DIR, CACHE_MUTEX, AUDIO_ANALYSIS_CACHE, debug_log, event_log,
    sanitize, read_metadata, write_metadata
)

try:
    from mutagen.easyid3 import EasyID3  # type: ignore
    ID3_AVAILABLE = True
except Exception:
    ID3_AVAILABLE = False

class GuiLogger(QObject):
    log = Signal(str)

GUI_LOGGER = GuiLogger()

class DebugConsole(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setMaximumHeight(220)
        self.setMinimumHeight(120)
        self.setReadOnly(True)
        font = QFont("Consolas", 11)
        if not font.exactMatch():
            font = QFont("Courier New", 11)
        self.setFont(font)
        self.setPlaceholderText("Console - Major events are always shown. Enable debug logging for detailed logs.")
    def append_log(self, message):
        self.append(message)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muzixer – YouTube DJ Lab")
        self.resize(1280, 920)
        app_font = QFont("Segoe UI", 11)
        inst = QApplication.instance()
        if inst:
            inst.setFont(app_font)
        try:
            self.ytmusic = YTMusic()
        except Exception:
            self.ytmusic = None
        self.search_results = []
        self.reco_tracks = []
        self.current_bpm = None
        self.current_key = None
        self.library_index = []
        self.bpm_index = {}
        self.pending_meta_by_path = {}

        self.search_timer = QTimer(); self.search_timer.setSingleShot(True); self.search_timer.timeout.connect(self.perform_search)
        self.library_timer = QTimer(); self.library_timer.setSingleShot(True); self.library_timer.timeout.connect(self.refresh_library_delayed)
        self.filter_timer = QTimer(); self.filter_timer.setSingleShot(True); self.filter_timer.timeout.connect(self.filter_library)
        self.reco_timer = QTimer(); self.reco_timer.setSingleShot(True); self.reco_timer.timeout.connect(self.load_recommendations)

        self.active_workers = []
        self.logger_connected = False
        self.network_progress_active = False
        self.analysis_progress_active = False
        self.download_in_progress = 0

        self.build_ui()
        self.apply_fonts()
        self.setup_shortcuts()
        self.refresh_library()

    def build_ui(self):
        self.main_splitter = QSplitter(Qt.Vertical)
        top_widget = QWidget(); main_layout = QVBoxLayout(top_widget)
        self.search_box = QLineEdit(); self.search_box.setPlaceholderText("Search YouTube Music"); self.search_box.returnPressed.connect(self.perform_search)
        self.search_btn = QPushButton("Search"); self.search_btn.clicked.connect(self.perform_search)

        search_header = QWidget(); shl = QVBoxLayout(search_header); shl.setContentsMargins(0,0,0,0)
        shl.addWidget(QLabel("Search YouTube Music:")); shl.addWidget(self.search_box); shl.addWidget(self.search_btn)
        main_layout.addWidget(search_header); main_layout.setStretchFactor(search_header, 0)

        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False); self.progress_bar.setRange(0,100); self.progress_bar.setFormat("%p%")
        main_layout.addWidget(self.progress_bar); main_layout.setStretchFactor(self.progress_bar, 0)

        lists_container = QWidget(); lists_layout = QHBoxLayout(lists_container); lists_layout.setContentsMargins(0,0,0,0)
        self.search_list = QListWidget(); self.reco_list = QListWidget()
        self.reco_list.setSelectionMode(QListWidget.MultiSelection)
        self.search_list.itemSelectionChanged.connect(self.queue_recommendations_for_current)
        self.search_list.itemDoubleClicked.connect(lambda _: self.load_recommendations_and_focus())
        lists_layout.addWidget(self.search_list); lists_layout.addWidget(self.reco_list)
        main_layout.addWidget(QLabel("Search Results → YouTube Recommendations:")); main_layout.addWidget(lists_container)
        main_layout.setStretchFactor(lists_container, 1)

        self.download_btn = QPushButton("Download Selected"); self.download_btn.clicked.connect(self.download_selected)
        main_layout.addWidget(self.download_btn); main_layout.setStretchFactor(self.download_btn, 0)

        main_layout.addWidget(QLabel("Downloaded Library (Left) → Next Up From Library (Right):"))
        lib_splitter = QSplitter(Qt.Horizontal)
        lib_left_widget = QWidget(); lib_left_layout = QVBoxLayout(lib_left_widget)
        self.lib_search = QLineEdit(); self.lib_search.setPlaceholderText("Search downloaded songs (title, BPM, key, artist)")
        self.lib_search.returnPressed.connect(self.filter_library); self.lib_search.textChanged.connect(self.filter_library_debounced)
        lib_left_layout.addWidget(self.lib_search)
        self.library = QTableWidget(); self.library.setColumnCount(4); self.library.setHorizontalHeaderLabels(["BPM","Key","Song","Artist"])\
            ; self.library.setSortingEnabled(True); self.library.setSelectionBehavior(QTableWidget.SelectRows); self.library.setSelectionMode(QTableWidget.SingleSelection)\
            ; self.library.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.library.horizontalHeader(); header.setSectionResizeMode(0, QHeaderView.ResizeToContents); header.setSectionResizeMode(1, QHeaderView.ResizeToContents)\
            ; header.setSectionResizeMode(2, QHeaderView.Stretch); header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.library.itemSelectionChanged.connect(self.update_next_up_from_library)
        lib_left_layout.addWidget(self.library)
        lib_right_widget = QWidget(); lib_right_layout = QVBoxLayout(lib_right_widget)
        self.next_up_list = QListWidget(); lib_right_layout.addWidget(self.next_up_list)
        lib_splitter.addWidget(lib_left_widget); lib_splitter.addWidget(lib_right_widget); lib_splitter.setStretchFactor(0,3); lib_splitter.setStretchFactor(1,2)
        main_layout.addWidget(lib_splitter); main_layout.setStretchFactor(lib_splitter, 3)

        console_container = QWidget(); console_container_layout = QVBoxLayout(console_container)
        console_controls = QHBoxLayout()
        self.show_console_cb = QCheckBox("Show Console"); self.show_console_cb.setChecked(False); self.show_console_cb.toggled.connect(self.toggle_console_visibility)
        self.debug_checkbox = QCheckBox("Enable Debug Logging"); self.debug_checkbox.setChecked(False); self.debug_checkbox.toggled.connect(self.toggle_debug)
        clear_btn = QPushButton("Clear Console"); clear_btn.clicked.connect(self.clear_console)
        shortcuts_btn = QPushButton("Keyboard Shortcuts"); shortcuts_btn.clicked.connect(self.show_shortcuts)
        console_controls.addWidget(self.show_console_cb); console_controls.addWidget(self.debug_checkbox); console_controls.addWidget(clear_btn); console_controls.addWidget(shortcuts_btn); console_controls.addStretch()
        console_container_layout.addLayout(console_controls)
        self.console = DebugConsole();
        if not self.logger_connected:
            GUI_LOGGER.log.connect(self.console.append_log); self.logger_connected = True
        console_container_layout.addWidget(self.console)
        self.console.setVisible(False)
        self.main_splitter.addWidget(top_widget); self.main_splitter.addWidget(console_container); self.main_splitter.setStretchFactor(0,7); self.main_splitter.setStretchFactor(1,2)
        final_layout = QVBoxLayout(); final_layout.addWidget(self.main_splitter); self.setLayout(final_layout)

    def apply_fonts(self):
        f = QFont("Segoe UI", 12)
        for w in [self.search_list, self.reco_list, self.next_up_list, self.library, self.search_box, self.lib_search, self.search_btn, self.download_btn]:
            w.setFont(f)

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+F"), self).activated.connect(lambda: self.search_box.setFocus())
        QShortcut(QKeySequence("Ctrl+L"), self).activated.connect(lambda: self.lib_search.setFocus())
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.show_youtube_recommendations_for_selection)
        QShortcut(QKeySequence("Right"), self).activated.connect(self.focus_recommendations)
        QShortcut(QKeySequence("Left"), self).activated.connect(self.focus_search_results)
        QShortcut(QKeySequence("Space"), self).activated.connect(self.space_action_global)
        QShortcut(QKeySequence("Ctrl+D"), self).activated.connect(self.download_selected)
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.download_selected)
        QShortcut(QKeySequence("Ctrl+Enter"), self).activated.connect(self.download_selected)
        QShortcut(QKeySequence("Ctrl+`"), self).activated.connect(lambda: self.show_console_cb.toggle())
        QShortcut(QKeySequence("Ctrl+B"), self).activated.connect(lambda: self.debug_checkbox.toggle())
        QShortcut(QKeySequence("F1"), self).activated.connect(self.show_shortcuts)
        QShortcut(QKeySequence("Ctrl+/"), self).activated.connect(self.show_shortcuts)
        QShortcut(QKeySequence("F5"), self).activated.connect(self.refresh_library)

    def show_shortcuts(self):
        dlg = QDialog(self); dlg.setWindowTitle("Keyboard Shortcuts"); layout = QVBoxLayout(dlg); tb = QTextBrowser(dlg); tb.setReadOnly(True); tb.setFont(QFont("Segoe UI", 11))
        tb.setHtml("""
        <h3>Keyboard Shortcuts</h3>
        <ul>
          <li><b>Ctrl+F</b> – Focus Search box</li>
          <li><b>Enter</b> / <b>Ctrl+S</b> – Run search</li>
          <li><b>Right / Left</b> – Switch focus between Search Results and YouTube Recommendations</li>
          <li><b>Up / Down</b> – Move within focused list/table</li>
          <li><b>Space</b> – Toggle selection in the YouTube Recommendations list</li>
          <li><b>Ctrl+R</b> – Show YouTube recommendations for the currently selected song</li>
          <li><b>Ctrl+D</b> / <b>Ctrl+Enter</b> – Download selected recommendations</li>
          <li><b>Ctrl+L</b> – Focus Library search</li>
          <li><b>Ctrl+`</b> – Toggle Console visibility</li>
          <li><b>Ctrl+B</b> – Toggle Debug logging</li>
          <li><b>F1</b> / <b>Ctrl+/</b> – Show this shortcuts dialog</li>
          <li><b>F5</b> – Refresh Library</li>
        </ul>
        """)
        layout.addWidget(tb); close_btn = QPushButton("Close", dlg); close_btn.clicked.connect(dlg.accept); layout.addWidget(close_btn); dlg.resize(520,480); dlg.exec()

    def space_action_global(self):
        if self.search_list.hasFocus():
            self.load_recommendations_and_focus()
        elif self.reco_list.hasFocus():
            it = self.reco_list.currentItem();
            if it:
                it.setSelected(not it.isSelected())

    def focus_search_results(self):
        self.search_list.setFocus();
        if self.search_list.count() > 0 and self.search_list.currentRow() < 0:
            self.search_list.setCurrentRow(0)

    def focus_recommendations(self):
        self.reco_list.setFocus();
        if self.reco_list.count() > 0 and self.reco_list.currentRow() < 0:
            self.reco_list.setCurrentRow(0)

    def toggle_console_visibility(self, visible: bool):
        self.console.setVisible(visible)

    def toggle_debug(self, enabled):
        from app.utils import DEBUG_ENABLED
        DEBUG_ENABLED = enabled
        event_log(f"Debug {'enabled' if enabled else 'disabled'}")

    def clear_console(self):
        self.console.clear()
        debug_log("console", "Console cleared")

    def perform_search(self):
        if not self.ytmusic:
            event_log("YouTube Music API not available"); return
        query = self.search_box.text().strip()
        if len(query) < 3: return
        self.search_list.clear(); self.reco_list.clear()
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0,0)
        self.network_progress_active = True; self.update_progress_visibility()
        event_log(f"Searching: {query}")
        worker = SearchWorker(self.ytmusic, query)
        worker.results_ready.connect(self.handle_search_results)
        worker.error_occurred.connect(self.handle_search_error)
        worker.finished.connect(lambda: self.cleanup_worker(worker))
        worker.finished.connect(self._end_network_progress)
        self.active_workers.append(worker); worker.start()

    def _end_network_progress(self):
        self.network_progress_active = False; self.progress_bar.setRange(0,100); self.update_progress_visibility()

    def handle_search_results(self, results):
        self.search_results = results
        items = []
        for r in results:
            artists = ", ".join(a.get("name","") for a in r.get("artists", []))
            items.append(f"{r.get('title','Unknown')} — {artists}")
        self.search_list.addItems(items)
        if self.search_list.count() > 0:
            self.search_list.setCurrentRow(0); self.search_list.setFocus()
        event_log(f"Search results: {len(results)}")

    def handle_search_error(self, error):
        GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Search error: {error}")

    def load_recommendations_and_focus(self):
        if self.search_list.count() == 0: return
        if self.search_list.currentRow() < 0: self.search_list.setCurrentRow(0)
        self.load_recommendations(); self.focus_recommendations()

    def queue_recommendations_for_current(self):
        if self.search_list.currentRow() < 0 or not self.search_results: return
        self.reco_timer.stop(); self.reco_timer.start(350)

    def load_recommendations(self):
        if not self.ytmusic: return
        idx = self.search_list.currentRow()
        if idx < 0 or idx >= len(self.search_results): return
        result = self.search_results[idx]
        video_id = result.get("videoId")
        if not video_id:
            title = result.get("title") or ""
            artists = ", ".join(a.get("name","") for a in result.get("artists", []) if a.get("name"))
            query = f"{title} {artists}".strip()
            if query:
                event_log(f"No direct videoId for selection; searching for: {query}")
                self.search_and_start_recos_by_title(query)
            return
        self.start_youtube_recos(video_id)

    def start_youtube_recos(self, video_id: str):
        self.progress_bar.setVisible(True); self.progress_bar.setRange(0,0)
        self.network_progress_active = True; self.update_progress_visibility()
        event_log("Loading YouTube recommendations...")
        worker = RecommendationWorker(self.ytmusic, video_id)
        worker.recommendations_ready.connect(self.handle_recommendations)
        worker.error_occurred.connect(self.handle_recommendation_error)
        worker.finished.connect(lambda: self.cleanup_worker(worker))
        worker.finished.connect(self._end_network_progress)
        self.active_workers.append(worker); worker.start()

    def handle_recommendations(self, tracks):
        self.reco_tracks = tracks; self.reco_list.clear(); self.reco_list.addItems([t.get("title","Unknown") for t in tracks])
        if self.reco_list.count() > 0:
            self.reco_list.setCurrentRow(0); self.reco_list.setFocus()
        event_log(f"YouTube recommendations loaded: {len(tracks)}")

    def handle_recommendation_error(self, error):
        GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Recommendation error: {error}")

    def show_youtube_recommendations_for_selection(self):
        if not self.ytmusic: return
        if self.search_list.hasFocus() and self.search_list.currentRow() >= 0:
            idx = self.search_list.currentRow()
            if 0 <= idx < len(self.search_results):
                vid = self.search_results[idx].get("videoId")
                if vid:
                    self.start_youtube_recos(vid); return
        if self.reco_list.hasFocus() and self.reco_list.currentRow() >= 0:
            row = self.reco_list.currentRow()
            if 0 <= row < len(self.reco_tracks):
                vid = self.reco_tracks[row].get("videoId")
                if vid:
                    self.start_youtube_recos(vid); return
            title = self.reco_list.item(row).text(); self.search_and_start_recos_by_title(title); return
        selected_rows = self.library.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self.library_index):
                item = self.library_index[row]; audio_path = item["path"]
                meta = read_metadata(audio_path); vid = meta.get("video_id")
                if vid:
                    self.start_youtube_recos(vid); return
                title = item["title"]; artist = item.get("artist") or ""; query = f"{title} {artist}".strip(); self.search_and_start_recos_by_title(query); return
        if self.next_up_list.hasFocus() and self.next_up_list.currentRow() >= 0:
            text = self.next_up_list.item(self.next_up_list.currentRow()).text(); title = text.split(" (", 1)[0].strip()
            artist = "";
            for it in self.library_index:
                if it["title"].lower() == title.lower():
                    artist = it.get("artist") or ""; break
            query = f"{title} {artist}".strip(); self.search_and_start_recos_by_title(query)

    def search_and_start_recos_by_title(self, query: str):
        try:
            results = self.ytmusic.search(query, filter="songs", limit=1)
            if results and results[0].get("videoId"):
                self.start_youtube_recos(results[0]["videoId"])
            else:
                event_log(f"No YouTube song result for: {query}")
        except Exception as e:
            GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Search for recos failed: {e}")

    def download_selected(self):
        if self.download_in_progress > 0:
            event_log("Download already in progress; ignoring duplicate request"); return
        selected_items = self.reco_list.selectedItems();
        if not selected_items: return
        start_count = 0; self.download_btn.setEnabled(False)
        for item in selected_items:
            row = self.reco_list.row(item)
            if row < 0 or row >= len(self.reco_tracks): continue
            track = self.reco_tracks[row]
            title = track.get("title","Unknown"); video_id = track.get("videoId")
            artists_list = track.get("artists", []); artist_str = ", ".join(a.get("name","") for a in artists_list) if artists_list else ""
            if not video_id: continue
            from app.utils import sanitize
            if Path(DOWNLOAD_DIR / f"{sanitize(title)}.mp3").exists(): continue
            worker = DownloadWorker(video_id, title, artist_str)
            worker.done.connect(self.download_finished); worker.error.connect(self.download_error)
            worker.finished.connect(lambda w=worker: self.cleanup_worker(w))
            worker.finished.connect(self._on_any_download_finished)
            self.active_workers.append(worker); start_count += 1; self.download_in_progress += 1; worker.start()
        if start_count > 0:
            event_log(f"Downloading selected: {start_count} track(s)")
        else:
            self.download_btn.setEnabled(True)

    def _on_any_download_finished(self):
        self.download_in_progress = max(0, self.download_in_progress - 1)
        if self.download_in_progress == 0:
            self.download_btn.setEnabled(True)

    def download_error(self, error_msg):
        GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Download error: {error_msg}")

    def download_finished(self, path, video_id, artist):
        base_title = Path(path).stem
        self.pending_meta_by_path[path] = {"video_id": video_id, "artist": artist or "", "title": base_title}
        if not path or not Path(path).exists():
            GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Downloaded file not found"); return
        worker = AudioAnalysisWorker(path)
        worker.analysis_complete.connect(self.handle_audio_analysis)
        worker.error_occurred.connect(self.handle_analysis_error)
        worker.progress_update.connect(self.handle_analysis_progress)
        self.analysis_progress_active = True; self.update_progress_visibility()
        self.active_workers.append(worker); worker.start()

    def _enrich_artist(self, video_id: str | None, title_guess: str) -> str:
        if not self.ytmusic: return ""
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

    def handle_audio_analysis(self, original_path, bpm, camelot):
        try:
            path = Path(original_path); base = path.stem; safe_title = re.sub(r"[^a-zA-Z0-9 _-]", "", base)
            bpm_txt = str(bpm) if bpm else "UNK"
            final_name = f"{bpm_txt} {camelot} - {safe_title}.mp3"; final_path = path.parent / final_name
            if final_path.exists():
                Path(original_path).unlink(missing_ok=True); event_log(f"Already exists: {final_name}"); self.invalidate_file_cache(); return
            Path(original_path).rename(final_path); event_log(f"Done: {final_name}"); self.invalidate_file_cache()
            meta_src = self.pending_meta_by_path.pop(str(original_path), {})
            display_title = re.sub(r"\.mp3$", "", final_path.name, flags=re.IGNORECASE).split(" - ",1)[1] if " - " in final_path.name else final_path.stem
            artist_val = meta_src.get("artist", "")
            if not artist_val:
                artist_val = self._enrich_artist(meta_src.get("video_id"), display_title) or ""
            meta = {"title": display_title, "artist": artist_val, "video_id": meta_src.get("video_id", ""), "bpm": bpm, "camelot": camelot}
            write_metadata(final_path, meta)
            self.embed_id3_tags(final_path, meta)
            if self.current_bpm is None and bpm:
                self.current_bpm = bpm; self.current_key = camelot
            self.library_timer.stop(); self.library_timer.start(300)
        except Exception as e:
            GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Post-process error: {e}")
        finally:
            self.analysis_progress_active = False; self.update_progress_visibility()

    def handle_analysis_error(self, error):
        GUI_LOGGER.log.emit(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [ERROR] Analysis error: {error}")
        self.analysis_progress_active = False; self.update_progress_visibility()

    def handle_analysis_progress(self, percent: int, stage: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(max(0, min(100, percent)))
        self.progress_bar.setFormat(f"Analyzing… {percent}% – {stage}")
        self.progress_bar.setVisible(True)

    def refresh_library_delayed(self):
        self.refresh_library(); self.invalidate_file_cache()

    def invalidate_file_cache(self):
        global _file_cache, _cache_timestamp
        _file_cache = {}; _cache_timestamp = time.time()

    def refresh_library(self):
        self.library_index = []; self.bpm_index = {}; self.library.setRowCount(0)
        if not DOWNLOAD_DIR.exists(): return
        files = list(DOWNLOAD_DIR.glob("*.mp3"));
        if not files: return
        def sort_key(p: Path):
            bpm, _ = self.extract_bpm_key(p.name); return bpm if bpm is not None else 999
        files.sort(key=sort_key); self.library.setRowCount(len(files))
        for row, p in enumerate(files):
            bpm, key = self.extract_bpm_key(p.name); title = self.extract_title(p.name); meta = read_metadata(p); artist = meta.get("artist", "")
            camelot_num, camelot_mode = self.parse_camelot(key)
            item = {"path": p, "bpm": bpm, "key": key, "title": title, "artist": artist, "camelot_num": camelot_num, "camelot_mode": camelot_mode}
            self.library_index.append(item)
            if bpm is not None: self.bpm_index.setdefault(bpm, []).append(item)
            bpm_txt = str(bpm) if bpm is not None else "UNK"; key_txt = key if key is not None else "UNK"
            self.library.setItem(row, 0, QTableWidgetItem(bpm_txt)); self.library.setItem(row, 1, QTableWidgetItem(key_txt)); self.library.setItem(row, 2, QTableWidgetItem(title)); self.library.setItem(row, 3, QTableWidgetItem(artist))

    def filter_library_debounced(self):
        self.filter_timer.stop(); self.filter_timer.start(150)

    def filter_library(self):
        text = self.lib_search.text().lower()
        if not text:
            for r in range(self.library.rowCount()): self.library.setRowHidden(r, False)
            return
        for r in range(self.library.rowCount()):
            bpm_item = self.library.item(r,0); key_item = self.library.item(r,1); name_item = self.library.item(r,2); artist_item = self.library.item(r,3)
            if not bpm_item or not key_item or not name_item or not artist_item:
                self.library.setRowHidden(r, True); continue
            bpm_v = bpm_item.text().lower(); key_v = key_item.text().lower(); name_v = name_item.text().lower(); artist_v = artist_item.text().lower()
            show = (text in bpm_v) or (text in key_v) or (text in name_v) or (text in artist_v)
            self.library.setRowHidden(r, not show)

    def gather_candidates_by_bpm(self, cur_bpm):
        if cur_bpm is None or not self.bpm_index: return self.library_index[:200]
        seen = set(); candidates = []
        def add_bucket(b):
            items = self.bpm_index.get(b, [])
            for it in items:
                pid = it["path"]
                if pid not in seen:
                    seen.add(pid); candidates.append(it)
        for delta in range(-5, 6): add_bucket(cur_bpm + delta)
        half = round(cur_bpm / 2) if cur_bpm else None; double = cur_bpm * 2 if cur_bpm else None
        for base in [half, double]:
            if base and base > 0:
                for delta in range(-3, 4): add_bucket(base + delta)
        if len(candidates) < 50:
            for delta in range(-8, 9): add_bucket(cur_bpm + delta)
        return candidates

    def update_next_up_from_library(self):
        self.next_up_list.clear()
        selected_rows = self.library.selectionModel().selectedRows()
        if not selected_rows: return
        row = selected_rows[0].row()
        if row < 0 or row >= len(self.library_index): return
        cur = self.library_index[row]
        cur_bpm = cur["bpm"]; cur_key = cur["key"]; cur_title = cur["title"]
        candidates = self.gather_candidates_by_bpm(cur_bpm)
        def mix_score(cur_bpm, cur_key, bpm, key):
            def parse_camelot(c):
                if not c or c == "UNK": return None, None
                try:
                    num = int(c[:-1]); mode = c[-1]; return num, mode
                except Exception:
                    return None, None
            def key_proximity_score(a, b):
                if not a or not b or "UNK" in (a,b): return 0
                n1,m1 = parse_camelot(a); n2,m2 = parse_camelot(b)
                if n1 is None or n2 is None: return 0
                if n1 == n2 and m1 == m2: return 100
                if n1 == n2 and m1 != m2: return 85
                dist = abs(n1 - n2); dist = min(dist, 12 - dist)
                if m1 == m2 and dist == 1: return 90
                if m1 == m2 and dist == 2: return 65
                if m1 == m2 and dist == 3: return 45
                return 0
            def bpm_tier_score(a, b):
                if not a or not b: return 0
                diff = abs(a - b)
                if diff <= 2: return 100
                if diff <= 5: return 85
                if abs(a - b * 2) <= 3 or abs(a * 2 - b) <= 3: return 70
                if diff <= 8: return 50
                return 0
            return int(round(0.6 * bpm_tier_score(cur_bpm, bpm) + 0.4 * key_proximity_score(cur_key, key)))
        scored = []
        for item in candidates:
            if item["path"] == cur["path"]: continue
            total = mix_score(cur_bpm, cur_key, item["bpm"], item["key"])
            if total > 0: scored.append((total, item))
        scored.sort(key=lambda x: (x[0], x[1]["title"]), reverse=True)
        best = scored[:10]
        for total, item in best:
            bpm_txt = str(item["bpm"]) if item["bpm"] is not None else "UNK"; key_txt = item["key"] if item["key"] is not None else "UNK"
            self.next_up_list.addItem(f"{total}% - {item['title']} ({bpm_txt} BPM, {key_txt})")
        event_log(f"Next up (top {len(best)}) for: {cur_title}")

    def cleanup_worker(self, worker: QThread):
        try:
            self.active_workers.remove(worker)
        except ValueError:
            pass
        worker.deleteLater()

    def closeEvent(self, event):
        for worker in self.active_workers[:]:
            if worker.isRunning():
                worker.quit(); worker.wait(2000)
        event.accept()

    def update_progress_visibility(self):
        visible = self.network_progress_active or self.analysis_progress_active
        self.progress_bar.setVisible(visible)
        if not visible:
            self.progress_bar.setFormat("%p%")
            self.progress_bar.setValue(0)

    def extract_bpm_key(self, filename):
        try:
            base_name = Path(filename).name
            if " - " not in base_name: return None, None
            head = base_name.split(" - ", 1)[0]; parts = head.split()
            if len(parts) < 2 or not parts[0].isdigit(): return None, None
            key_part = parts[1]
            if len(key_part) < 2 or not key_part[:-1].isdigit() or key_part[-1] not in "AB": return None, None
            return int(parts[0]), key_part
        except Exception:
            return None, None

    def extract_title(self, filename):
        name = Path(filename).name
        title = name.split(" - ", 1)[1] if " - " in name else name
        return re.sub(r"\.mp3$", "", title, flags=re.IGNORECASE)

    def parse_camelot(self, c):
        if not c or c == "UNK": return None, None
        try:
            num = int(c[:-1]); mode = c[-1]; return num, mode
        except Exception:
            return None, None

    def embed_id3_tags(self, audio_path: Path, meta: dict):
        if not ID3_AVAILABLE: return
        try:
            tags = EasyID3(str(audio_path))
        except Exception:
            try:
                from mutagen.id3 import ID3, ID3NoHeaderError  # type: ignore
                try:
                    ID3(str(audio_path))
                except ID3NoHeaderError:
                    from mutagen.id3 import ID3 as _ID3  # type: ignore
                    _id3 = _ID3(); _id3.save(str(audio_path))
                tags = EasyID3(str(audio_path))
            except Exception:
                return
        try:
            if meta.get("title"): tags["title"] = [meta["title"]]
            if meta.get("artist"): tags["artist"] = [meta["artist"]]
            if meta.get("bpm"): tags["bpm"] = [str(meta["bpm"])]
            if meta.get("camelot"): tags["initialkey"] = [meta["camelot"]]
            tags.save()
        except Exception:
            pass

__all__ = ["MainWindow"]
