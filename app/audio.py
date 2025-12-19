import re
from pathlib import Path
from typing import Tuple, Optional

import librosa
from librosa import feature as _librosa_feature

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

KEY_MAP = {
    "C": ("8B", "5A"), "C#": ("3B", "12A"), "D": ("10B", "7A"),
    "D#": ("5B", "2A"), "E": ("12B", "9A"), "F": ("7B", "4A"),
    "F#": ("2B", "11A"), "G": ("9B", "6A"), "G#": ("4B", "1A"),
    "A": ("11B", "8A"), "A#": ("6B", "3A"), "B": ("1B", "10A")
}


def _rotate(lst, n):
    n %= len(lst)
    return lst[n:] + lst[:n]


def infer_key_mode_from_chroma(chroma) -> Tuple[Optional[str], Optional[str]]:
    import numpy as _np
    if chroma is None or chroma.size == 0:
        return None, None
    chroma_mean = chroma.mean(axis=1)
    if chroma_mean.max() > 0:
        chroma_mean = chroma_mean / (chroma_mean.max() + 1e-9)
    best_score = -1.0
    best_key = None
    best_mode = None
    for i in range(12):
        prof_maj = _np.array(_rotate(_KS_PROFILE_MAJOR, i))
        prof_min = _np.array(_rotate(_KS_PROFILE_MINOR, i))
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
    import numpy as _np
    if y is None or len(y) == 0:
        return y
    seg_len = int(target_seconds * sr)
    if len(y) <= seg_len:
        return y
    step = int(5 * sr)
    best_start = 0
    best_energy = -1.0
    for start in range(0, max(1, len(y) - seg_len + 1), step):
        end = start + seg_len
        window = y[start:end]
        e = float(_np.sum(window.astype(_np.float32) ** 2))
        if e > best_energy:
            best_energy = e
            best_start = start
    return y[best_start: best_start + seg_len]


def key_to_camelot(key: Optional[str], mode: str = "minor") -> Optional[str]:
    if not key:
        return None
    major, minor = KEY_MAP.get(key, (None, None))
    if mode not in ("major", "minor"):
        mode = "minor"
    return minor if mode == "minor" else major


def analyze_audio_batch(path: str):
    y_full, sr = librosa.load(path, mono=True, duration=90, sr=22050)
    y = pick_informative_segment(y_full, sr, target_seconds=30)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=1024)
    bpm = int(round(float(tempo))) if tempo is not None else None
    chroma = _librosa_feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
    key_name, mode = infer_key_mode_from_chroma(chroma)
    return bpm, key_name, mode

