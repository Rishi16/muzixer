import sys
from pathlib import Path

# Make sure project root is on sys.path when run as a script (python app\run.py)
if __package__ is None and __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

# Import MainWindow from the UI module
from app.ui_main import MainWindow  # type: ignore


def main():
    # Ensure yt-dlp is available
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        print("Error: yt-dlp not installed. Install with: pip install yt-dlp")
        sys.exit(1)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
