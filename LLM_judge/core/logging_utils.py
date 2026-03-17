import sys


def log_stage(message: str) -> None:
    """Print a pipeline stage message, safely handling narrow console encodings (e.g. Windows cp1252)."""
    try:
        print(message)
    except (UnicodeEncodeError, UnicodeDecodeError):
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        safe = str(message).encode(encoding, errors="replace").decode(encoding, errors="replace")
        try:
            print(safe)
        except Exception:
            pass  # last resort: silently skip unprintable log line
