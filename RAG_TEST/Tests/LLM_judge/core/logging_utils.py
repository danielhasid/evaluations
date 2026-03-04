import sys


def log_stage(message: str) -> None:
    """Print a pipeline stage message, safely handling narrow console encodings (e.g. Windows cp1252)."""
    try:
        print(message)
    except UnicodeEncodeError:
        safe = message.encode(sys.stdout.encoding or "ascii", errors="replace").decode(sys.stdout.encoding or "ascii")
        print(safe)
