"""
logger.py — writes every decision to both the console (INFO+) and a file (DEBUG+).

WHY: When you're debugging "why did query #47 go to remote instead of local?",
you want a permanent record, not a print() that scrolled off your terminal.
"""

import logging
import os
import sys
from src.config import CONFIG

os.makedirs("logs", exist_ok=True)

# Windows consoles default to cp1252, which can't encode the emoji used in
# log messages (❌, ✅, 💸, ...) and raises UnicodeEncodeError mid-log.
# Reconfigure stdout/stderr to UTF-8 so those log lines don't crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured (avoids duplicate log lines if called twice)
        return logger

    logger.setLevel(getattr(logging, CONFIG.LOG_LEVEL))

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    ))

    file_handler = logging.FileHandler(CONFIG.LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
