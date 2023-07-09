"""Top level init."""

from pathlib import Path

from dotenv import dotenv_values

config = dotenv_values(".env")

TRAINING_DIR = Path(__file__).parent.parent.parent / "training"
TRAINING_DIR.mkdir(exist_ok=True)
