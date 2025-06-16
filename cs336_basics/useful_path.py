import socket
from pathlib import Path

WORK_DIR = Path(__file__).parent.parent
DATA_DIR = WORK_DIR / "data" if socket.gethostname() == "TABLET-WEN" else Path("/data/tuoge")
MODEL_DIR = DATA_DIR / "models"
