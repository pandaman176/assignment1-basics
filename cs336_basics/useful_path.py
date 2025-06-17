import socket
from pathlib import Path

WORK_DIR = Path(__file__).parent.parent
DATA_DIR = WORK_DIR / "data" if socket.gethostname() == "TABLET-WEN" else Path("/root/autodl-fs")
MODEL_DIR = DATA_DIR / "models" if socket.gethostname() == "TABLET-WEN" else Path("/root/autodl-tmp/models")
