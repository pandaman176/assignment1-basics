import socket
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" if socket.gethostname() == "TABLET-WEN" else Path("/data/tuoge")
DUMP_PATH = Path(__file__).parent.parent / "models"
SMALL_DATA_DIR = Path(__file__).parent.parent / "data" if socket.gethostname() == "TABLET-WEN" else Path(__file__).parent.parent / "small_data"
