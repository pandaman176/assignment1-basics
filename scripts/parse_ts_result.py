from cs336_basics import json_saver
import json
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
DUMP_PATH = CURRENT_DIR.parent / "models"
vocab: dict[int, bytes] = json_saver.load_vocab(DUMP_PATH / "tinystories_vocab.json")
merges: list[tuple[bytes, bytes]] = json_saver.load_merges(DUMP_PATH / "tinystories_merges.json")

# find longest token in vocab
longest_token = max(vocab.keys(), key=lambda x: len(vocab[x]))
print(f"longest token: {vocab[longest_token]}")
