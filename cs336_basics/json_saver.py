import json
import os
import ast


def dump_vocab(vocab: dict[int, bytes], file_path: str | os.PathLike, indent: int = 2):
    vocab_json = {k: v.__repr__() for k, v in vocab.items()}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, indent=2)


def dump_merges(merges: list[tuple[bytes, bytes]], file_path: str | os.PathLike, indent: int = 2):
    merges_json = [(a.__repr__(), b.__repr__()) for a, b in merges]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(merges_json, f, indent=2)


def load_vocab(file_path: str | os.PathLike) -> dict[int, bytes]:
    with open(file_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    return {int(k): ast.literal_eval(v) for k, v in vocab_json.items()}


def load_merges(file_path: str | os.PathLike) -> list[tuple[bytes, bytes]]:
    with open(file_path, "r", encoding="utf-8") as f:
        merges_json = json.load(f)
    return [(ast.literal_eval(a), ast.literal_eval(b)) for a, b in merges_json]
