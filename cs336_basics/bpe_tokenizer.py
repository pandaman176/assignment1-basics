import os
import regex as re
from typing import Iterable, Iterator, Self
from .json_saver import load_vocab, load_merges


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """Initialize a Tokenizer with a vocabulary and a list of merges."""
        self.vocab = vocab  # {(0, b'\x00'), (1, b'\x01'), (2, b'\x02')}
        self.merges = merges  # [(b' ', b't'), (b'h', b'e'), (b' ', b'a'), (b' ', b's')]
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Self:
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        if not text:
            return []
        byte2int = {v: k for k, v in self.vocab.items()}
        if not self.special_tokens:
            # when special_tokens is None, pattern will match everything, so
            # we need to explicitly consider no special token case
            split_texts: list[str] = [text]
        else:
            # sort special_tokens by length, in case there is overlapping
            # e.g. match <|endoftext|><|endoftext|> first and then <|endoftext|>
            pattern = "(" + "|".join(map(re.escape, sorted(self.special_tokens, key=len, reverse=True))) + ")"
            split_texts: list[str] = re.splititer(pattern, text)
        result = []
        for split_text in split_texts:
            # print(f"{split_text=}")
            if self.special_tokens and split_text in self.special_tokens:
                b_special_token: bytes = split_text.encode("utf-8")
                result.append(byte2int[b_special_token])
                continue
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            pre_token_matchs = re.finditer(PAT, split_text)
            for match in pre_token_matchs:
                pre_token = match.group()
                b_pre_token: bytes = pre_token.encode("utf-8")
                # print(f"    {b_pre_token=}")
                bytes_list: list[bytes] = [bytes([byte]) for byte in b_pre_token]
                # print(f"    {bytes_list=}")
                for merge in self.merges:
                    # print(f"    find {merge=}")
                    i = 0
                    while i + 1 < len(bytes_list):
                        if (bytes_list[i], bytes_list[i + 1]) == merge:
                            # print(f"\n        merge {(bytes_list[i], bytes_list[i+1])}")
                            bytes_list[i : i + 2] = [bytes_list[i] + bytes_list[i + 1]]
                        i += 1
                for token in bytes_list:
                    result.append(byte2int[token])
        # print(result)
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory.
        """
        for text in iterable:
            ids = self.encode(text)
            for id_ in ids:
                yield id_

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens = [self.vocab[token_id] for token_id in ids]
        decoded_bytes = b"".join(tokens)
        try:
            decoded_str = decoded_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # fallback: replace invalid bytes
            decoded_str = decoded_bytes.decode("utf-8", errors="replace")
        return decoded_str
