import os
import multiprocessing
import regex as re
from typing import BinaryIO
from pathlib import Path
from collections import Counter


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize(
    file_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
) -> dict[tuple, int]:
    """
    Given a chunk of text, return a dictionary of pre-tokens and their counts.
    """
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # remove special tokens
        raw_texts = re.splititer("|".join(map(re.escape, special_tokens)), chunk)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        freq_dict = Counter()
        for raw_text in raw_texts:
            for match in re.finditer(PAT, raw_text):  # use finditer to save memory
                word = match.group()
                freq_dict[word] += 1
        freq_dict = {tuple(k.encode("utf-8")): v for k, v in freq_dict.items()}
        return freq_dict


def count_pairs(chunk):
    local_freq = Counter()
    for k, v in chunk:
        for pair in zip(k[:-1], k[1:]):
            local_freq[pair] += v
    return local_freq


def paral_count_pairs(
    freq_dict: dict[tuple, int],
    num_processes: int = 4,
) -> Counter:
    items = list(freq_dict.items())
    chunk_size = (len(items) + num_processes - 1) // num_processes
    chunks = [items[i * chunk_size : (i + 1) * chunk_size] for i in range(num_processes)]

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(count_pairs, chunks)
    freq_of_pairs = Counter()
    for result in results:
        freq_of_pairs.update(result)
    return freq_of_pairs


def train_bpe(
    file_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the string, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        file_path (str | os.PathLike): The data to train a BPE tokenizer on.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    num_processes = 4
    num_special_tokens = len(special_tokens)
    num_non_special_tokens = 256
    rounds = vocab_size - num_special_tokens - num_non_special_tokens

    vocab = {i: bytes([i]) for i in range(num_non_special_tokens)}
    merge: list[tuple[bytes, bytes]] = []
    new_token = num_non_special_tokens
    # -------------parallelize this part
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        tasks = [(file_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        # 使用进程池并行地处理每块
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(pre_tokenize, tasks)

        freq_dict = Counter()
        for sub_freq_dict in results:
            freq_dict.update(sub_freq_dict)
    # -------------end parallelize
    freq_of_pairs = Counter()
    # -------------parallelize this part didn't improve much
    # freq_of_pairs = paral_count_pairs(freq_dict, num_processes)
    for k, v in freq_dict.items():
        # assert type(k) == tuple  # (bytes, bytes, bytes, ...)
        for pair in zip(k[:-1], k[1:]):
            freq_of_pairs[pair] = freq_of_pairs.get(pair, 0) + v
    # -------------end of find couting freq
    for round in range(rounds):
        
        max_freq = max(freq_of_pairs.values())
        candidates = [pair for pair, freq in freq_of_pairs.items() if freq == max_freq]
        #candidates = [(111, 207), (109, 203)...]
        max_freq_pair = max(candidates, key=lambda x: (vocab[x[0]], vocab[x[1]])) # choose lexicographically greater pair if tied
        vocab[new_token] = vocab[max_freq_pair[0]] + vocab[max_freq_pair[1]]
        merge.append((vocab[max_freq_pair[0]], vocab[max_freq_pair[1]]))
        new_freq_dict = {}
        freq_of_pairs.pop(max_freq_pair)
        for k, v in freq_dict.items():
            new_k = []
            i = 0
            while i < len(k):
                if i < len(k) - 1 and (k[i], k[i + 1]) == max_freq_pair:
                    # 增量更新freq_of_pairs
                    if i > 0:
                        freq_of_pairs[(k[i - 1], k[i])] -= v
                        freq_of_pairs[(k[i - 1], new_token)] += v
                    if i < len(k) - 2:
                        freq_of_pairs[(k[i + 1], k[i + 2])] -= v
                        freq_of_pairs[(new_token, k[i + 2])] += v
                    new_k.append(new_token)
                    i += 2
                else:
                    new_k.append(k[i])
                    i += 1
            new_freq_dict[tuple(new_k)] = v
        new_token += 1
        freq_dict = new_freq_dict

    for special_token in special_tokens:
        vocab[new_token] = special_token.encode("utf-8")
        new_token += 1

    return (vocab, merge)


def main():
    special_tokens = ["<|endoftext|>"]
    num_special_tokens = len(special_tokens)
    initial_num_tokens = 256
    #file_path = Path("./data/TinyStoriesV2-GPT4-valid.txt").resolve()
    file_path = "tests/fixtures/corpus.en"
    vocab, merges = train_bpe(
        file_path, 500, special_tokens,
    )
    print(vocab)
    print(merges[90:100])
    print(type(vocab[1]))


if __name__ == "__main__":
    # import cProfile
    # cProfile.run("main()", filename="profile.out")
    main()
