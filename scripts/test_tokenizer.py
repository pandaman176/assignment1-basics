from cs336_basics import bpe, logger_config, json_saver, bpe_tokenizer
from pathlib import Path
import os
import time
import socket

logger = logger_config.setup_logger(__name__)


def main():
    # 当前文件的目录
    CURRENT_DIR = Path(__file__).parent
    special_tokens = ["<|endoftext|>"]
    logger.info(f"hostname: {socket.gethostname()}")
    file_path = (
        "./data/TinyStoriesV2-GPT4-valid.txt"
        if socket.gethostname() == "TABLET-WEN"
        else "/data/tuoge/TinyStoriesV2-GPT4-train.txt"
    )
    DUMP_PATH = CURRENT_DIR.parent / "models"
    # check if dump file exists
    if (
        not (DUMP_PATH / "tinystories_valid_vocab.json").exists()
        and not (DUMP_PATH / "tinystories_valid_merges.json").exists()
    ):
        logger.info("start training bpe on tinystories")
        start_time = time.time()
        vocab, merges = bpe.train_bpe(
            file_path,
            300 if socket.gethostname() == "TABLET-WEN" else 10_000,
            special_tokens,
            verbose=True,
            use_heap=True,
        )
        time_cost = time.time() - start_time
        logger.info(f"finish training bpe on tinystories, {time_cost=:.2f}s")
        logger.info(f"dump file ...")
        json_saver.dump_vocab(vocab, DUMP_PATH / "tinystories_valid_vocab.json")
        json_saver.dump_merges(merges, DUMP_PATH / "tinystories_valid_merges.json")

        logger.info(f"vocab size {len(vocab)}")
        logger.info(f"merges size {len(merges)}")

    logger.info("create tokenizer")
    tokenizer = bpe_tokenizer.BPETokenizer.from_files(
        vocab_filepath=DUMP_PATH / "tinystories_valid_vocab.json",
        merges_filepath=DUMP_PATH / "tinystories_valid_merges.json",
        special_tokens=special_tokens,
    )
    example_text = "a"
    example_vocab = {
        0: b" ",
        1: b"a",
        2: b"c",
        3: b"e",
        4: b"h",
        5: b"t",
        6: b"th",
        7: b" c",
        8: b" a",
        9: b"the",
        10: b" at",
        11: b"<|endoftext|>",
    }
    example_merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    example_tokenizer = bpe_tokenizer.BPETokenizer(example_vocab, example_merges, special_tokens)
    encoded_text = example_tokenizer.encode(example_text)
    print(encoded_text)
    decoded_text = example_tokenizer.decode(encoded_text)
    print(decoded_text)


if __name__ == "__main__":
    main()
