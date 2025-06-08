from cs336_basics import bpe, logger_config, json_saver
from pathlib import Path
import os
import time

logger = logger_config.setup_logger(__name__)

def main():
    # 当前文件的目录
    CURRENT_DIR = Path(__file__).parent
    special_tokens = ["<|endoftext|>"]
    file_path = "/data/tuoge/TinyStoriesV2-GPT4-valid.txt"
    DUMP_PATH = CURRENT_DIR.parent / "models"
    logger.info("start training bpe on tinystories")
    start_time = time.time()
    vocab, merges = bpe.train_bpe(
        file_path,
        10_000,
        special_tokens,
        verbose=True,
        use_heap=False
    )
    time_cost = time.time() - start_time
    start_time = time.time()
    logger.info(f"finish training bpe, {time_cost=:.2f}s")
    vocab_2, merges_2 = bpe.train_bpe(
        file_path,
        10_000,
        special_tokens,
        verbose=True,
        use_heap=True
    )
    time_cost = time.time() - start_time
    logger.info(f"finish training bpe use heap, {time_cost=:.2f}s")
    assert len(vocab) == len(vocab_2)
    assert len(merges) == len(merges_2)

if __name__ == "__main__":
    main()
