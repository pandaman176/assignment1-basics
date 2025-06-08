from cs336_basics import bpe, logger_config, json_saver
from pathlib import Path
import os
import time

logger = logger_config.setup_logger(__name__)

def main():
    # 当前文件的目录
    CURRENT_DIR = Path(__file__).parent
    special_tokens = ["<|endoftext|>"]
    file_path = CURRENT_DIR.parent / "data/TinyStoriesV2-GPT4-valid.txt"
    DUMP_PATH = CURRENT_DIR.parent / "models"
    logger.info("start training bpe on tinystories")
    start_time = time.time()
    vocab, merges = bpe.train_bpe(
        file_path,
        500,
        special_tokens,
        verbose=True,
    )
    time_cost = time.time() - start_time
    logger.info(f"finish training bpe on tinystories, {time_cost=:.2f}s")
    logger.info(f"dump file ...")
    json_saver.dump_vocab(vocab, DUMP_PATH / "tinystories_vocab.json")
    json_saver.dump_merges(merges, DUMP_PATH / "tinystories_merges.json")

    logger.info(f"vocab size {len(vocab)}")
    logger.info(f"merges size {len(merges)}")

if __name__ == "__main__":
    main()