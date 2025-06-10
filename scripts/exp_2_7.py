from cs336_basics import logger_config, bpe_tokenizer, useful_path
import os
import time
import numpy as np


logger = logger_config.setup_logger(__name__)

def exp_on_ts():
    DUMP_PATH = useful_path.DUMP_PATH
    DATA_PATH = useful_path.DATA_DIR

    special_tokens = ["<|endoftext|>"]
    ts_tokenizer = bpe_tokenizer.BPETokenizer.from_files(
        vocab_filepath=DUMP_PATH / "tinystories_valid_vocab.json",
        merges_filepath=DUMP_PATH / "tinystories_valid_merges.json",
        special_tokens=special_tokens,
    )
    with open(DATA_PATH / "TinyStoriesV2-GPT4-sample.txt", "r", encoding="utf-8") as f:
        sample_text = f.read()
    token_ids = ts_tokenizer.encode(sample_text)
    num_bytes = len(sample_text.encode("utf-8"))
    num_tokens = len(token_ids)
    # 避免除以 0
    if num_tokens == 0:
        return float("inf")
    compress_ratio = num_bytes / num_tokens
    logger.info(f"tiny story tokenizer compress ratio on tinystories: {compress_ratio:.2f}")
    with open(DATA_PATH / "owt_sample.txt", "r", encoding="utf-8") as f:
        sample_text = f.read()
    token_ids = ts_tokenizer.encode(sample_text)
    num_bytes = len(sample_text.encode("utf-8"))
    num_tokens = len(token_ids)
    # 避免除以 0
    if num_tokens == 0:
        return float("inf")
    compress_ratio = num_bytes / num_tokens
    logger.info(f"tiny story tokenizer compress ratio on owt: {compress_ratio:.2f}")

    with open(DATA_PATH / "TinyStoriesV2-GPT4-sample.txt", "r", encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)
        f_size = f.tell()
        f.seek(0)
        ids = []
        start_time = time.time()
        for _id in ts_tokenizer.encode_iterable(f):
            ids.append(_id)
        time_cost = time.time() - start_time
        throughput = f_size / time_cost
        logger.info(f"throughput : {throughput:.2f}bytes/s")
    
def train_onts():
    DUMP_PATH = useful_path.DUMP_PATH
    DATA_PATH = useful_path.DATA_DIR

    special_tokens = ["<|endoftext|>"]
    ts_tokenizer = bpe_tokenizer.BPETokenizer.from_files(
        vocab_filepath=DUMP_PATH / "tinystories_valid_vocab.json",
        merges_filepath=DUMP_PATH / "tinystories_valid_merges.json",
        special_tokens=special_tokens,
    )

    logger.info("encode TinyStoriesV2-GPT4-train.txt")
    with open(DATA_PATH / "TinyStoriesV2-GPT4-sample.txt", "r", encoding="utf-8") as f:
        ids = []
        for _id in ts_tokenizer.encode_iterable(f):
            ids.append(_id)
        arr = np.array(ids, dtype=np.uint16)
        np.save(DUMP_PATH / "tinystories_train_ids.npy", arr)
        logger.info(f"save tinystories_train_ids.npy")

def main():
    exp_on_ts()
    train_onts()


if __name__ == "__main__":
    main()
