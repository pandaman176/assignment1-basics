import numpy as np
import torch
from einops import repeat

from cs336_basics import MyModules, transformer_train, bpe_tokenizer
from cs336_basics.useful_path import DATA_DIR, MODEL_DIR

def test():
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 数据路径
    data_paths = {
        "vocab_filepath": DATA_DIR / "tinystories_vocab.json",
        "merges_filepath": DATA_DIR / "tinystories_merges.json",
        "special_tokens": ["<|endoftext|>"],
        "final_model_path": FINAL_MODEL_PATH,  # 最终模型保存路径
        "training_dataset_path": TRAIN_DATA_PATH,
        "validation_dataset_path": VAL_DATA_PATH,  # 验证集路径
    }
    
    tokenizer = bpe_tokenizer.BPETokenizer.from_files(
        vocab_filepath=data_paths["vocab_filepath"],
        merges_filepath=data_paths["merges_filepath"],
        special_tokens=data_paths["special_tokens"],
    )
    training_dataset = np.load(data_paths['training_dataset_path'], mmap_mode='r+') # 使用内存映射
    validation_dataset = np.load(data_paths['validation_dataset_path'], mmap_mode='r+')
    print(training_dataset.shape)
    print(validation_dataset.shape)
    train_head = tokenizer.decode(training_dataset[0:100])
    val_head = tokenizer.decode(validation_dataset[0:100])
    #print(train_head)
    print(val_head)
    print("===========")
    print(validation_dataset[0:100])


if __name__ == "__main__":
    FINAL_MODEL_PATH = MODEL_DIR / "finals/final_model_v1.pt"
    TRAIN_DATA_PATH = DATA_DIR / "tinystories_train_ids.npy"
    VAL_DATA_PATH = DATA_DIR / "tinystories_v2_sample_ids.npy"  # 验证集路径
    test()
