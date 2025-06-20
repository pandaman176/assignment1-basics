
import torch
from einops import rearrange, repeat
from loguru import logger

from cs336_basics import MyModules, transformer_train, bpe_tokenizer
from cs336_basics.useful_path import WORK_DIR, DATA_DIR, MODEL_DIR

def test():
    # 模型参数
    model_config = {
        "vocab_size": 10000,      # 词汇表大小
        "context_length": 256,    # 上下文长度
        "num_layers": 4,          # Transformer Block数
        "num_heads": 16,          # 注意力头数
        "d_model": 512,           # 嵌入空间维度
        "d_ff": 1344,             # 前馈网络维度
        "rope_theta": 10000,      # RoPE参数
    }
    
    # 优化器参数
    optim_config = {
        "lr": 3e-4,               # 学习率
        "weight_decay": 1e-2,     # 权重衰减
        "betas": (0.9, 0.999),    # AdamW的beta参数
        "max_norm": 1.0,          # 梯度裁剪的最大范数
    }
    
    # 训练参数
    train_config = {
        "batch_size": 16,         # 批次大小
        "total_epochs": 0.5,      # 训练轮数
        "checkpoint_freq": 2000,  # 每隔多少步保存一次检查点
        "log_freq": 10,           # 每隔多少步记录一次日志
        "val_freq": 400,          # 每隔多少步在验证集上评估
        "val_batch_size": 16,     # 验证时的批次大小
        "val_batches": 20,        # 验证时使用的批次数量
    }
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化模型
    model = MyModules.transformer_lm(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_model=model_config["d_model"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
        device=device,
    )
    # 数据路径
    data_paths = {
        "vocab_filepath": DATA_DIR / "tinystories_vocab.json",
        "merges_filepath": DATA_DIR / "tinystories_merges.json",
        "special_tokens": ["<|endoftext|>"],
        "final_model_path": FINAL_MODEL_PATH,  # 最终模型保存路径
    }
    
    transformer_train.load_checkpoint(
        data_paths["final_model_path"],
        model=model,
        optimizer=None,
    )
    prompts = [
        "he quick brown fox jumps over the lazy dog",
        "Once upon a time,",
        "Tom and Lily are best friends.",
    ]
    tokenizer = bpe_tokenizer.BPETokenizer.from_files(
        vocab_filepath=data_paths["vocab_filepath"],
        merges_filepath=data_paths["merges_filepath"],
        special_tokens=data_paths["special_tokens"],
    )
    temperature = 1.0
    top_p = 0.9
    outputs = generate(
        model=model,
        prompts=prompts,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=temperature,
        top_p=top_p,
        context_length=model_config["context_length"],
        end_token="<|endoftext|>",
        device=device,
    )
    print("=====output======")
    for i, output in enumerate(outputs):
        print(f"prompt: {prompts[i]}")
        print(f"output: {output}")

def top_p_sample(probs, top_p):
    # Top-p sampling
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs <= top_p        # True 的位置表示要屏蔽
    sorted_mask[..., 0] = True                   # 保底，防止第一个 token 被屏蔽


    sorted_probs_fixed = sorted_probs * sorted_mask
    sorted_probs_normalized = sorted_probs_fixed / sorted_probs_fixed.sum(dim=-1, keepdim=True)

    probs_filtered = torch.zeros_like(probs)
    # 使用scatter操作还原原始顺序
    probs_filtered.scatter_(1, sorted_indices, sorted_probs_normalized)

    return probs_filtered

@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompts: list[str],
    tokenizer,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    context_length: int = 256,
    end_token: str = None,
    device: torch.device | None = None,
) -> list[str]:
    print(f"{temperature=}, {top_p=}")
    model.eval()
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # form input tensor
    inputs_ids = []
    len_input_ids = []
    for prompt in prompts:
        # encode the prompt
        input_ids = torch.tensor(tokenizer.encode(prompt),dtype=torch.int32).to(device)
        inputs_ids.append(input_ids)
        len_input_ids.append(len(input_ids))
    
    len_input_ids = torch.tensor(len_input_ids,dtype=torch.int64).to(device)
    # (batch_size,)

    # 如果batch中的序列长度不一致，，则无法形成矩阵，，需要padding
    pad_token_id = tokenizer.encode(end_token)[0]
    padded_inputs = torch.full(
        (len(inputs_ids), context_length),
        fill_value=pad_token_id,
        dtype=torch.int32,
        device=device,
    )
    for i, input_ids in enumerate(inputs_ids):
        padded_inputs[i, :len_input_ids[i]] = input_ids
    # (batch_size, context_length)

    end_token_id = tokenizer.encode(end_token)[0]
    is_end = torch.zeros(padded_inputs.shape[0], dtype=torch.bool, device=device)
    # 记录每个prompt是否结束

    for num in range(max_new_tokens):
        logits = model(padded_inputs)  # 假设输出 shape 为 (batch_size, seq_len, vocab_size)
        index = len_input_ids - 1 + num
        index = repeat(index, 'b -> b 1 v', v=logits.shape[-1])
        # 取最后一个位置的 logits
        logits = torch.gather(logits, dim=1, index=index).squeeze(1)
        # (batch_size, vocab_size)

        logits = logits / temperature
        probs = MyModules.softmax(logits, dim=-1)

        probs = top_p_sample(probs, top_p=top_p)

        next_token_ids = torch.multinomial(probs, num_samples=1).to(dtype=torch.int32)
        next_token_index = (len_input_ids + num).unsqueeze(1)
        padded_inputs.scatter_(1, next_token_index, next_token_ids)
        # 原地修改

        is_end = is_end | (next_token_ids == end_token_id)

        if is_end.all():
            break

    outputs = []
    for i in range(padded_inputs.shape[0]):
        output_ids = padded_inputs[i, :].cpu().numpy()
        output_text = tokenizer.decode(output_ids, end_token_id=end_token_id)
        outputs.append(output_text)

    return outputs

if __name__ == "__main__":
    FINAL_MODEL_PATH = MODEL_DIR / "finals/final_model_v1.pt"
    test()
