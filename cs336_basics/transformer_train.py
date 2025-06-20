import torch
from typing import Tuple
import typing
import numpy as np
import os

def get_batch(
    x: np.ndarray, 
    batch_size: int, 
    context_length: int, 
    device: torch.device  
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 选择起始索引，确保不会越界
    max_start = x.shape[0] - context_length 
    start_indices = np.random.randint(0, max_start, size=batch_size)

    # 构造 input 和 target 序列
    inputs = torch.stack([
        torch.tensor(x[i : i + context_length], dtype=torch.long)
        for i in start_indices
    ])

    targets = torch.stack([
        torch.tensor(x[i + 1 : i + 1 + context_length], dtype=torch.long)
        for i in start_indices
    ])

    return inputs.to(device), targets.to(device)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> int:
    # 构造一个字典，包含模型状态、优化器状态以及当前迭代数
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    # 使用torch.save将字典保存到文件路径或类文件对象out中
    torch.save(checkpoint, out)
    return iteration

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    
    checkpoint = torch.load(src, map_location='cpu')  # 你也可以用其他设备
    model.load_state_dict(checkpoint["model_state_dict"]) if model is not None else None
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) if optimizer is not None else None
    iteration = checkpoint["iteration"]
    return iteration
