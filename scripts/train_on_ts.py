import torch
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
import wandb
import os
from tqdm import tqdm
from einops import rearrange
from loguru import logger
import argparse

from cs336_basics import MyModules, transformer_train, bpe_tokenizer
from cs336_basics.useful_path import WORK_DIR, DATA_DIR, MODEL_DIR

def train():
    # 设置所有超参数
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
        "batch_size": 32,         # 批次大小
        "total_tokens": 327_680_000,  # 训练总token数
        "checkpoint_freq": 2000,  # 每隔多少步保存一次检查点
        "log_freq": 1,           # 每隔多少步记录一次日志
        "val_freq": 400,          # 每隔多少步在验证集上评估
        "val_batch_size": 16,     # 验证时的批次大小
        "val_batches": 20,        # 验证时使用的批次数量
    }
    
    # 数据路径
    data_paths = {
        "training_dataset_path": TRAIN_DATA_PATH,
        "validation_dataset_path": VAL_DATA_PATH,  # 验证集路径
        "checkpoint_load_path": CHECKPOINT_LOAD_PATH,  # 模型检查点路径
        "checkpoint_save_format": CHECKPOINT_SAVE_FORMAT,  # 检查点保存路径格式
        "final_model_path": FINAL_MODEL_PATH,  # 最终模型保存路径
    }
    
    # 初始化wandb
    run = wandb.init(
        project="cs336-assignment-1",
        name="train_v1",
        config={
            "model": model_config,
            "optimizer": optim_config,
            "training": train_config,
        }
    )
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    
    # 初始化模型
    logger.info("开始初始化模型...")
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
    model = model.to(device)
    logger.info("模型初始化完成。")

    # 初始化优化器
    logger.info("开始初始化优化器...")
    optimizer = MyModules.AdamW(
        model.parameters(),
        lr=optim_config["lr"],
        weight_decay=optim_config["weight_decay"],
        betas=optim_config["betas"],
    )
    logger.info("优化器初始化完成。")

    # 如果有checkpoint，则加载checkpoint
    start_iter = 1
    if data_paths["checkpoint_load_path"]:
        logger.info(f"开始加载模型检查点: {data_paths['checkpoint_load_path']}")
        start_iter = transformer_train.load_checkpoint(
            data_paths["checkpoint_load_path"],
            model=model,
            optimizer=optimizer
        )
        start_iter += 1
        logger.info(f"模型检查点加载成功，当前迭代次数: {start_iter}")
    else:
        logger.info("没有提供模型检查点，开始从头训练。")
    
    # 加载数据集
    logger.info(f"开始加载数据集，训练集：{data_paths['training_dataset_path']}, 验证集：{data_paths['validation_dataset_path']}")
    training_dataset = np.load(data_paths['training_dataset_path'], mmap_mode='r+') # 使用内存映射
    validation_dataset = None
    if data_paths['validation_dataset_path']:
        validation_dataset = np.load(data_paths['validation_dataset_path'], mmap_mode='r+')
    logger.info("数据集加载完成")

    # 计算训练所需step
    total_tokens = train_config["total_tokens"]
    total_steps = total_tokens // (train_config["batch_size"] * model_config["context_length"])
    logger.info(f"总token数: {total_tokens}, batch大小: {train_config['batch_size']}, 上下文长度: {model_config['context_length']}")
    logger.info(f"总训练步数: {total_steps}")

    # step循环开始
    logger.info("开始训练模型...")
    for step in tqdm(range(start_iter, total_steps + 1), desc="训练进度", unit="step"):
        model.train()
        # 清空梯度
        optimizer.zero_grad()

        # 使用余弦退火更新学习率
        lr_now = MyModules.cosine_decay_schedule(
            iteration=step,
            max_learning_rate=optim_config["lr"],
            min_learning_rate=optim_config["lr"] * 0.01,
            warmup_iters=int(0.05 * total_steps),
            anneal_iters=total_steps,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now
        
        # 获取batch数据
        inputs, targets = transformer_train.get_batch(
            training_dataset,
            batch_size=train_config["batch_size"],
            context_length=model_config["context_length"],
            device=device
        )

        # 前向传播
        logits = model(inputs)

        # 计算损失
        loss = MyModules.cross_entropy_loss(logits, targets)

        # 反向传播和优化参数
        loss.backward()

        # 计算梯度的 L2 范数
        if step % train_config["log_freq"] == 0:
            grad_norm = MyModules.grad_norm(model.parameters())
        
        MyModules.clip_grad_norm_(model.parameters(), max_norm=optim_config["max_norm"]) # 梯度裁剪

        optimizer.step()

        # 日志记录
        if step % train_config["log_freq"] == 0:
            logger.info(f"Step {step}, Loss: {loss.item()}, Grad L2 Norm: {grad_norm}")

            # 使用wandb记录损失和梯度范数
            wandb.log({"train_loss": loss.item(), "lr": lr_now, "grad_l2_norm": grad_norm, "step": step})
        
        # 在验证集上评估模型
        if validation_dataset is not None and step % train_config["val_freq"] == 0:
            logger.info(f"在验证集上评估模型...")
            model.eval()
            with torch.no_grad():
                num_batches = train_config["val_batches"]
                val_loss = 0
                for i in range(num_batches):
                    x_val, y_val = transformer_train.get_batch(
                        validation_dataset,
                        batch_size=train_config["val_batch_size"],
                        context_length=model_config["context_length"],
                        device=device
                    )
                    val_logits = model(x_val)
                    val_loss += MyModules.cross_entropy_loss(val_logits, y_val)
                val_loss /= num_batches # 求平均j
                print(f"[Step {step}] Validation loss: {val_loss.item():.4f}")
            model.train()
            logger.info(f"验证集损失: {val_loss}")
            wandb.log({"val_loss": val_loss, "step": step})
        
        # 保存检查点
        if step % train_config["checkpoint_freq"] == 0:
            checkpoint_save_path = data_paths["checkpoint_save_format"].format(step)
            logger.info(f"正在保存模型检查点到: {checkpoint_save_path}")
            transformer_train.save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=checkpoint_save_path
            )
            logger.info("模型检查点保存成功。")
    logger.info("模型训练完成。")
    
    # 保存最终模型
    logger.info(f"正在保存最终模型到: {data_paths['final_model_path']}")
    transformer_train.save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=total_steps,
        out=data_paths["final_model_path"],
    )
    logger.info("最终模型保存成功。")
    
    # 关闭wandb
    wandb.finish()


if __name__ == "__main__":
    log_path = WORK_DIR / "logs/train_v2.log"
    logger.add(log_path, rotation="1 day", retention="7 days", level="INFO")
    TRAIN_DATA_PATH = DATA_DIR / "tinystories_train_ids.npy"
    VAL_DATA_PATH = DATA_DIR / "tinystories_valid_ids.npy"  # 验证集路径
    CHECKPOINT_LOAD_PATH = None  # 模型检查点路径
    CHECKPOINT_SAVE_FORMAT = str(MODEL_DIR / "checkpoints/checkpoint_v2_{}.pt")  # 检查点保存路径格式
    FINAL_MODEL_PATH = MODEL_DIR / "finals/final_model_v2.pt"  # 最终模型保存路径

    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--offline-wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()
    if args.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    if args.offline_wandb:
        os.environ["WANDB_MODE"] = "offline"
    train()
