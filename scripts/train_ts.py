import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from einops import rearrange
from cs336_basics import transformer_train, MyModules, bpe_tokenizer
import os

# ======= 训练过程 =======
def train(args):
    device = torch.device(args.device)

    # 加载数据
    train_data = np.load(args.train_data, mmap_mode='r')
    val_data = np.load(args.val_data, mmap_mode='r')

    # 模型和优化器
    model = MyModules.transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    ).to(device)
    optimizer = MyModules.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = MyModules.cross_entropy_loss()

    # 加载 checkpoint（如果有）
    start_step = 0
    if args.resume and os.path.exists(args.checkpoint_path):
        start_step = transformer_train.load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Resumed from step {start_step}")

    for step in range(start_step, args.total_steps):
        model.train()
        inputs, targets = transformer_train.get_batch(train_data, args.batch_size, args.context_length, device)
        # inputs: (batch_size, context_length)
        # targets: (batch_size, context_length)
        logits = model(inputs) # (batch_size, context_length, vocab_size)
        loss = criterion(logits, targets) # (batch_size, vocab_size)

        optimizer.zero_grad()
        # 是每次更新前都必须调用的函数，用于清空梯度，否则梯度会累加导致训练错误。
        loss.backward()
        MyModules.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        # 日志输出
        if step % args.log_every == 0:
            print(f"[Step {step}] Training loss: {loss.item():.4f}")

        # 验证集评估
        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = transformer_train.get_batch(val_data, args.batch_size, args.context_length, device)
                val_logits = model(x_val)
                val_loss = criterion(val_logits, y_val)
                print(f"[Step {step}] Validation loss: {val_loss.item():.4f}")

        # 保存 checkpoint
        if step % args.save_every == 0:
            transformer_train.save_checkpoint(model, optimizer, step, args.checkpoint_path)

# ======= 参数解析器 =======
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad', type=float, default=2.0)
    parser.add_argument('--total_steps', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pt')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)
