from torch import nn
import torch
from einops import einsum

class Linear(nn.Module):

    def __init__(
        self, 
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # Create a tensor for weights with shape (out_features, in_features)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.W = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        # 计算标准差 std = sqrt(2 / (in + out))
        std = (2 / (in_features + out_features)) ** 0.5
        # 使用 trunc_normal_ 初始化权重
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
