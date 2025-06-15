from torch import nn
import torch
import torch.nn.functional as F
import math
from torch import Tensor
from einops import einsum, reduce, rearrange, repeat
from jaxtyping import Float, Bool, Int

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
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        # 计算标准差 std = sqrt(2 / (in + out))
        std = (2 / (in_features + out_features)) ** 0.5
        # 使用 trunc_normal_ 初始化权重
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # 创建 embedding 矩阵参数，形状 [num_embeddings, embedding_dim]
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # 按照要求用截断正态分布初始化，均值0，方差1，截断区间[-3, 3]
        # 注意 std = 1 (方差1)，截断区间[-3,3]
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids形状任意，元素是索引，返回对应embedding
        # 这里不允许用nn.functional.embedding，自己实现索引
        # 直接用tensor索引即可，PyTorch的张量支持整数索引
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.scale = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        sq_mean = reduce(x**2, "batch sequence d_model -> batch sequence 1", "mean")
        rms = torch.sqrt(sq_mean + self.eps)
        norm_x = x / rms

        return norm_x * self.scale

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
        """
        
        return self.w2(self.w1(x) * self.sigmoid(self.w1(x)) * self.w3(x)) 

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None) :
        """Construct the
        RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires that d_k be even")
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        # 计算频率
        freq = 1.0 / ( self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k ) )
        positions = torch.arange(self.max_seq_len)
        angles = einsum(positions, freq, "max_seq_len, d_k -> max_seq_len d_k")
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self, 
        x: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> torch.Tensor :
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
        that you should tolerate x with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
        x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors along
        the sequence dimension.
        """
        cos = self.cos_cache[token_positions] # shape (..., seq_len, d_k // 2)
        sin = self.sin_cache[token_positions] 
        
        x1 = x[..., 0::2] # even positions shapre (..., seq_len, d_k // 2)
        x2 = x[..., 1::2] # odd positions

        output1 = x1 * cos - x2 * sin
        output2 = x1 * sin + x2 * cos
        output = rearrange([output1, output2], 'out_num ... d_k -> ... (d_k out_num)') # 交错排列output1 output2

        return output
        
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Write a function to apply the softmax operation on a tensor. Your function should
    take two parameters: a tensor and a dimension i, and apply softmax to the i-th dimension of the input
    tensor. The output tensor should have the same shape as the input tensor, but its i-th dimension will
    now have a normalized probability distribution. Use the trick of subtracting the maximum value in
    the i-th dimension from all elements of the i-th dimension to avoid numerical stability issues.
    """
    max = torch.amax(x, dim=dim, keepdim=True)
    exp_x = torch.exp(x - max) # for numerical stability
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
        Q: Float[Tensor, "batch_size ... n d_k"],
        K: Float[Tensor, "batch_size ... m d_k"],
        V: Float[Tensor, "batch_size ... m d_v"],
        mask: Bool[Tensor, "... seq_len seq_len"] | None = None,
) -> Float[Tensor, "batch_size ... d_v"]:
    """Implement the scaled dot-product attention function. Your implementation should
    handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
    (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
    dimensions (if provided). The implementation should return an output with the shape (batch_size,
    ..., d_v). See section 3.3 for a discussion on batch-like dimensions.
    Your implementation should also support an optional user-provided boolean mask of shape (seq_len,
    seq_len). The attention probabilities of positions with a mask value of True should collectively sum
    to 1, and the attention probabilities of positions with a mask value of False should be zero.
    """

    d_k = Q.size(-1)
    # 第 i 个 query 对第 j 个 key 的相关性（相似度）
    scores = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))  # 屏蔽位置设为 -inf
    
    attn = softmax(scores, dim=-1) # 某个query对所有key的相关性 进行softmax归一化

    output = einsum(attn, V,"... n m, ... m d_v -> ... n d_v")
    
    return output

class MHSA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        # Linear projections
        self.qkv_proj = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if theta is not None:
            self.rope = RoPE(self.theta, self.head_dim ,self.max_seq_len, device=device, dtype=dtype) 

    def forward(
            self, 
            x: Float[Tensor, " ... sequence_length d_in"],
            mask: Bool[Tensor, " ... sequence_length sequence_length"] | None = None,
            token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        
        # Linear projections and reshape to make num_head to be a batch dimension for parallelization
        # calculate qkv in one go
        qkv = rearrange(self.qkv_proj(x), " ... sequence_length (t num_heads d_k) -> t ... num_heads sequence_length d_k", t=3, num_heads=self.num_heads)
        q,k,v = qkv[0], qkv[1], qkv[2]

        if self.theta is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        # Reshape and apply final linear projection
        attn_output = rearrange(attn_output, "... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)", num_heads=self.num_heads)
        output = self.out_proj(attn_output)
        
        return output
        
class transformer_block(nn.Module):
    def __init__(
        self,
        d_model: int, # Dimensionality of the transformer block input.
        num_heads: int, # Number of heads to use in multi-headed attention.
        d_ff: int, # Dimensionality of the position-wise feed-forward inner layer.
        theta: float | None = None, # RoPE parameter.
        max_seq_len: int | None = None, # RoPE parameter used to pre-cache.
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        
        self.attn = MHSA(d_model, num_heads, theta, max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        mask: Bool[Tensor, "... sequence_length sequence_length"] | None = None, # mask attention
        token_positions: Int[Tensor, "... sequence_length"] | None = None,# RoPE
    ) -> Float[Tensor, "... sequence_length d_model"]:
        x = x + self.attn(self.ln1(x), mask, token_positions) # first sub layer for self attention
        x = x + self.ffn(self.ln2(x)) # second sub layer for position-wise feed forward
        return x

    def load_weights(self, weights: dict[str, torch.Tensor]):
        """only used for adapter to load the weights not for general use"""
        with torch.no_grad():
            q_proj_weight = weights["attn.q_proj.weight"]
            k_proj_weight = weights["attn.k_proj.weight"]
            v_proj_weight = weights["attn.v_proj.weight"]
            qkv_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
            self.attn.qkv_proj.weight.copy_(qkv_proj_weight)
            self.attn.out_proj.weight.copy_(weights["attn.output_proj.weight"])

            self.ffn.w1.weight.copy_(weights["ffn.w1.weight"])
            self.ffn.w2.weight.copy_(weights["ffn.w2.weight"])
            self.ffn.w3.weight.copy_(weights["ffn.w3.weight"])

            self.ln1.scale.copy_(weights["ln1.weight"])
            self.ln2.scale.copy_(weights["ln2.weight"])

class transformer_lm(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype
        
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            transformer_block(d_model, num_heads, d_ff, rope_theta, context_length) 
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
            self,
            x: Int[Tensor, " batch_size sequence_length"],
            mask: Bool[Tensor, " batch_size sequence_length sequence_length"] | None = None,
            token_positions: Int[Tensor, " batch_size sequence_length"] | None = None,
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:

        x = self.token_embeddings(x) # (batch_size, sequence_length, d_model)
        for layer in self.layers:
            x = layer(x, mask, token_positions) # (batch_size, sequence_length, d_model)
        x = self.ln_final(x) # (batch_size, sequence_length, d_model)
        x = self.lm_head(x) # (batch_size, sequence_length, vocab_size)
        # 不在这里进行softmax, 因为直接在最后利用log-softmax计算loss
        # x = softmax(x, dim=-1) # (batch_size, sequence_length, vocab_size)
        return x
    
    def load_weights(self, weights: dict[str, torch.Tensor]):
        """only used for adapter to load the weights not for general use"""
        with torch.no_grad():
            self.token_embeddings.weight.copy_(weights["token_embeddings.weight"])
            for index, layer in enumerate(self.layers):
                q_proj_weight = weights[f"layers.{index}.attn.q_proj.weight"]
                k_proj_weight = weights[f"layers.{index}.attn.k_proj.weight"]
                v_proj_weight = weights[f"layers.{index}.attn.v_proj.weight"]
                qkv_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
                layer.attn.qkv_proj.weight.copy_(qkv_proj_weight)
                layer.attn.out_proj.weight.copy_(weights[f"layers.{index}.attn.output_proj.weight"])

                layer.ffn.w1.weight.copy_(weights[f"layers.{index}.ffn.w1.weight"])
                layer.ffn.w2.weight.copy_(weights[f"layers.{index}.ffn.w2.weight"])
                layer.ffn.w3.weight.copy_(weights[f"layers.{index}.ffn.w3.weight"])

                layer.ln1.scale.copy_(weights[f"layers.{index}.ln1.weight"])
                layer.ln2.scale.copy_(weights[f"layers.{index}.ln2.weight"])
            self.ln_final.scale.copy_(weights["ln_final.weight"])
            self.lm_head.weight.copy_(weights["lm_head.weight"])
