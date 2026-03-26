import torch
import torch.nn as nn
import math

# -----------------------------
# Multi-Head Self Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 一次產生 Q/K/V，比較省事
        self.qkv = nn.Linear(dim, dim * 3, bias=True)

        # 縮放係數 (1/sqrt(d_k))
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 注意力與輸出投影
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (B, N, C)  -> B: batch, N: tokens, C: dim
        """
        B, N, C = x.shape

        # [B, N, 3*C] -> [B, N, 3, h, d] -> [3, B, h, N, d]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]   # 各是 (B, h, N, d)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B, h, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加權求和回 V
        y = torch.matmul(attn, v)           # (B, h, N, d)
        y = y.transpose(1, 2).contiguous()  # (B, N, h, d)
        y = y.view(B, N, C)                 # 合併 heads -> (B, N, C)

        # 最後做一次線性投影
        y = self.proj(y)
        return y


class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=drop_rate),
        )

    def forward(self, x):
        return super().forward(x)


class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super().__init__(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12),
        )

    def forward(self, x):
        return super().forward(x)


class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim)
        self.ln1 = nn.LayerNorm(dim, eps=1e-12)
        self.ln2 = nn.LayerNorm(dim, eps=1e-12)
        self.mlp = MLP(dim, hidden_dim)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        # SA 殘差
        sa = self.self_attn(x)
        sa = self.drop(sa)
        x = self.ln1(x + sa)

        # MLP 殘差
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)
        return x
