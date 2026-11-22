"""Way B v2 Dual-Granularity Transformer + Multi-Scale Char-CNN 3-Class Sufficiency Model.

Architecture:
- Branch 1 (Semantic Context): Factorized Word Embedding (64 -> 160) + 2-Layer Pre-Norm
  Transformer Encoder (4 heads, d_ff = 512) + CLS / Attention / Masked Mean Pooling.
- Branch 2 (Morphology / Typos / Abbreviations / Punctuation): Multi-Scale Character CNN
  (kernels 3, 5, 7; 64 channels each) + Masked Global Max + Mean Pooling.
- Shared Fusion Trunk -> Dual Output Heads:
  1. `logits` [B]: backward-compatible binary gate (`address_sufficient` vs rest)
  2. `sufficiency_logits` [B, 3]: 3-class `["not_address", "address_insufficient", "address_sufficient"]`
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Explicit MHA using nn.Linear so ONNX export and INT8 dynamic quantization stay optimal."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        key_mask = (1.0 - attention_mask).unsqueeze(1).unsqueeze(2)
        scores = scores + key_mask * (-1e4)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.o_proj(out)


class TransformerEncoderBlock(nn.Module):
    """Pre-Norm Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.drop_ff = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), attention_mask))
        h = self.ff2(self.drop_ff(F.gelu(self.ff1(self.norm2(x)))))
        x = x + self.drop2(h)
        return x


class LayaDualGranularitySufficiencyModel(nn.Module):
    """Dual-Granularity Transformer + Multi-Scale Char-CNN with Dual Binary & 3-Class Sufficiency Heads."""

    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        max_word_len: int = 64,
        word_emb_dim: int = 64,
        d_model: int = 160,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 2,
        char_emb_dim: int = 32,
        char_channels: int = 64,
        num_sufficiency_classes: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.max_word_len = max_word_len
        self.d_model = d_model

        # Branch 1: Factorized Word Embedding + 2-Layer Transformer Encoder
        self.word_emb = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0)
        self.word_proj = nn.Linear(word_emb_dim, d_model)
        self.pos_emb = nn.Embedding(max_word_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, n_heads, d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.attn_pool_score = nn.Linear(d_model, 1)
        self.word_branch_out = nn.Sequential(
            nn.Linear(d_model * 3, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Branch 2: Multi-Scale Character CNN (kernels 3, 5, 7)
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(char_emb_dim, char_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(char_emb_dim, char_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(char_emb_dim, char_channels, kernel_size=7, padding=3)
        self.char_branch_out = nn.Sequential(
            nn.Linear(char_channels * 6, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Shared Dual-Granularity Fusion Trunk -> Binary Gate + 3-Class Sufficiency Heads
        self.fusion_trunk = nn.Sequential(
            nn.Linear(192 + 128, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.binary_head = nn.Linear(96, 1)
        self.sufficiency_head = nn.Linear(96, num_sufficiency_classes)

    def _pool_conv(self, conv_out: torch.Tensor, char_mask: torch.Tensor) -> torch.Tensor:
        act = F.gelu(conv_out)
        masked_for_max = act + (1.0 - char_mask) * (-1e4)
        max_pool = masked_for_max.amax(dim=-1)
        denom = char_mask.sum(dim=-1).clamp(min=1.0)
        mean_pool = (act * char_mask).sum(dim=-1) / denom
        return torch.cat([max_pool, mean_pool], dim=-1)

    def forward(
        self,
        word_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        char_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Branch 1: Word Transformer
        seq_len = word_ids.size(1)
        mask_f = attention_mask.to(dtype=torch.float32)
        pos_ids = torch.arange(seq_len, device=word_ids.device, dtype=torch.long).unsqueeze(0)

        w = self.word_proj(self.word_emb(word_ids)) + self.pos_emb(pos_ids)
        w = self.emb_drop(self.emb_norm(w))

        for layer in self.layers:
            w = layer(w, mask_f)
        w = self.final_norm(w)

        cls_repr = w[:, 0, :]
        attn_logits = self.attn_pool_score(w).squeeze(-1) + (1.0 - mask_f) * (-1e4)
        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)
        attn_repr = (w * attn_weights).sum(dim=1)

        mask_exp = mask_f.unsqueeze(-1)
        mean_repr = (w * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)

        word_feat = self.word_branch_out(torch.cat([cls_repr, attn_repr, mean_repr], dim=-1))

        # Branch 2: Multi-Scale Char CNN
        c_mask = (char_ids != 0).to(dtype=torch.float32).unsqueeze(1)
        c_emb = self.char_emb(char_ids).transpose(1, 2)
        c3 = self._pool_conv(self.conv3(c_emb), c_mask)
        c5 = self._pool_conv(self.conv5(c_emb), c_mask)
        c7 = self._pool_conv(self.conv7(c_emb), c_mask)
        char_feat = self.char_branch_out(torch.cat([c3, c5, c7], dim=-1))

        # Shared Fusion -> Dual Output Heads
        fused = self.fusion_trunk(torch.cat([word_feat, char_feat], dim=-1))
        logits = self.binary_head(fused).squeeze(-1)
        sufficiency_logits = self.sufficiency_head(fused)
        return logits, sufficiency_logits
