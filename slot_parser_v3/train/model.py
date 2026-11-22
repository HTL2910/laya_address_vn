"""Compact Word+Char Hybrid BiGRU + Linear-CRF model for Vietnamese address NER."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from slot_parser_v3.tokenizer import BIO_TAGS, build_bio_transition_mask


class SlotEncoder(nn.Module):
    """ONNX-exportable Word + Char-CNN + 2-Layer BiGRU + Linear emission encoder."""

    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        num_tags: int = len(BIO_TAGS),
        word_emb_dim: int = 64,
        char_emb_dim: int = 24,
        char_channels: int = 48,
        gru_hidden: int = 96,
        gru_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        self.word_emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.char_conv = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=char_channels,
            kernel_size=3,
            padding=1,
        )
        self.input_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=word_emb_dim + char_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.output_drop = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden * 2, num_tags)

    def forward(
        self,
        word_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        char_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute token emission logits [B, L, num_tags] from word & char IDs."""
        batch_size, seq_len = word_ids.shape
        char_len = char_ids.shape[2]

        w_feat = self.word_emb(word_ids)

        flat_chars = char_ids.reshape(batch_size * seq_len, char_len)
        c_emb = self.char_emb(flat_chars).transpose(1, 2)
        c_conv = torch.relu(self.char_conv(c_emb))
        char_valid = (flat_chars != 0).unsqueeze(1).to(c_conv.dtype)
        c_conv = c_conv * char_valid
        c_feat = torch.amax(c_conv, dim=-1).reshape(batch_size, seq_len, -1)

        mask_f = attention_mask.unsqueeze(-1).to(w_feat.dtype)
        x = torch.cat([w_feat, c_feat], dim=-1)
        x = self.input_drop(x) * mask_f

        # Explicit dynamic batch initial hidden state avoids ONNX fixed-batch warnings.
        h0 = torch.zeros(
            (self.gru_layers * 2, batch_size, self.gru_hidden),
            dtype=x.dtype,
            device=x.device,
        )
        out, _ = self.gru(x, h0)
        emissions = self.head(self.output_drop(out)) * mask_f
        return emissions


class LinearCRF(nn.Module):
    """Exact PyTorch Linear-CRF layer with hard BIO structural constraints."""

    def __init__(self, num_tags: int = len(BIO_TAGS), invalid_score: float = -1e4) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.invalid_score = float(invalid_score)

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        valid_start_np, valid_trans_np = build_bio_transition_mask()
        self.register_buffer("valid_start_mask", torch.from_numpy(valid_start_np))
        self.register_buffer("valid_trans_mask", torch.from_numpy(valid_trans_np))

    def constrained_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (start_transitions, transitions, end_transitions) with BIO mask applied."""
        start_t = self.start_transitions.masked_fill(~self.valid_start_mask, self.invalid_score)
        trans = self.transitions.masked_fill(~self.valid_trans_mask, self.invalid_score)
        return start_t, trans, self.end_transitions

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute batch mean exact CRF negative log-likelihood."""
        max_len = int(mask.sum(dim=1).max().item())
        emit = emissions[:, :max_len, :]
        gold = tags[:, :max_len]
        seq_mask = mask[:, :max_len]

        start_t, trans, end_t = self.constrained_parameters()
        batch_size = emit.shape[0]
        batch_idx = torch.arange(batch_size, device=emit.device)

        first_tags = gold[:, 0]
        gold_score = start_t[first_tags] + emit[batch_idx, 0, first_tags]
        alpha = start_t.unsqueeze(0) + emit[:, 0, :]

        trans_b = trans.unsqueeze(0)
        for t in range(1, max_len):
            m = seq_mask[:, t].to(emit.dtype)
            emit_t = emit[:, t, :]
            step_gold = trans[gold[:, t - 1], gold[:, t]] + emit_t[batch_idx, gold[:, t]]
            gold_score = gold_score + step_gold * m

            next_alpha = torch.logsumexp(alpha.unsqueeze(2) + trans_b, dim=1) + emit_t
            m_col = m.unsqueeze(1)
            alpha = next_alpha * m_col + alpha * (1.0 - m_col)

        seq_ends = seq_mask.sum(dim=1).long() - 1
        last_tags = gold[batch_idx, seq_ends]
        gold_score = gold_score + end_t[last_tags]

        alpha = alpha + end_t.unsqueeze(0)
        log_z = torch.logsumexp(alpha, dim=1)
        return (log_z - gold_score).mean()


class SlotTaggerCRF(nn.Module):
    """Combined encoder + Linear-CRF training module."""

    def __init__(self, encoder: SlotEncoder, crf: LinearCRF) -> None:
        super().__init__()
        self.encoder = encoder
        self.crf = crf

    def forward(
        self,
        word_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        char_ids: torch.Tensor,
        tags: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar CRF NLL loss for a batch."""
        emissions = self.encoder(word_ids, attention_mask, char_ids)
        return self.crf.neg_log_likelihood(emissions, tags, attention_mask)
