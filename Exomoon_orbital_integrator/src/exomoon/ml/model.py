"""
exomoon/ml/model.py — MoonRNN: GRU or LSTM physics emulator.

The model takes a sequence of per-step state features concatenated with
fixed system parameters (conditioning) and predicts the next-step state.

Architecture
------------
input  : [state_t | sys_params]   shape (batch, seq, SYS_DIM + STATE_DIM)
RNN    : nn.GRU or nn.LSTM        hidden_size, num_layers
output : linear head              → (batch, seq, OUT_DIM)
           OUT_DIM = 5 distance/speed + 2 flag logits

Loss   : MSE for distances/speeds + 3× BCEWithLogitsLoss for flags
"""

from __future__ import annotations

import json
import os
from typing import Literal

import torch
import torch.nn as nn

RnnType = Literal["gru", "lstm"]

# Feature dims (must match dataset.py)
SYS_DIM   = 12   # system params
STATE_DIM =  8   # per-step state
OUT_DIM   =  7   # 5 dist/speed + 2 flag logits
N_DIST    =  5   # number of regression outputs
N_FLAGS   =  2   # number of binary classification outputs
FLAG_WEIGHT = 3.0  # BCE loss weight relative to MSE


class MoonRNN(nn.Module):
    """
    Physics emulator for exomoon trajectory prediction.

    Parameters
    ----------
    system_dim : int   — dimension of constant system params (default 12)
    state_dim  : int   — dimension of per-step state (default 8)
    hidden     : int   — GRU/LSTM hidden size
    layers     : int   — number of stacked RNN layers
    rnn_type   : "gru" | "lstm"
    dropout    : float — dropout between RNN layers (0 = disabled)
    """

    def __init__(
        self,
        system_dim: int   = SYS_DIM,
        state_dim:  int   = STATE_DIM,
        hidden:     int   = 256,
        layers:     int   = 2,
        rnn_type:   RnnType = "gru",
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.rnn_type   = rnn_type
        self.hidden     = hidden
        self.layers     = layers
        self.system_dim = system_dim
        self.state_dim  = state_dim

        input_dim = system_dim + state_dim   # 20 by default

        rnn_dropout = dropout if layers > 1 else 0.0
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim, hidden, layers,
                batch_first=True, dropout=rnn_dropout,
            )
        else:
            self.rnn = nn.GRU(
                input_dim, hidden, layers,
                batch_first=True, dropout=rnn_dropout,
            )

        # Output head: shared linear → split into dist head + flag head
        self.head_dist  = nn.Linear(hidden, N_DIST)    # MSE targets
        self.head_flags = nn.Linear(hidden, N_FLAGS)   # BCE targets (logits)

    def forward(
        self,
        state_seq:   torch.Tensor,   # (batch, T, STATE_DIM)
        sys_params:  torch.Tensor,   # (batch, SYS_DIM)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        dist_pred  : (batch, T, N_DIST)   — predicted distances/speeds (relu)
        flag_logits: (batch, T, N_FLAGS)  — raw logits for stable/habitable
        """
        batch, T, _ = state_seq.shape

        # Expand sys_params along time and concatenate
        sys_exp  = sys_params.unsqueeze(1).expand(-1, T, -1)   # (batch, T, SYS_DIM)
        rnn_in   = torch.cat([state_seq, sys_exp], dim=-1)     # (batch, T, SYS_DIM+STATE_DIM)

        rnn_out, _ = self.rnn(rnn_in)    # (batch, T, hidden) — works for both GRU and LSTM

        dist_pred   = torch.relu(self.head_dist(rnn_out))    # distances are non-negative
        flag_logits = self.head_flags(rnn_out)               # raw logits

        return dist_pred, flag_logits

    # ── loss ──────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_loss(
        dist_pred:    torch.Tensor,   # (batch, T, N_DIST)
        flag_logits:  torch.Tensor,   # (batch, T, N_FLAGS)
        targets:      torch.Tensor,   # (batch, T, OUT_DIM = N_DIST + N_FLAGS)
    ) -> tuple[torch.Tensor, dict]:
        """
        Combined loss: MSE for distances/speeds + weighted BCE for flags.
        Returns (total_loss, {'mse': ..., 'bce': ...}).
        """
        target_dist  = targets[..., :N_DIST]
        target_flags = targets[..., N_DIST:]

        mse_loss = nn.functional.mse_loss(dist_pred, target_dist)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            flag_logits, target_flags
        )
        total = mse_loss + FLAG_WEIGHT * bce_loss
        return total, {"mse": mse_loss.item(), "bce": bce_loss.item()}

    # ── serialisation helpers ─────────────────────────────────────────────────
    def get_config(self) -> dict:
        return {
            "rnn_type":   self.rnn_type,
            "hidden":     self.hidden,
            "layers":     self.layers,
            "system_dim": self.system_dim,
            "state_dim":  self.state_dim,
        }

    def save(self, out_dir: str, filename: str = "gru_model.pt") -> None:
        """Save state dict + model_config.json to out_dir."""
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_dir, filename))
        cfg_path = os.path.join(out_dir, "model_config.json")
        with open(cfg_path, "w") as f:
            json.dump(self.get_config(), f, indent=2)

    @classmethod
    def load(cls, out_dir: str, filename: str = "gru_model.pt",
             map_location: str = "cpu") -> "MoonRNN":
        """Load from out_dir using saved model_config.json."""
        cfg_path = os.path.join(out_dir, "model_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        model = cls(**cfg)
        state = torch.load(os.path.join(out_dir, filename), map_location=map_location,
                           weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model
