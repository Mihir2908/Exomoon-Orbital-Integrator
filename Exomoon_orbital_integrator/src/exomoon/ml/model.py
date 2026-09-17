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
# STATE_DIM = 7: stable/habitable removed from GRU inputs to eliminate binary
# attractors during autoregressive inference. They remain as TARGET_COLS (flag head
# BCE loss), so the model still learns stability — just not from explicit binary feedback.
# moon_temp_norm is a second, independently-headed encoding of habitability (see
# dataset.py) -- added to test whether two separately-learned views of the same
# underlying fact disagree usefully at exactly the steps where one of them goes wrong.
# SYS_DIM = 14: includes a_inner_au/a_outer_au alongside rhill_AU (both are deterministic
# closed-form physics quantities, not literature thresholds).
SYS_DIM   = 14   # system params
STATE_DIM =  7   # per-step state (moon_planet_dist_norm, moon_star_dist_norm, moon_temp_norm, planet_star_dist, moon_speed, planet_speed, t_frac)
OUT_DIM   =  9   # 6 dist/speed + 3 flag logits
N_DIST    =  6   # number of regression outputs
N_FLAGS   =  3   # number of binary classification outputs (stable, habitable, habitable_from_temp)
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
        dist_pred  : (batch, T, N_DIST)   — predicted distances/speeds
        flag_logits: (batch, T, N_FLAGS)  — raw logits for stable/habitable/habitable_from_temp

        dist_pred column order must match dataset.py's TARGET_DIST_COLS:
        [moon_planet_dist_norm, moon_star_dist_norm, moon_temp_norm, planet_star_dist_norm,
         moon_speed, planet_speed].
        Only column 0 (Hill-radius fraction), 3 (planet_star_dist_norm = planet_star_dist/ap_AU,
        orbital phase), 4 (moon_speed), and 5 (planet_speed) are physically non-negative --
        relu applies to those.
        Columns 1 (moon_star_dist_norm) and 2 (moon_temp_norm) are signed HZ-band-position
        quantities: negative means past the inner edge, >0.881 (post-arcsinh) means past
        the outer edge -- relu would erase exactly that information.
        """
        batch, T, _ = state_seq.shape

        # Expand sys_params along time and concatenate
        sys_exp  = sys_params.unsqueeze(1).expand(-1, T, -1)   # (batch, T, SYS_DIM)
        rnn_in   = torch.cat([state_seq, sys_exp], dim=-1)     # (batch, T, SYS_DIM+STATE_DIM)

        rnn_out, _ = self.rnn(rnn_in)    # (batch, T, hidden) — works for both GRU and LSTM

        raw_dist = self.head_dist(rnn_out)
        dist_pred = torch.cat([
            torch.relu(raw_dist[..., 0:1]),    # moon_planet_dist_norm >= 0
            raw_dist[..., 1:3],                 # moon_star_dist_norm, moon_temp_norm: unconstrained (signed)
            torch.relu(raw_dist[..., 3:6]),    # planet_star_dist, moon_speed, planet_speed >= 0
        ], dim=-1)
        flag_logits = self.head_flags(rnn_out)               # raw logits

        return dist_pred, flag_logits

    # ── loss ──────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_loss(
        dist_pred:    torch.Tensor,   # (batch, T, N_DIST)
        flag_logits:  torch.Tensor,   # (batch, T, N_FLAGS)
        targets:      torch.Tensor,   # (batch, T, OUT_DIM = N_DIST + N_FLAGS)
        dist_mask:    torch.Tensor = None,   # (batch, T, 1) -- 1 while stable, 0 after escape
        flag_pos_weight: torch.Tensor = None,   # (N_FLAGS,) -- per-column BCE positive-class weight
    ) -> tuple[torch.Tensor, dict]:
        """
        Combined loss: MSE for distances/speeds + weighted BCE for flags.

        dist_mask (if given) excludes post-escape rows from the MSE term: once a
        moon leaves the planet's Hill sphere, its raw distance/speed is unbounded
        ballistic drift irrelevant to the stability-mapping task, and without
        masking a handful of such rows can dominate the regression loss. The flag
        (BCE) term is never masked -- stable/habitable remain true, meaningful
        labels for every row regardless of distance scale.

        flag_pos_weight (if given) corrects for class imbalance independently per
        flag column. habitable=1 occurs in only ~12% of training rows (LHS samples
        Ts/rs_solar/ap_AU independently, so most sampled systems never put the
        planet anywhere near their own star's HZ) -- without per-column reweighting,
        plain BCE collapses to predicting habitable=0 almost everywhere, since that
        already minimizes loss on the majority class. stable is not similarly
        imbalanced (~81% positive) and needs no correction.

        Returns (total_loss, {'mse': ..., 'bce': ...}).
        """
        target_dist  = targets[..., :N_DIST]
        target_flags = targets[..., N_DIST:]

        if dist_mask is not None:
            sq_err   = (dist_pred - target_dist) ** 2
            valid_n  = (dist_mask.sum() * N_DIST).clamp(min=1.0)
            mse_loss = (sq_err * dist_mask).sum() / valid_n
        else:
            mse_loss = nn.functional.mse_loss(dist_pred, target_dist)

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            flag_logits, target_flags, pos_weight=flag_pos_weight
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

    def save(self, out_dir: str, filename: str = None) -> None:
        """Save state dict + {rnn_type}_model_config.json to out_dir."""
        os.makedirs(out_dir, exist_ok=True)
        if filename is None:
            filename = f"{self.rnn_type}_model.pt"
        torch.save(self.state_dict(), os.path.join(out_dir, filename))
        cfg_path = os.path.join(out_dir, f"{self.rnn_type}_model_config.json")
        with open(cfg_path, "w") as f:
            json.dump(self.get_config(), f, indent=2)

    @classmethod
    def load(cls, out_dir: str, rnn_type: str = "gru",
             map_location: str = "cpu") -> "MoonRNN":
        """Load from out_dir using {rnn_type}_model_config.json."""
        cfg_path = os.path.join(out_dir, f"{rnn_type}_model_config.json")
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(out_dir, "model_config.json")  # backward compat
        with open(cfg_path) as f:
            cfg = json.load(f)
        model = cls(**cfg)
        state = torch.load(os.path.join(out_dir, f"{rnn_type}_model.pt"),
                           map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model
