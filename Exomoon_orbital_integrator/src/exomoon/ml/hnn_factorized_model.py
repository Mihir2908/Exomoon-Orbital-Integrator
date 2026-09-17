"""
exomoon/ml/hnn_factorized_model.py — Physics-shaped factorized HNN potential.

Architecture (v5 — physics-scaled):

    V_total = V_sp(d_sp_n)              [hardcoded — ∂V_sp/∂d_sp_n = 1/d_sp_n² exactly]
            + V_sm(d_sm_n, sys_enc)     [physics-shaped: -C_sm(sys_enc) / d_sm_n]
            + V_pm(d_pm_n, sys_enc)     [physics-shaped: -C_pm(sys_enc) / d_pm_n]

Each C_x(sys_enc) is a tiny MLP from 5-D sys_enc → positive scalar (via softplus).

Why physics-shaped potentials?
-------------------------------
t_pm = C_pm_true / d_pm_n²  where C_pm_true = (mm/ms) × (ap/rhill).
t_pm spans ~9 orders of magnitude across the training set ENTIRELY because of 1/d_pm_n²;
C_pm_true itself only varies ~5 orders of magnitude and is constant per-system across
timesteps.

A generic MLP V_pm(d_pm_n) trained on this dataset finds a global minimum of
pred_pm ≈ 0 everywhere: the 80% escaped-habitable training rows (d_pm_n up to 736,
t_pm ≈ 1e-11) contribute near-zero loss for any positive prediction, while only the
20% stable rows (d_pm_n ≤ 1, t_pm ≈ 0.001–0.01) contribute non-negligible loss.
The model "gives up" on learning t_pm to minimise average loss.

Physics-shaped V_pm = -C_pm(sys_enc) / d_pm_n → ∂V_pm/∂d_pm_n = C_pm/d_pm_n²
log-MSE loss becomes: (log C_pm_pred − log C_pm_true)²
This is CONSTANT across all d_pm_n values for a given system — no distribution
imbalance between stable and escaped-habitable rows. Both contribute equal gradient
signal for learning C_pm from sys_enc.

Additional benefit: if C_pm is learned exactly, V_θ = V_Newton, H_θ = H_Newton,
and circular-orbit initial conditions are energetically correct — directly solving
the root cause of HNN v12 failure.

Training: same hnn_factorized_train.py, same autograd.grad(V, d_n_leaf) call.
Inference: same KDK leapfrog with forces from autograd.

ISOLATION: no imports from dataset.py, model.py, train.py, inference.py,
           models_temphead/, models/, eval_aux_mlp_output/.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
from torch import Tensor

SYS_DIM = 5   # ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU
D_DIM   = 3   # d_sp_n, d_sm_n, d_pm_n


def _make_c_net(hidden: int, layers: int) -> nn.Sequential:
    """Tiny MLP: sys_enc (5-D) → scalar C value (raw, before softplus)."""
    parts: list[nn.Module] = [nn.Linear(SYS_DIM, hidden), nn.Tanh()]
    for _ in range(layers - 1):
        parts += [nn.Linear(hidden, hidden), nn.Tanh()]
    parts.append(nn.Linear(hidden, 1))
    return nn.Sequential(*parts)


class HNNFactorized(nn.Module):
    """
    Physics-shaped factorized HNN (v5).

    V_total = -1/d_sp_n                          (V_sp: hardcoded Newtonian, no learning)
            − softplus(C_sm_net(sys_enc))/d_sm_n  (V_sm: C_sm learned from sys params)
            − softplus(C_pm_net(sys_enc))/d_pm_n  (V_pm: C_pm learned from sys params)

    Forces via autograd(V_total, d_n_leaf) — always positive by construction.

    C_sm_true ≡ mm/mp  (mass ratio, varies 3 orders of magnitude across training set)
    C_pm_true ≡ (mm/ms) × (ap/rhill)  (varies 5 orders of magnitude)

    Both C values depend only on sys_enc, not on distance — so log-MSE loss is
    constant across d values for a given system, eliminating the distribution
    imbalance that caused v1–v4 to converge to pred=0 for t_pm.
    """

    def __init__(self, hidden: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        # Two tiny C-networks; V_sp is hardcoded, no C_sp_net needed.
        self.C_sm_net = _make_c_net(hidden, layers)
        self.C_pm_net = _make_c_net(hidden, layers)

    def forward(self, d_n: Tensor, sys_enc: Tensor) -> Tensor:
        """
        d_n      : (batch, 3)  — [d_sp_n, d_sm_n, d_pm_n]  leaf tensor for autograd
        sys_enc  : (batch, 5)  — StandardScaler-normalised [ms, mp, mm, ap, rhill]
        Returns  : (batch,)    — total scalar potential V_total

        V_sp = -1/d_sp_n              → ∂V_sp/∂d_sp = +1/d_sp²  (always positive)
        V_sm = -C_sm / d_sm_n         → ∂V_sm/∂d_sm = +C_sm/d_sm²  (C_sm > 0 via softplus)
        V_pm = -C_pm / d_pm_n         → ∂V_pm/∂d_pm = +C_pm/d_pm²  (C_pm > 0 via softplus)
        """
        d_sp = d_n[:, 0:1]   # (batch, 1)
        d_sm = d_n[:, 1:2]
        d_pm = d_n[:, 2:3]

        # V_sp: hardcoded exact Newtonian — no parameters, autograd gives t_sp = 1/d_sp²
        V_sp = -1.0 / d_sp                                              # (batch, 1)

        # V_sm: physics-shaped, C_sm learned from system parameters
        C_sm = torch.nn.functional.softplus(self.C_sm_net(sys_enc))    # (batch, 1), > 0
        V_sm = -C_sm / d_sm                                            # (batch, 1)

        # V_pm: physics-shaped, C_pm learned from system parameters
        C_pm = torch.nn.functional.softplus(self.C_pm_net(sys_enc))    # (batch, 1), > 0
        V_pm = -C_pm / d_pm                                            # (batch, 1)

        return (V_sp + V_sm + V_pm).squeeze(-1)                        # (batch,)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(),
                   os.path.join(out_dir, "hnn_factorized_model.pt"))
        cfg = {"model_type": "factorized_v5_physics_scaled",
               "hidden": self.hidden, "layers": self.layers}
        with open(os.path.join(out_dir, "hnn_factorized_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, model_dir: str, map_location: str = "cpu") -> "HNNFactorized":
        cfg_path = os.path.join(model_dir, "hnn_factorized_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        model = cls(hidden=cfg["hidden"], layers=cfg["layers"])
        model.load_state_dict(
            torch.load(os.path.join(model_dir, "hnn_factorized_model.pt"),
                       map_location=map_location, weights_only=True)
        )
        model.to(map_location)
        return model
