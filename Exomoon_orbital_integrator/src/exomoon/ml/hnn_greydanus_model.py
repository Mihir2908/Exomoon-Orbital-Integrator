"""
exomoon/ml/hnn_greydanus_model.py — Option E (v2): Greydanus-style joint H_θ(d_n, p_n, sys_enc).

Unlike the v12 HNN which learns only V_θ(d_n, sys_enc), this joint model takes
both physics-normalised pair distances AND normalised canonical momenta:

  H_θ(d_sp_n, d_sm_n, d_pm_n, p_s_n, p_p_n, p_m_n, sys_enc) → scalar

where p_n_i = (m_i/mp) * v_n_i  — normalised canonical momentum (NOT velocity).
  Star:   p_n_star   = (ms/mp) * v_n_star   — O(1) even though v_star is tiny
  Planet: p_n_planet = 1.0     * v_n_planet
  Moon:   p_n_moon   = (mm/mp) * v_n_moon

Two supervision signals constrain H_θ during training:
  1. Force supervision:     log-MSE( |∂H_θ/∂d_n|,  t_grad )  — same as HNN v12
  2. Momentum supervision:  MSRE(    ∂H_θ/∂p_n_i,  v_n_i  )  — Hamilton: ∂H/∂p = v

For the true Newtonian H = T(p) + V(q):
  ∂H/∂q_i = ∂V/∂q_i  — force, purely positional
  ∂H/∂p_i = p_i/m_i = v_i — velocity, Hamilton's equation

At inference, forces = -∂H_θ(d_n, p_n_current)/∂q using current velocities converted
to p_n. KDK leapfrog with p_n recomputed at each half-step.

ISOLATION: no imports from dataset.py, model.py, train.py, inference.py.
Never reads/writes models/, models_temphead/, eval_aux_mlp_output/, models_hnn_dist_v12/.
"""

import json
import os

import torch
import torch.nn as nn
from torch import Tensor

D_DIM   = 3    # d_sp_n, d_sm_n, d_pm_n
P_DIM   = 9    # [p_s_x, p_s_y, p_s_z, p_p_x, p_p_y, p_p_z, p_m_x, p_m_y, p_m_z]
SYS_DIM = 5    # ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU

V_DIM   = P_DIM   # alias — kept for backward compat with any callers checking .v_dim


class HNNGreydanus(nn.Module):
    """
    Joint H_θ(d_n, p_n, sys_enc) → scalar (dimensionless Hamiltonian H/H_ref).

    H_ref = G·ms·mp/ap = 4π²·ms·mp/ap  [Msun·AU²/yr²].

    Input p_n_i = (m_i/mp)*v_n_i — canonical normalised momentum.
    Forces at inference: -∂H_θ/∂q (autograd w.r.t. absolute positions).
    Momentum supervision: ∂H_θ/∂p_n_i = v_n_i  (Hamilton's equation ∂H/∂p = v).
    """

    def __init__(
        self,
        d_dim:   int   = D_DIM,
        p_dim:   int   = P_DIM,
        sys_dim: int   = SYS_DIM,
        hidden:  int   = 128,
        layers:  int   = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_dim   = d_dim
        self.p_dim   = p_dim
        self.v_dim   = p_dim   # alias for backward compat
        self.sys_dim = sys_dim
        self.hidden  = hidden
        self.layers  = layers
        self.dropout = dropout

        in_dim = d_dim + p_dim + sys_dim   # 3 + 9 + 5 = 17
        parts: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            parts += [nn.Linear(d, hidden), nn.Tanh()]
            if dropout > 0.0:
                parts.append(nn.Dropout(dropout))
            d = hidden
        parts.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*parts)

    def forward(self, d_n: Tensor, p_n: Tensor, sys_enc: Tensor) -> Tensor:
        """
        d_n     : (batch, 3)  physics-normalised distances [d_sp/ap, d_sm/ap, d_pm/rhill]
        p_n     : (batch, 9)  normalised momenta (m_i/mp)*v_n [ps_x/y/z, pp_x/y/z, pm_x/y/z]
        sys_enc : (batch, 5)  StandardScaler-normalised [ms, mp, mm, ap, rhill]
        Returns : (batch,)    H_θ scalar per sample
        """
        x = torch.cat([d_n, p_n, sys_enc], dim=-1)
        return self.net(x).squeeze(-1)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_dir, "hnn_model.pt"))
        cfg = {
            "model_type": "greydanus",
            "input_type": "momentum",   # p_n = (m_i/mp)*v_n, NOT velocity
            "d_dim":      self.d_dim,
            "p_dim":      self.p_dim,
            "sys_dim":    self.sys_dim,
            "hidden":     self.hidden,
            "layers":     self.layers,
            "dropout":    self.dropout,
        }
        with open(os.path.join(out_dir, "hnn_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, model_dir: str, map_location: str = "cpu") -> "HNNGreydanus":
        cfg_path = os.path.join(model_dir, "hnn_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg.pop("model_type", None)
        cfg.pop("input_type", None)
        # Support old checkpoints that used v_dim instead of p_dim
        if "v_dim" in cfg and "p_dim" not in cfg:
            cfg["p_dim"] = cfg.pop("v_dim")
        model = cls(**cfg)
        weights = torch.load(
            os.path.join(model_dir, "hnn_model.pt"),
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(weights)
        return model


def is_greydanus_dir(model_dir: str) -> bool:
    """Return True if model_dir contains an HNNGreydanus checkpoint."""
    cfg_path = os.path.join(model_dir, "hnn_config.json")
    if not os.path.exists(cfg_path):
        return False
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg.get("model_type") == "greydanus"
