"""
exomoon/ml/hnn_model_hill.py  —  Hill-frame Hamiltonian Neural Network.

Root-cause fix for inertial-frame HNN catastrophic tidal cancellation.

In the inertial frame (Track 1 / hnn_model.py), the star applies ~230-240 AU/yr²
to both planet AND moon.  The tidal force that governs moon dynamics is their
difference ≈ 2 AU/yr².  Any ~5% differential error in predicted star forces gives
~665% tidal error → catastrophic moon ejection.

In the Hill frame (co-rotating with the planet's orbital motion, centred on the
planet), the star's gravity appears as a direct primary tidal term in the effective
potential  (~Ω²×x_h²),  not as a difference of two large forces.  The HNN sees the
moon position q_h in the rotating frame and its gradient gives the tidal force
directly — no large-number subtraction at inference.

Input to HNN_Hill:
  q_h_norm : (batch, 3)   [x_h/rhill, y_h/rhill, z_h/rhill]  normalised Hill position
  sys_enc  : (batch, 6)   log-standardised system params (includes Ω)

Output: scalar V_θ  (dimensionless, normalised by V_ref = G·ms·mp/ap)

Physical conservative force on moon in Hill frame:
    F_cons_h = −∂V_θ/∂q_h_norm × V_ref / rhill    [Msun·AU/yr²]
    a_cons_h = F_cons_h / mm_msun                   [AU/yr²]

Coriolis (velocity-dependent) is NOT in V_θ — added analytically during integration.
Centrifugal IS encoded in V_θ through the training targets.

Isolation guarantee:  never touches models/, models_temphead/, models_hnn_log/,
  models_hnn_tidal/, models_hnn_dist_v12/, eval_aux_mlp_output/, models_force_mlp/,
  models_hnn_greydanus/, models_hnn_greydanus_optionC/, models_hnn_greydanus_maskedpm/.
  Always writes to models_hnn_hill/ (or user-specified --out).
"""

import json
import os

import torch
import torch.nn as nn
from torch import Tensor

Q_DIM_HILL   = 3   # Hill-frame position: (x_h/rhill, y_h/rhill, z_h/rhill)
SYS_DIM_HILL = 6   # ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU, Omega_yr


class HNN_Hill(nn.Module):
    """
    Hill-frame HNN.

    Architecture mirrors HNN in hnn_model.py (feedforward + Tanh) but with a
    3D position vector input instead of 3 pair-distance scalars, and a 6D sys_enc
    (includes instantaneous Ω) instead of 5D.

    Input dim: 3 + 6 = 9.
    Output: scalar V_θ(q_h_norm, sys_enc).
    """

    def __init__(
        self,
        hidden:  int   = 128,
        layers:  int   = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden  = hidden
        self.layers  = layers
        self.dropout = dropout

        in_dim = Q_DIM_HILL + SYS_DIM_HILL   # 9
        parts: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            parts += [nn.Linear(d, hidden), nn.Tanh()]
            if dropout > 0.0:
                parts.append(nn.Dropout(dropout))
            d = hidden
        parts.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*parts)

    def forward(self, q_h_norm: Tensor, sys_enc: Tensor) -> Tensor:
        """
        q_h_norm : (B, 3)  normalised Hill-frame coords
        sys_enc  : (B, 6)  log-standardised system params
        Returns  : (B,)    scalar dimensionless potential V_θ
        """
        x = torch.cat([q_h_norm, sys_enc], dim=-1)
        return self.net(x).squeeze(-1)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            self.state_dict(),
            os.path.join(out_dir, "hnn_hill_model.pt"),
        )
        cfg = {
            "hidden":  self.hidden,
            "layers":  self.layers,
            "dropout": self.dropout,
        }
        with open(os.path.join(out_dir, "hnn_hill_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, model_dir: str, map_location: str = "cpu") -> "HNN_Hill":
        with open(os.path.join(model_dir, "hnn_hill_config.json")) as f:
            cfg = json.load(f)
        model = cls(**cfg)
        weights = torch.load(
            os.path.join(model_dir, "hnn_hill_model.pt"),
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(weights)
        return model


def is_hill_dir(model_dir: str) -> bool:
    """Return True if model_dir contains a saved HNN_Hill checkpoint."""
    return os.path.isfile(os.path.join(model_dir, "hnn_hill_config.json"))
