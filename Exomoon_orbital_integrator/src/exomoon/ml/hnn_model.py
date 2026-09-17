"""
exomoon/ml/hnn_model.py — Hamiltonian Neural Network (learned gravitational potential).

The HNN learns V_θ(d_n, sys_enc) → scalar.

Standard coordinate (tidal_coords=False):
    d_n = [d_sp_n, d_sm_n, d_pm_n]
    d_sp_n = d_sp / ap_AU   (star-planet)
    d_sm_n = d_sm / ap_AU   (star-moon)
    d_pm_n = d_pm / rhill_AU (planet-moon)

Tidal coordinate (tidal_coords=True):
    d_n = [d_sp_n, δ_tidal_n, d_pm_n]
    δ_tidal_n = (d_sm - d_sp) / rhill_AU   (signed tidal displacement)
    Replaces d_sm_n to avoid catastrophic cancellation in the tidal force.
    The star-on-moon force is learned directly from t_tidal = (mm/mp)/d_sm_n² × (rhill/ap),
    a small quantity (~t_pm magnitude), rather than as the difference of two large forces.

Training: gradient ∂V_θ/∂d_n is matched to dimensionless force targets t_grad.
Inference: chain rule through distance computation propagates forces back to
absolute coordinates; multiply by V_ref = G·ms·mp/ap to recover physical forces.

Architecture: feedforward network with Tanh activations (smooth → reliable autograd).
  Input  : d_n (3 values) + sys_enc (5 params, StandardScaler-normalised) = 8
           sys_enc = [ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU]
           ap_AU and rhill_AU are required so the model can learn the (ap/rhill)
           coefficient in t_pm = (mm/ms)·(ap/rhill)/d_pm_n².
  Hidden : layers × hidden with Tanh
  Output : 1 scalar (dimensionless potential)

ISOLATION: no imports from dataset.py, model.py, train.py, inference.py.
Never reads/writes models/, models_temphead/, eval_aux_mlp_output/.
"""

import json
import os

import torch
import torch.nn as nn
from torch import Tensor

D_DIM   = 3   # pair distances: d_sp_n, d_sm_n, d_pm_n
SYS_DIM = 5   # ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU


class HNN(nn.Module):
    """
    Learned gravitational potential V_θ(d_n, sys_enc) → scalar.

    Input is 3 physics-normalised pair distances (not absolute positions).
    This removes all translational/rotational symmetry from the input space,
    reducing the effective input dimension from 9 to 3.

    log_inputs=True: forward() applies ln(d_n) before the network.
        Training: autograd through the log correctly recovers ∂V/∂d_n_leaf
                  (the 1/d chain-rule factor cancels against the target — no
                  change to the loss or t_grad values needed).
        Inference: autograd through q_leaf → d_n → ln_d_n → V is automatic.
        sys_enc: must be StandardScaler-fitted on log(sys_raw) when log_inputs=True
                 (the caller is responsible; see hnn_dataset.make_hnn_splits).
    """

    def __init__(
        self,
        d_dim:        int             = D_DIM,
        sys_dim:      int             = SYS_DIM,   # must match dataset SYS_DIM
        hidden:       int             = 128,
        layers:       int             = 2,
        layer_sizes:  list[int] | None = None,
        dropout:      float           = 0.0,
        log_inputs:   bool            = False,
        tidal_coords: bool            = False,
    ) -> None:
        """
        layer_sizes: optional list of hidden-layer widths, e.g. [128, 64].
            Overrides hidden/layers when provided.
        dropout: probability applied after each hidden Tanh. 0.0 disables.
        log_inputs: if True, distances are log-transformed before the network.
        tidal_coords: if True, input[1] is δ_tidal_n = (d_sm - d_sp)/rhill instead of
            d_sm_n.  Because δ_tidal_n can be negative, the log transform is applied
            only to inputs[0] (d_sp_n) and inputs[2] (d_pm_n); inputs[1] is passed
            raw.  sys_enc must be fitted on log(sys_raw) when tidal_coords=True
            (same requirement as log_inputs=True).  Mutually exclusive with log_inputs
            — set exactly one.
        """
        super().__init__()
        self.d_dim        = d_dim
        self.sys_dim      = sys_dim
        self.layer_sizes  = layer_sizes
        self.dropout      = dropout
        self.log_inputs   = log_inputs
        self.tidal_coords = tidal_coords

        in_dim = d_dim + sys_dim   # = 8 (3 distances + 5 sys params)
        parts  = []
        d      = in_dim

        if layer_sizes is not None:
            for h in layer_sizes:
                parts += [nn.Linear(d, h), nn.Tanh()]
                if dropout > 0.0:
                    parts.append(nn.Dropout(dropout))
                d = h
            self.hidden = layer_sizes[0]
            self.layers = len(layer_sizes)
        else:
            self.hidden = hidden
            self.layers = layers
            for _ in range(layers):
                parts += [nn.Linear(d, hidden), nn.Tanh()]
                if dropout > 0.0:
                    parts.append(nn.Dropout(dropout))
                d = hidden

        parts.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*parts)

    def forward(self, d_n: Tensor, sys_enc: Tensor) -> Tensor:
        """
        Compute scalar potential.

        d_n     : (batch, 3)  physics-normalised pair distances (or tidal coords)
        sys_enc : (batch, 5)  StandardScaler-normalised params
                              (fitted on log(sys_raw) when log_inputs or tidal_coords)
        Returns : (batch,)    V_θ scalar per sample

        tidal_coords=True: d_n = [d_sp_n, δ_tidal_n, d_pm_n].
            log applied to slots 0 and 2 only; slot 1 is raw (can be negative).
        log_inputs=True: log applied to all three slots (original behaviour).
        """
        if self.tidal_coords:
            # Partial log: slots 0 (d_sp_n > 0) and 2 (d_pm_n > 0) are log-transformed;
            # slot 1 (δ_tidal_n, signed) is passed through raw.
            d_sp_log = torch.log(d_n[:, 0:1].clamp(min=1e-8))
            d_tidal  = d_n[:, 1:2]   # raw signed value, |δ_tidal_n| <= am_hill <= 1
            d_pm_log = torch.log(d_n[:, 2:3].clamp(min=1e-8))
            d_in = torch.cat([d_sp_log, d_tidal, d_pm_log], dim=-1)
        elif self.log_inputs:
            d_in = torch.log(d_n.clamp(min=1e-8))
        else:
            d_in = d_n
        x = torch.cat([d_in, sys_enc], dim=-1)
        return self.net(x).squeeze(-1)

    @property
    def use_log_sys_enc(self) -> bool:
        """True when sys_enc must be fitted/applied on log(sys_raw)."""
        return self.log_inputs or self.tidal_coords

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_dir, "hnn_model.pt"))
        cfg = {
            "d_dim":        self.d_dim,
            "sys_dim":      self.sys_dim,
            "hidden":       self.hidden,
            "layers":       self.layers,
            "layer_sizes":  self.layer_sizes,
            "dropout":      self.dropout,
            "log_inputs":   self.log_inputs,
            "tidal_coords": self.tidal_coords,
        }
        with open(os.path.join(out_dir, "hnn_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, model_dir: str, map_location: str = "cpu") -> "HNN":
        with open(os.path.join(model_dir, "hnn_config.json")) as f:
            cfg = json.load(f)
        # Backward-compat defaults for keys absent in older checkpoints
        cfg.setdefault("log_inputs",   False)
        cfg.setdefault("tidal_coords", False)
        model = cls(**cfg)
        weights = torch.load(
            os.path.join(model_dir, "hnn_model.pt"),
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(weights)
        return model
