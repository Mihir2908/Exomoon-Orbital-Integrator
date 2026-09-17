"""
exomoon/ml/force_mlp_model.py — Three-submodel direct force regression MLPs.

Alternative to HNN v12 for trajectory preview. HNN v12 (models_hnn_dist_v12/)
remains the primary completed trajectory model; this is a second option that
eliminates HNN's autograd chain-rule scaling problem by directly regressing
forces from distances without differentiating through a learned potential.

Three completely independent MLPs, one per body pair:
  MLP_sp: (d_sp_n, sys_enc) -> t_sp  [star-planet force magnitude]
  MLP_sm: (d_sm_n, sys_enc) -> t_sm  [star-moon force magnitude]
  MLP_pm: (d_pm_n, sys_enc) -> t_pm  [planet-moon force magnitude]

Each MLP sees only its own distance component — physically correct because
Newtonian pair forces depend only on their own pair separation, not on the
other distances.

Force targets (same dimensionless convention as hnn_dataset.py):
  t_sp = 1 / d_sp_n^2
  t_sm = (mm/mp) / d_sm_n^2
  t_pm = (mm/ms) * (ap/rhill) / d_pm_n^2

sys_enc: [ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU] — 5-dim,
  StandardScaler-normalised (identical format to HNN v12's hnn_sys_scaler.pkl).
  The rhill_AU and ap_AU entries let MLP_pm learn the (ap/rhill) coefficient
  in t_pm and MLP_sm learn the (mm/mp) coefficient in t_sm implicitly.

Output activation: Softplus (forces are always positive).
Training loss: log-MSE (scale-invariant, matches HNN v12).

ISOLATION: zero imports from dataset.py, model.py, train.py, inference.py.
Never reads/writes models/, models_temphead/, eval_aux_mlp_output/, models_hnn_dist_v12/.
"""

import json
import os

import torch
import torch.nn as nn

SYS_DIM = 5   # ms_solar, mp_earth, mm_earth, ap_AU, rhill_AU  (matches hnn_dataset.py)
D_DIM   = 3   # d_sp_n, d_sm_n, d_pm_n


class ForceMLP(nn.Module):
    """Single-pair direct force regression MLP.

    Input : (d_n_i: (B,1), sys_enc: (B, sys_dim)) -> concatenated (B, 1+sys_dim)
    Hidden: `layers` x `hidden` units with GELU activation
    Output: (B, 1) positive force magnitude via Softplus
    """

    def __init__(self, sys_dim: int = SYS_DIM, hidden: int = 128, layers: int = 3):
        super().__init__()
        in_dim = 1 + sys_dim
        dims   = [in_dim] + [hidden] * layers
        blocks = []
        for i in range(len(dims) - 1):
            blocks += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        blocks.append(nn.Linear(hidden, 1))
        self.net     = nn.Sequential(*blocks)
        self.softplus = nn.Softplus()
        self.sys_dim  = sys_dim
        self.hidden   = hidden
        self.layers   = layers

    def forward(self, d_n_i: torch.Tensor, sys_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d_n_i   : (B, 1)       normalized distance for this pair
            sys_enc : (B, sys_dim) scaled system parameters
        Returns:
            (B, 1) dimensionless force magnitude (strictly positive)
        """
        x = torch.cat([d_n_i, sys_enc], dim=-1)
        return self.softplus(self.net(x))


class ForceMLPEnsemble:
    """Container for the three per-pair force regression MLPs.

    Not an nn.Module itself — the three MLPs are independent and trained
    with a single combined optimizer in train_force_mlp.py.
    """

    PAIR_NAMES = ("sp", "sm", "pm")
    CFG_FILE   = "force_mlp_config.json"

    def __init__(self, mlp_sp: ForceMLP, mlp_sm: ForceMLP, mlp_pm: ForceMLP):
        self.mlp_sp = mlp_sp
        self.mlp_sm = mlp_sm
        self.mlp_pm = mlp_pm
        self._mlps  = [mlp_sp, mlp_sm, mlp_pm]

    # ── convenience passthrough ────────────────────────────────────────────────

    def eval(self) -> "ForceMLPEnsemble":
        for m in self._mlps:
            m.eval()
        return self

    def train(self) -> "ForceMLPEnsemble":
        for m in self._mlps:
            m.train()
        return self

    def parameters(self):
        params = []
        for m in self._mlps:
            params.extend(m.parameters())
        return params

    def __iter__(self):
        return iter(self._mlps)

    # ── batched forward ────────────────────────────────────────────────────────

    def forward(self, d_n: torch.Tensor, sys_enc: torch.Tensor) -> torch.Tensor:
        """Predict all three force components in one call.

        Args:
            d_n     : (B, 3)       [d_sp_n, d_sm_n, d_pm_n]
            sys_enc : (B, sys_dim)
        Returns:
            (B, 3) [t_sp, t_sm, t_pm] — positive force magnitudes
        """
        t_sp = self.mlp_sp(d_n[:, 0:1], sys_enc)   # (B, 1)
        t_sm = self.mlp_sm(d_n[:, 1:2], sys_enc)
        t_pm = self.mlp_pm(d_n[:, 2:3], sys_enc)
        return torch.cat([t_sp, t_sm, t_pm], dim=-1)   # (B, 3)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, out_dir: str, cfg: dict) -> None:
        os.makedirs(out_dir, exist_ok=True)
        for name, mlp in zip(self.PAIR_NAMES, self._mlps):
            torch.save(
                mlp.state_dict(),
                os.path.join(out_dir, f"force_mlp_{name}.pt"),
            )
        with open(os.path.join(out_dir, self.CFG_FILE), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> "ForceMLPEnsemble":
        cfg_path = os.path.join(model_dir, cls.CFG_FILE)
        with open(cfg_path) as f:
            cfg = json.load(f)
        mlps = []
        for name in cls.PAIR_NAMES:
            m = ForceMLP(
                sys_dim=cfg.get("sys_dim", SYS_DIM),
                hidden =cfg["hidden"],
                layers =cfg["layers"],
            )
            m.load_state_dict(
                torch.load(
                    os.path.join(model_dir, f"force_mlp_{name}.pt"),
                    map_location="cpu",
                    weights_only=True,
                )
            )
            mlps.append(m)
        return cls(*mlps)
