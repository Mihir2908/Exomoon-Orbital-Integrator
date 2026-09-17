"""
hnn_train_hill.py  —  Training script for the Hill-frame HNN.

Trains HNN_Hill on ml_dataset.parquet.  Saves all artifacts to --out (default:
models_hnn_hill/).

ISOLATION GUARANTEE
  Never writes to models/, models_temphead/, models_hnn_log/, models_hnn_tidal/,
  models_hnn_dist_v12/, eval_aux_mlp_output/, models_force_mlp/,
  models_hnn_greydanus/, models_hnn_greydanus_optionC/, models_hnn_greydanus_maskedpm/.

Usage
-----
  cd Exomoon_orbital_integrator/src
  py hnn_train_hill.py --data ml_dataset.parquet --epochs 30 --hidden 256 --layers 3

  # Smaller test run:
  py hnn_train_hill.py --data ml_dataset.parquet --epochs 5 --hidden 128 --layers 2

Artifacts saved to --out dir
  hnn_hill_model.pt         — best model weights (lowest val_loss)
  hnn_hill_config.json      — architecture config
  hnn_hill_sys_scaler.pkl   — StandardScaler fitted on training sys_raw
  hnn_hill_train_status.json   — live progress (updated each epoch)
  hnn_hill_training_history.json  — loss curves (written on completion)

Loss function
  L_total = L_mag + LAMBDA_SIGN * L_sign

  L_mag: component-wise log-MSE on magnitudes (unchanged from original, scale-invariant).
  L_sign: normalized sign hinge per component j:
      relu(-pred_j * sign(target_j) / (|target_j| + EPS_SIGN)).mean()
  Properties:
    - Zero gradient for correctly-signed predictions (zero when pred*sign(target) > 0)
    - Positive (gradient = -sign(target)/(|target|+eps_sign)) for wrong-sign predictions
    - ZERO for zero targets: sign(0)=0 in PyTorch → product=0 → relu(0)=0 ✓
      (avoids the signed-log blow-up where target≈0 drove |pred|→1)
  EPS_SIGN=1e-3 bounds gradient magnitude at ~1000/element.
  LAMBDA_SIGN=50: makes sign loss comparable to L_mag at 50% wrong-sign rate.
"""

import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exomoon.ml.hnn_model_hill   import HNN_Hill
from exomoon.ml.hnn_dataset_hill import make_hnn_hill_splits, save_hill_sys_scaler


# ── CLI ────────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser(description="Train Hill-frame HNN")
ap.add_argument("--data",    default="ml_dataset.parquet",
                help="Path to Parquet training data")
ap.add_argument("--out",     default="models_hnn_hill",
                help="Output directory (default: models_hnn_hill)")
ap.add_argument("--epochs",  type=int,   default=30)
ap.add_argument("--batch",   type=int,   default=256)
ap.add_argument("--lr",      type=float, default=1e-3)
ap.add_argument("--hidden",  type=int,   default=128,
                help="Hidden units per layer (default 128)")
ap.add_argument("--layers",  type=int,   default=2,
                help="Number of hidden layers (default 2)")
ap.add_argument("--dropout", type=float, default=0.0)
ap.add_argument("--val_frac",type=float, default=0.20)
ap.add_argument("--seed",    type=int,   default=42)
ap.add_argument("--patience",type=int,   default=7,
                help="ReduceLROnPlateau patience (default 7)")
ap.add_argument("--warmstart", default=None,
                help="Directory to load initial weights from (warm-start continuation)")
ap.add_argument("--mono_weight", type=float, default=5.0,
                help="Weight for gradient monotonicity regulariser (0 = disabled). "
                     "Penalises repulsive radial gradient inside Hill sphere.")
ap.add_argument("--n_mono", type=int, default=128,
                help="Random Hill-sphere points sampled per batch for monotonicity penalty")
args = ap.parse_args()

# Safety: refuse to write to any protected directory
_PROTECTED = {
    "models", "models_temphead", "models_hnn_log", "models_hnn_tidal",
    "models_hnn_dist_v12", "eval_aux_mlp_output", "models_force_mlp",
    "models_hnn_greydanus", "models_hnn_greydanus_optionC",
    "models_hnn_greydanus_maskedpm",
    "models_hnn_hill",        # log-MSE checkpoint — baseline, never overwrite
    "models_hnn_hill_sign",   # signed log-MSE — failed (zero-target blow-up)
    "models_hnn_hill_hinge",  # hinge checkpoint — good baseline, never overwrite
    "models_hnn_hill_hinge2", # warm-start continuation at lr=1e-4, never overwrite
    "models_hnn_hill_hinge3", # warm-start continuation at lr=5e-5, never overwrite
    "models_hnn_hill_hinge4",             # PRODUCTION MODEL, never overwrite
    "models_hnn_hill_hinge5_fresh",       # combined dataset run, never overwrite
    "models_hnn_hill_hinge6",             # warm-start from hinge5_fresh, never overwrite
    "models_hnn_hill_hinge7",             # best val=0.252, catastrophic in eval, never overwrite
    "models_hnn_hill_hinge8_rollout",     # rollout attempt (killed ep10), never overwrite
    "models_hnn_hill_hinge8_rollout_v2",  # rollout_v2 — FAILED gate eval, protected
    "models_hnn_hill_hinge8_test",        # 1-epoch Path 1 smoke test, never overwrite
    "models_hnn_hill_pilot",              # early pilot (5 epochs, high val_loss), never overwrite
    "models_hnn_hill_rollout_v3",         # rollout_v3 — FAILED gate eval, protected
    "models_hnn_hill_hinge8_mono",        # Path 1 monotonicity — mono=0 all 30 epochs, colossal disaster, never overwrite
    "models_temphead_ss", "models_temphead_ss_prefix", "models_logfix",
}
_out_base = os.path.basename(os.path.normpath(args.out))
if _out_base in _PROTECTED:
    raise RuntimeError(
        f"Refusing to write to protected directory '{args.out}'. "
        "Use a different --out path (e.g. models_hnn_hill)."
    )

OUT_DIR = os.path.join(SRC, args.out)
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = args.data if os.path.isabs(args.data) else os.path.join(SRC, args.data)
STATUS_PATH  = os.path.join(OUT_DIR, "hnn_hill_train_status.json")
HISTORY_PATH = os.path.join(OUT_DIR, "hnn_hill_training_history.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Hill-frame HNN training")
print(f"  Data   : {DATA_PATH}")
print(f"  Out    : {OUT_DIR}")
print(f"  Device : {device}")
print(f"  Config : hidden={args.hidden}  layers={args.layers}  "
      f"epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
print()


# ── Loss function ──────────────────────────────────────────────────────────────

_EPS_MAG  = 1e-6   # magnitude log-MSE floor (near-zero components)
_EPS_SIGN = 1e-3   # sign-hinge normalization floor (≈ min typical t_hill magnitude)
_LAMBDA_SIGN = 50.0


def _monotonicity_penalty(
    model:           nn.Module,
    sys_enc_batch:   torch.Tensor,   # (B, 6)  log-standardised — reused from training batch
    n_mono:          int,
    device:          torch.device,
) -> torch.Tensor:
    """
    Penalise negative radial gradient of V inside the Hill sphere.

    Physical constraint: V must deepen toward the planet (∂V/∂r > 0 inside Hill sphere),
    so that F = -∂V/∂q_h_norm points inward (attractive).  A negative radial gradient
    (∂V/∂r < 0) gives a repulsive force — the inversion failure mode.

    Sign convention:
        radial_grad_V = (q_h_norm · ∂V/∂q_h_norm) / |q_h_norm|
        Correct:   radial_grad_V > 0  →  F points inward  (attractive)
        Inverted:  radial_grad_V < 0  →  F points outward (repulsive)
    Penalty = mean(relu(-radial_grad_V)) over n_mono random points in Hill sphere.

    Uses create_graph=True so gradients flow back through the penalty to model params.
    Does NOT supervise V to match Newtonian values — Option C compliant.
    """
    # Sample linear-uniform random positions inside unit Hill sphere.
    # Direction: normalise Gaussian noise → uniform on unit sphere surface.
    # Radius: U[0,1] linear — gives 11× more inner-orbit coverage (r<0.3)
    # vs volume-uniform r^(1/3), without excluding outer orbits.
    # Motivated by eval: all GT-stable failures occur at am_hill < 0.55;
    # linear sampling puts ~55% of mass in that region vs 17% for r^(1/3).
    n_dir = torch.randn(n_mono, 3, device=device)
    n_dir = n_dir / (n_dir.norm(dim=1, keepdim=True) + 1e-10)
    r = torch.rand(n_mono, 1, device=device)   # linear, not cubic
    q_rand = (r * n_dir).detach().requires_grad_(True)  # (n_mono, 3)

    # Sample sys_enc from current batch with replacement
    idx = torch.randint(0, sys_enc_batch.shape[0], (n_mono,), device=device)
    sys_rand = sys_enc_batch[idx].detach()  # (n_mono, 6)

    with torch.enable_grad():
        V_rand = model(q_rand, sys_rand)                               # (n_mono,)
        grad_rand = torch.autograd.grad(
            V_rand.sum(), q_rand, create_graph=True
        )[0]                                                           # (n_mono, 3)

    # Radial component: q · ∇V / |q|
    q_norm = q_rand.norm(dim=1) + 1e-10   # (n_mono,)
    radial_grad = (q_rand * grad_rand).sum(dim=1) / q_norm            # (n_mono,)

    # Penalise repulsive gradient (radial_grad < 0)
    return F.relu(-radial_grad).mean()


def _hill_force_loss(
    grad_pred:  torch.Tensor,   # (B, 3)  predicted ∂V_θ/∂q_h_norm
    t_target:   torch.Tensor,   # (B, 3)  target ∂V_θ/∂q_h_norm
) -> torch.Tensor:
    """
    L_total = L_mag + LAMBDA_SIGN * L_sign

    L_mag: component-wise log-MSE on magnitudes (scale-invariant, sign-agnostic).
    L_sign: normalized sign hinge — zero gradient for correctly-signed predictions,
            non-zero gradient for wrong-sign predictions, zero for target=0 exactly.

    Key property: sign(0) = 0 in PyTorch, so for zero targets (z-component):
        norm_agree = pred * 0 / (0 + eps_sign) = 0 → relu(0) = 0 → no sign loss.
    This avoids the signed-log-MSE blow-up where near-zero targets drove |pred|→1.
    """
    L_mag  = torch.zeros(1, device=grad_pred.device)
    L_sign = torch.zeros(1, device=grad_pred.device)
    for j in range(3):
        # Magnitude (unchanged from original)
        p_j = grad_pred[:, j].abs() + _EPS_MAG
        t_j = t_target[:, j].abs() + _EPS_MAG
        L_mag = L_mag + (torch.log(p_j) - torch.log(t_j)).pow(2).mean()
        # Sign hinge: relu(-pred * sign(target) / (|target| + eps_sign))
        norm_agree = (grad_pred[:, j] * t_target[:, j].sign()
                      / (t_target[:, j].abs() + _EPS_SIGN))
        L_sign = L_sign + F.relu(-norm_agree).mean()
    return L_mag / 3.0 + _LAMBDA_SIGN * L_sign / 3.0


# ── Data ───────────────────────────────────────────────────────────────────────

print("Loading and preprocessing data …")
t0_load = time.perf_counter()
train_ds, val_ds, sys_scaler = make_hnn_hill_splits(
    DATA_PATH, val_frac=args.val_frac, seed=args.seed
)
print(f"  Train: {len(train_ds):,} rows  Val: {len(val_ds):,} rows  "
      f"({time.perf_counter() - t0_load:.1f}s)")

train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=(device.type == "cuda"))
val_loader   = DataLoader(val_ds,   batch_size=args.batch * 4, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))

save_hill_sys_scaler(sys_scaler, OUT_DIR)
print("  Scaler saved.")


# ── Model ─────────────────────────────────────────────────────────────────────

model = HNN_Hill(
    hidden  = args.hidden,
    layers  = args.layers,
    dropout = args.dropout,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: HNN_Hill  params={n_params:,}")

if args.warmstart:
    ws_dir = args.warmstart if os.path.isabs(args.warmstart) else os.path.join(SRC, args.warmstart)
    ws_ckpt = os.path.join(ws_dir, "hnn_hill_model.pt")
    ws_state = torch.load(ws_ckpt, map_location=device)
    model.load_state_dict(ws_state)
    print(f"  Warm-start weights loaded from {ws_ckpt}")
print()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=args.patience
)


# ── Training loop ──────────────────────────────────────────────────────────────

best_val_loss = float("inf")
history = {"train_loss": [], "val_loss": [], "mono_loss": [], "epochs": args.epochs,
           "hyperparams": vars(args)}

t_start = time.perf_counter()

for epoch in range(1, args.epochs + 1):
    # ── Train ──────────────────────────────────────────────────────────────
    model.train()
    train_losses = []
    mono_losses  = []

    for q_h_norm, sys_enc, t_hill, _stable in train_loader:
        q_h_norm = q_h_norm.to(device)
        sys_enc  = sys_enc.to(device)
        t_hill   = t_hill.to(device)

        # Leaf tensor for autograd: ∂V_θ/∂q_h_norm
        q_leaf = q_h_norm.detach().requires_grad_(True)

        with torch.enable_grad():
            V    = model(q_leaf, sys_enc)
            grad = torch.autograd.grad(
                V.sum(), q_leaf, create_graph=True
            )[0]              # (B, 3)

        loss = _hill_force_loss(grad, t_hill)

        # Gradient monotonicity regulariser — penalise repulsive radial gradient
        mono_pen = torch.zeros(1, device=device)
        if args.mono_weight > 0:
            mono_pen = _monotonicity_penalty(model, sys_enc, args.n_mono, device)
            loss = loss + args.mono_weight * mono_pen

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())
        mono_losses.append(mono_pen.item())

    # ── Validate ────────────────────────────────────────────────────────────
    model.eval()
    val_losses = []

    with torch.no_grad():
        for q_h_norm, sys_enc, t_hill, _stable in val_loader:
            q_h_norm = q_h_norm.to(device)
            sys_enc  = sys_enc.to(device)
            t_hill   = t_hill.to(device)

            q_leaf = q_h_norm.detach().requires_grad_(True)
            with torch.enable_grad():
                V    = model(q_leaf, sys_enc)
                grad = torch.autograd.grad(
                    V.sum(), q_leaf, create_graph=False
                )[0]

            loss = _hill_force_loss(grad, t_hill)
            val_losses.append(loss.item())

    train_loss = float(np.mean(train_losses))
    val_loss   = float(np.mean(val_losses))
    mono_loss  = float(np.mean(mono_losses))
    elapsed    = time.perf_counter() - t_start

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["mono_loss"].append(mono_loss)

    mono_str = f"  mono={mono_loss:.5f}" if args.mono_weight > 0 else ""
    print(f"  Epoch {epoch:>3}/{args.epochs}  "
          f"train={train_loss:.5f}  val={val_loss:.5f}{mono_str}  "
          f"lr={optimizer.param_groups[0]['lr']:.2e}  "
          f"elapsed={elapsed:.0f}s")

    scheduler.step(val_loss)

    # Save best model (judged by val_loss only — mono penalty excluded from val)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save(OUT_DIR)
        print(f"    * New best saved (val={best_val_loss:.5f})")

    # Live status
    status = {
        "status":     "running",
        "epoch":      epoch,
        "total_epochs": args.epochs,
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "mono_loss":  mono_loss,
        "best_val":   best_val_loss,
        "elapsed_s":  round(elapsed, 1),
    }
    with open(STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)

# ── Done ───────────────────────────────────────────────────────────────────────

history["best_val_loss"] = best_val_loss
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=2)

status["status"] = "complete"
with open(STATUS_PATH, "w") as f:
    json.dump(status, f, indent=2)

print(f"\nDone.  best_val_loss={best_val_loss:.5f}  total={time.perf_counter()-t_start:.0f}s")
print(f"Artifacts in: {OUT_DIR}")
