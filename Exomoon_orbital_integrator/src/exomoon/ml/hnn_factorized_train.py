"""
exomoon/ml/hnn_factorized_train.py — Train the factorized HNN gravitational potential.

Same training objective as hnn_train.py:
    autograd.grad(V_total, d_n_leaf) ≈ t_grad = [t_sp, t_sm, t_pm]

Key difference: V_total = V_sp + V_sm + V_pm (HNNFactorized) so cross-partial
derivatives are zero by construction — t_pm is learned by V_pm alone without
interference from the ~200× larger t_sp signal in shared layers.

Usage
-----
    py -m exomoon.ml.hnn_factorized_train \\
        --data ml_dataset.parquet \\
        --out  models_hnn_dist_factorized/ \\
        --epochs 60 --batch 512 --lr 1e-3 --hidden 64 --layers 3

Warmstart (continue from checkpoint):
    py -m exomoon.ml.hnn_factorized_train \\
        --data ml_dataset.parquet \\
        --out  models_hnn_dist_factorized_v2/ \\
        --epochs 30 --lr 2.5e-4 \\
        --warmstart_dir models_hnn_dist_factorized/ \\
        --epoch_offset 60

Outputs (never touches models/, models_temphead/, eval_aux_mlp_output/):
    hnn_factorized_model.pt           — best weights by val_loss
    hnn_factorized_config.json        — architecture config
    hnn_sys_scaler.pkl                — StandardScaler for sys_params
    hnn_factorized_train_status.json  — live progress updated each epoch
    hnn_factorized_training_history.json — final loss curves

ISOLATION: no imports from dataset.py, model.py, train.py, inference.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from exomoon.ml.hnn_dataset          import make_hnn_splits, save_sys_scaler
from exomoon.ml.hnn_factorized_model import HNNFactorized


def train(
    data_path:     str,
    out_dir:       str,
    epochs:        int   = 60,
    batch_size:    int   = 512,
    lr:            float = 1e-3,
    weight_decay:  float = 0.0,
    hidden:        int   = 64,
    layers:        int   = 3,
    val_frac:      float = 0.20,
    seed:          int   = 42,
    device:        str   = "cpu",
    warmstart_dir: str | None = None,
    epoch_offset:  int   = 0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print("HNN training (factorized)")
    print(f"  data : {data_path}")
    print(f"  out  : {out_dir}")
    print(f"  arch : V_sp=hardcoded  +  C_sm({hidden}×{layers})  +  C_pm({hidden}×{layers})")
    print(f"  optim: epochs={epochs}  batch={batch_size}  lr={lr}  weight_decay={weight_decay}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    train_ds, val_ds, sys_scaler = make_hnn_splits(data_path, val_frac=val_frac, seed=seed)
    save_sys_scaler(sys_scaler, out_dir)
    print(f"  hnn_sys_scaler.pkl saved to {out_dir}/")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device != "cpu"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────────────
    model   = HNNFactorized(hidden=hidden, layers=layers).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_param:,} trainable parameters  "
          f"(V_sp hardcoded, C_sm_net + C_pm_net each {hidden}×{layers})")

    if warmstart_dir:
        wpath = os.path.join(warmstart_dir, "hnn_factorized_model.pt")
        model.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
        print(f"  Warmstarted from: {wpath}")
    else:
        # v5 physics-shaped architecture: forces are ALWAYS positive by construction
        # (∂V_sp/∂d = 1/d² > 0; ∂V_sm/∂d = softplus(C_sm)/d² > 0 by softplus;
        #  ∂V_pm/∂d = softplus(C_pm)/d² > 0 by softplus).
        # No sign-flip initialisation needed — softplus guarantees positive C values
        # regardless of the sign of the pre-softplus output.
        print("  Init: forces always positive by softplus construction (no sign-flip needed)")

    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Standard log-MSE.  With the v5 physics-shaped architecture forces are
        # always positive (softplus guarantees C_sm, C_pm > 0; 1/d > 0), so
        # pred is always positive and clamp(1e-12) will never activate.
        return (torch.log(pred.clamp(min=1e-12)) - torch.log(target)).pow(2).mean()

    # ── Training loop ──────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [], "epochs": 0,
        "hyperparams": {"hidden": hidden, "layers": layers, "lr": lr, "batch_size": batch_size},
    }
    best_val = float("inf")
    t_start  = time.time()

    print(f"\n{'Epoch':>6}  {'train':>10}  {'val':>10}  {'lr':>8}  {'t (s)':>7}")
    print("-" * 48)

    for epoch in range(1, epochs + 1):

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_total, n_train = 0.0, 0
        for d_n, sys_enc, t_grad in train_loader:
            d_n     = d_n.to(device)
            sys_enc = sys_enc.to(device)
            t_grad  = t_grad.to(device)

            opt.zero_grad()
            d_n_leaf = d_n.detach().requires_grad_(True)
            V        = model(d_n_leaf, sys_enc)
            grad     = torch.autograd.grad(V.sum(), d_n_leaf, create_graph=True)[0]

            loss = loss_fn(grad, t_grad)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            train_total += loss.item()
            n_train     += 1

        train_loss = train_total / max(n_train, 1)

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_total, n_val = 0.0, 0
        with torch.enable_grad():
            for d_n, sys_enc, t_grad in val_loader:
                d_n     = d_n.to(device)
                sys_enc = sys_enc.to(device)
                t_grad  = t_grad.to(device)

                d_n_leaf = d_n.detach().requires_grad_(True)
                V        = model(d_n_leaf, sys_enc)
                grad     = torch.autograd.grad(V.sum(), d_n_leaf, create_graph=False)[0]

                val_total += loss_fn(grad, t_grad).item()
                n_val     += 1

        val_loss = val_total / max(n_val, 1)
        sched.step(val_loss)

        elapsed = time.time() - t_start
        cur_lr  = opt.param_groups[0]["lr"]
        print(f"{epoch + epoch_offset:>6d}  {train_loss:>10.5f}  {val_loss:>10.5f}  "
              f"{cur_lr:>8.1e}  {elapsed:>7.1f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        status = {"status": "running", "epoch": epoch, "total_epochs": epochs,
                  "train_loss": train_loss, "val_loss": val_loss, "elapsed_s": elapsed}
        with open(os.path.join(out_dir, "hnn_factorized_train_status.json"), "w") as f:
            json.dump(status, f)

        if val_loss < best_val:
            best_val = val_loss
            model.save(out_dir)
            print(f"         * saved best model (val={best_val:.5f})")

    # ── Finalise ───────────────────────────────────────────────────────────────
    history["epochs"] = epochs
    with open(os.path.join(out_dir, "hnn_factorized_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    status["status"] = "complete"
    with open(os.path.join(out_dir, "hnn_factorized_train_status.json"), "w") as f:
        json.dump(status, f)

    print(f"\nTraining complete.  Best val_loss = {best_val:.5f}")
    print(f"Model and config saved to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train factorized HNN gravitational potential")
    ap.add_argument("--data",          required=True)
    ap.add_argument("--out",           default="models_hnn_dist_factorized/")
    ap.add_argument("--epochs",        type=int,   default=60)
    ap.add_argument("--batch",         type=int,   default=512)
    ap.add_argument("--lr",            type=float, default=1e-3)
    ap.add_argument("--weight_decay",  type=float, default=0.0)
    ap.add_argument("--hidden",        type=int,   default=64)
    ap.add_argument("--layers",        type=int,   default=3)
    ap.add_argument("--device",        default="cpu")
    ap.add_argument("--warmstart_dir", default=None)
    ap.add_argument("--epoch_offset",  type=int,   default=0)
    args = ap.parse_args()

    train(
        data_path=args.data, out_dir=args.out, epochs=args.epochs,
        batch_size=args.batch, lr=args.lr, weight_decay=args.weight_decay,
        hidden=args.hidden, layers=args.layers, device=args.device,
        warmstart_dir=args.warmstart_dir, epoch_offset=args.epoch_offset,
    )


if __name__ == "__main__":
    main()
