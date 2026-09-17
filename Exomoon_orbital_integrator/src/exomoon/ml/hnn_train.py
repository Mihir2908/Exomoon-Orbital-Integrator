"""
exomoon/ml/hnn_train.py — Train the distance-based HNN gravitational potential.

Training objective: the HNN learns V_θ(d_n, sys_enc) such that
    ∂V_θ/∂d_n ≈ t_grad
where t_grad = [t_sp, t_sm, t_pm] are dimensionless force targets derived from
the analytical Newtonian potential (see hnn_dataset.py for target derivation).

Key training detail: computing ∂V_θ/∂d_n requires autograd.grad(V, d_n_leaf)
inside the forward pass, and back-propagating MSE(grad, t_grad) through that
gradient requires create_graph=True.

Usage
-----
    py -m exomoon.ml.hnn_train \\
        --data ml_dataset.parquet \\
        --out  models_hnn_dist/   \\
        --epochs 30 --batch 512 --lr 1e-3 --hidden 256 --layers 3

Outputs (never touches models/, models_temphead/, eval_aux_mlp_output/):
    hnn_model.pt              — best weights by val_loss
    hnn_config.json           — architecture config (required by HNN.load)
    hnn_sys_scaler.pkl        — StandardScaler for sys_params
    hnn_train_status.json     — live progress updated each epoch
    hnn_training_history.json — final loss curves

ISOLATION: no imports from dataset.py, model.py, train.py, inference.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from exomoon.ml.hnn_dataset import make_hnn_splits, make_hnn_splits_optionC, make_hnn_splits_tidal, save_sys_scaler
from exomoon.ml.hnn_model   import HNN


def train(
    data_path:     str,
    out_dir:       str,
    epochs:        int              = 30,
    batch_size:    int              = 512,
    lr:            float            = 1e-3,
    weight_decay:  float            = 0.0,
    dropout:       float            = 0.0,
    hidden:        int              = 128,
    layers:        int              = 2,
    layer_sizes:   list[int] | None = None,
    val_frac:      float            = 0.20,
    seed:          int              = 42,
    device:        str              = "cpu",
    warmstart_dir: str | None       = None,
    epoch_offset:  int              = 0,
    lambda_value:  float            = 0.0,
    log_inputs:    bool             = False,
    tidal_coords:  bool             = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    arch_str = "→".join(str(h) for h in layer_sizes) if layer_sizes else f"{hidden}×{layers}"
    use_value = lambda_value > 0.0
    label = "distance-based"
    if tidal_coords:
        label += " + tidal-coord inputs (Option 2)"
    elif log_inputs:
        label += " + log-space inputs"
    if use_value:
        label += " + Option C value supervision"
    print(f"HNN training ({label})")
    print(f"  data         : {data_path}")
    print(f"  out          : {out_dir}")
    print(f"  arch         : {arch_str}")
    print(f"  log_inputs   : {log_inputs}  tidal_coords: {tidal_coords}")
    print(f"  optim        : epochs={epochs}  batch={batch_size}  lr={lr}  "
          f"weight_decay={weight_decay}  dropout={dropout}")
    if use_value:
        print(f"  lambda_value : {lambda_value}  (L_total = L_grad + {lambda_value}·L_value)")

    # ── Dataset ────────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    if tidal_coords:
        if use_value:
            raise ValueError("--tidal_coords is incompatible with --lambda_value (Option C)")
        train_ds, val_ds, sys_scaler = make_hnn_splits_tidal(
            data_path, val_frac=val_frac, seed=seed,
        )
    elif use_value:
        train_ds, val_ds, sys_scaler = make_hnn_splits_optionC(
            data_path, val_frac=val_frac, seed=seed, log_sys_enc=log_inputs,
        )
    else:
        train_ds, val_ds, sys_scaler = make_hnn_splits(
            data_path, val_frac=val_frac, seed=seed, log_sys_enc=log_inputs,
        )
    save_sys_scaler(sys_scaler, out_dir)
    print(f"  hnn_sys_scaler.pkl saved to {out_dir}/")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model   = HNN(hidden=hidden, layers=layers, layer_sizes=layer_sizes,
                  dropout=dropout, log_inputs=log_inputs,
                  tidal_coords=tidal_coords).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_param:,} trainable parameters  (input dim = {model.d_dim + model.sys_dim})")

    if warmstart_dir:
        wpath = os.path.join(warmstart_dir, "hnn_model.pt")
        model.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
        print(f"  Warmstarted from: {wpath}")

    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=5, factor=0.5,
    )

    # Component-wise force loss:
    #   t_sp, t_sm : log-MSE on ALL rows (always finite and positive)
    #   t_pm       : log-MSE on STABLE rows only (d_pm_n <= 1)
    #
    # Why mask t_pm but not t_sm:
    #   Escaped moons have d_pm_n up to 735× rhill → t_pm → ~10^-6 → log(t_pm) → -14.
    #   Applying log-MSE to those rows sends huge erroneous gradients through t_pm.
    #   t_sm is never masked because escaped moons still orbit the star; t_sm stays
    #   finite (d_sm_n ≤ ~16) and physically meaningful even for escaped trajectories.
    #
    # Why log-MSE for all three (not MSRE):
    #   t_sm spans ~7 orders of magnitude (mm/mp varies 10^-4 × d_sm_n^2 varies 250×).
    #   MSRE with eps=0.01 becomes blind when t_sm << eps. log-MSE is scale-invariant.
    _LOG_EPS = 1e-9

    def _force_loss(
        dV_dd:  torch.Tensor,   # (batch, 3) predicted ∂V/∂d_n
        t_grad: torch.Tensor,   # (batch, 3) force targets [t_sp, t_sm, t_pm]
        stable: torch.Tensor,   # (batch,)   1.0 where d_pm_n <= 1
    ) -> torch.Tensor:
        eps = _LOG_EPS

        # t_sp and t_sm: log-MSE on all rows
        L_sp = (torch.log(dV_dd[:, 0].clamp(min=eps)) - torch.log(t_grad[:, 0].clamp(min=eps))).pow(2).mean()
        L_sm = (torch.log(dV_dd[:, 1].clamp(min=eps)) - torch.log(t_grad[:, 1].clamp(min=eps))).pow(2).mean()

        # t_pm: log-MSE on stable (bound) rows only
        pm_w    = stable.to(dV_dd.device)
        pred_pm = dV_dd[:, 2].clamp(min=eps)
        targ_pm = t_grad[:, 2].clamp(min=eps)
        sq_pm   = (torch.log(pred_pm) - torch.log(targ_pm)).pow(2)
        L_pm    = (sq_pm * pm_w).sum() / pm_w.sum().clamp(min=1.0)

        return (L_sp + L_sm + L_pm) / 3.0

    # ── Training loop ──────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_grad_loss": [], "train_val_loss_component": [],
        "epochs": 0,
        "hyperparams": {
            "hidden": hidden, "layers": layers, "layer_sizes": layer_sizes,
            "lr": lr, "batch_size": batch_size, "lambda_value": lambda_value,
            "log_inputs": log_inputs, "tidal_coords": tidal_coords,
        },
    }
    best_val = float("inf")
    t_start  = time.time()
    status   = {}

    if use_value:
        print(f"\n{'Epoch':>6}  {'train':>10}  {'val':>10}  {'L_grad':>10}  {'L_val':>10}  {'lr':>8}  {'t (s)':>7}")
        print("-" * 72)
    else:
        print(f"\n{'Epoch':>6}  {'train':>10}  {'val':>10}  {'lr':>8}  {'t (s)':>7}")
        print("-" * 48)

    for epoch in range(1, epochs + 1):

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_total      = 0.0
        train_grad_total = 0.0
        train_val_total  = 0.0
        n_train          = 0

        for batch in train_loader:
            if use_value:
                d_n, sys_enc, t_grad, stable, v_newton_n = batch
                v_newton_n = v_newton_n.to(device)
            else:
                d_n, sys_enc, t_grad, stable = batch

            d_n     = d_n.to(device)
            sys_enc = sys_enc.to(device)
            t_grad  = t_grad.to(device)
            stable  = stable.to(device)

            opt.zero_grad()

            # d_n_leaf: enables autograd.grad w.r.t. the distance inputs
            d_n_leaf = d_n.detach().requires_grad_(True)
            V        = model(d_n_leaf, sys_enc)                        # (batch,)
            grad     = torch.autograd.grad(
                V.sum(), d_n_leaf, create_graph=True,
            )[0]                                                        # (batch, 3)

            L_grad = _force_loss(grad, t_grad, stable)
            if use_value:
                L_val = (V - v_newton_n).pow(2).mean()
                loss  = L_grad + lambda_value * L_val
                train_grad_total += L_grad.item()
                train_val_total  += L_val.item()
            else:
                loss = L_grad

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            train_total += loss.item()
            n_train     += 1

        train_loss      = train_total      / max(n_train, 1)
        train_grad_loss = train_grad_total / max(n_train, 1)
        train_val_comp  = train_val_total  / max(n_train, 1)

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_total      = 0.0
        val_grad_total = 0.0
        val_val_total  = 0.0
        n_val          = 0

        with torch.enable_grad():   # autograd.grad needs grad even in eval
            for batch in val_loader:
                if use_value:
                    d_n, sys_enc, t_grad, stable, v_newton_n = batch
                    v_newton_n = v_newton_n.to(device)
                else:
                    d_n, sys_enc, t_grad, stable = batch

                d_n     = d_n.to(device)
                sys_enc = sys_enc.to(device)
                t_grad  = t_grad.to(device)
                stable  = stable.to(device)

                d_n_leaf = d_n.detach().requires_grad_(True)
                V        = model(d_n_leaf, sys_enc)
                grad     = torch.autograd.grad(
                    V.sum(), d_n_leaf, create_graph=False,
                )[0]

                L_grad = _force_loss(grad, t_grad, stable)
                if use_value:
                    L_val = (V - v_newton_n).pow(2).mean()
                    val_total += (L_grad + lambda_value * L_val).item()
                    val_grad_total += L_grad.item()
                    val_val_total  += L_val.item()
                else:
                    val_total += L_grad.item()
                n_val += 1

        val_loss      = val_total      / max(n_val, 1)
        val_grad_loss = val_grad_total / max(n_val, 1)
        val_val_comp  = val_val_total  / max(n_val, 1)

        sched.step(val_loss)

        elapsed = time.time() - t_start
        cur_lr  = opt.param_groups[0]["lr"]

        if use_value:
            print(f"{epoch + epoch_offset:>6d}  {train_loss:>10.5f}  {val_loss:>10.5f}  "
                  f"{val_grad_loss:>10.5f}  {val_val_comp:>10.5f}  "
                  f"{cur_lr:>8.1e}  {elapsed:>7.1f}")
        else:
            print(f"{epoch + epoch_offset:>6d}  {train_loss:>10.5f}  {val_loss:>10.5f}  "
                  f"{cur_lr:>8.1e}  {elapsed:>7.1f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_grad_loss"].append(train_grad_loss)
        history["train_val_loss_component"].append(train_val_comp)

        status = {
            "status":       "running",
            "epoch":        epoch,
            "total_epochs": epochs,
            "train_loss":   train_loss,
            "val_loss":     val_loss,
            "elapsed_s":    elapsed,
        }
        with open(os.path.join(out_dir, "hnn_train_status.json"), "w") as f:
            json.dump(status, f)

        if val_loss < best_val:
            best_val = val_loss
            model.save(out_dir)
            print(f"         * saved best model (val={best_val:.5f})")

    # ── Finalise ───────────────────────────────────────────────────────────────
    history["epochs"] = epochs
    with open(os.path.join(out_dir, "hnn_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    status["status"] = "complete"
    with open(os.path.join(out_dir, "hnn_train_status.json"), "w") as f:
        json.dump(status, f)

    print(f"\nTraining complete.  Best val_loss = {best_val:.5f}")
    print(f"Model and config saved to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train HNN gravitational potential (distance-based)")
    ap.add_argument("--data",         required=True,          help="Path to ml_dataset.parquet")
    ap.add_argument("--out",          default="models_hnn_dist/",
                                                               help="Output directory (never use models/ or models_temphead/)")
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch",        type=int,   default=512)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dropout",      type=float, default=0.0)
    ap.add_argument("--hidden",       type=int,   default=128)
    ap.add_argument("--layers",       type=int,   default=2)
    ap.add_argument("--layer_sizes",  type=str,   default=None,
                    help="Comma-separated hidden widths, e.g. '128,64'. Overrides --hidden/--layers.")
    ap.add_argument("--device",       default="cpu")
    ap.add_argument("--warmstart_dir", default=None,
                    help="Load best weights from this directory before training (continue from checkpoint).")
    ap.add_argument("--epoch_offset",  type=int,   default=0,
                    help="Add to epoch numbers in printed output (e.g. 60 to show epochs 61-90).")
    ap.add_argument("--lambda_value",  type=float, default=0.0,
                    help="Option C: weight on value supervision loss λ·(V_θ - V_Newton_n)². "
                         "0 = gradient-only (original). Recommended: 0.1")
    ap.add_argument("--log_inputs",    action="store_true", default=False,
                    help="Log-space Track 1: pass ln(d_n) to network and fit sys_scaler "
                         "on log(sys_raw).  Loss and targets (t_grad) are unchanged — "
                         "the chain rule through the log recovers the same ∂V/∂d_n targets. "
                         "Save to a new --out dir, e.g. models_hnn_log/.")
    ap.add_argument("--tidal_coords",  action="store_true", default=False,
                    help="Option 2 — tidal coordinate: replace d_sm_n with "
                         "δ_tidal_n=(d_sm-d_sp)/rhill as input slot 1.  Avoids "
                         "catastrophic cancellation in the tidal force.  Partial "
                         "log applied to slots 0 and 2 only; slot 1 is raw (signed). "
                         "sys_scaler always log-fitted.  Use --out models_hnn_tidal/.")
    args = ap.parse_args()

    layer_sizes = [int(x) for x in args.layer_sizes.split(",")] if args.layer_sizes else None

    train(
        data_path=args.data,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        layer_sizes=layer_sizes,
        hidden=args.hidden,
        layers=args.layers,
        device=args.device,
        warmstart_dir=args.warmstart_dir,
        epoch_offset=args.epoch_offset,
        lambda_value=args.lambda_value,
        log_inputs=args.log_inputs,
        tidal_coords=args.tidal_coords,
    )


if __name__ == "__main__":
    main()
