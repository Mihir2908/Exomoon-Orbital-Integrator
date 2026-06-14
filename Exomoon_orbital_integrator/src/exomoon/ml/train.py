"""
exomoon/ml/train.py — Training CLI and callable function for MoonRNN.

CLI usage:
    python -m exomoon.ml.train \
        --data ml_dataset.parquet \
        --epochs 30 --batch 64 --lr 1e-3 \
        --hidden 256 --layers 2 --rnn_type gru \
        --out models/

Saves to out/:
    gru_model.pt          — model weights
    model_config.json     — architecture config (loaded by inference)
    normalizer.pkl        — (sys_scaler, state_scaler) tuple
    training_history.json — loss curves per epoch
    train_status.json     — live progress (updated each epoch, used by /ml/train/status)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# ── allow running as python -m exomoon.ml.train from src/ ────────────────────
_src = os.path.join(os.path.dirname(__file__), "..", "..")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

from exomoon.ml.dataset import (
    make_splits, MoonDataset, save_normalizer,
    SYS_COLS, STATE_COLS, TARGET_COLS, _safe_cols,
)
from exomoon.ml.model import MoonRNN


def _write_status(out_dir: str, status: dict) -> None:
    path = os.path.join(out_dir, "train_status.json")
    with open(path, "w") as f:
        json.dump(status, f)


def train(
    data_path:  str,
    out_dir:    str   = "models/",
    epochs:     int   = 30,
    batch_size: int   = 64,
    lr:         float = 1e-3,
    hidden:     int   = 256,
    layers:     int   = 2,
    rnn_type:   str   = "gru",
    val_frac:   float = 0.20,
    seed:       int   = 42,
    verbose:    bool  = True,
    status_cb           = None,   # optional callable(epoch, total, train_loss, val_loss)
) -> dict:
    """
    Train MoonRNN and save all artefacts to out_dir.

    Returns the training history dict.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        raise RuntimeError("PyTorch required — pip install torch --index-url https://download.pytorch.org/whl/cpu")

    os.makedirs(out_dir, exist_ok=True)

    # Status: starting
    _write_status(out_dir, {"status": "running", "epoch": 0, "total_epochs": epochs,
                            "train_loss": None, "val_loss": None})

    if verbose:
        print(f"[train] Loading {data_path} …", flush=True)

    train_df, val_df, sys_scaler, state_scaler = make_splits(
        data_path, val_frac=val_frac, seed=seed
    )
    save_normalizer((sys_scaler, state_scaler), os.path.join(out_dir, "normalizer.pkl"))

    if verbose:
        n_train = train_df["sim_id"].nunique()
        n_val   = val_df["sim_id"].nunique()
        print(f"[train] Train sims: {n_train}  |  Val sims: {n_val}", flush=True)

    train_ds = MoonDataset(train_df, sys_scaler, state_scaler)
    val_ds   = MoonDataset(val_df,   sys_scaler, state_scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[train] Device: {device}  |  rnn_type={rnn_type}  hidden={hidden}  layers={layers}", flush=True)

    model = MoonRNN(rnn_type=rnn_type, hidden=hidden, layers=layers).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )

    history = {
        "train_loss":          [],
        "val_loss":            [],
        "flag_accuracy":       [],   # validation flag accuracy
        "flag_accuracy_train": [],   # training flag accuracy
        "dist_mae":            [],
        "hyperparams": {
            "rnn_type": rnn_type, "hidden": hidden, "layers": layers,
            "lr": lr, "batch_size": batch_size, "epochs": epochs,
        },
    }

    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        train_flag_correct, train_flag_total = 0, 0
        for state_seq, sys_p, targets in train_loader:
            state_seq = state_seq.to(device)
            sys_p     = sys_p.to(device)
            targets   = targets.to(device)

            optimiser.zero_grad()
            dist_pred, flag_logits = model(state_seq, sys_p)
            loss, _ = MoonRNN.compute_loss(dist_pred, flag_logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())
            with torch.no_grad():
                flag_pred_t = (flag_logits.detach() > 0).float()
                train_flag_correct += (flag_pred_t == targets[..., -2:]).sum().item()
                train_flag_total   += targets[..., -2:].numel()

        train_loss     = float(np.mean(train_losses))
        train_flag_acc = train_flag_correct / max(1, train_flag_total)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        val_losses, flag_correct, flag_total, dist_maes = [], 0, 0, []
        with torch.no_grad():
            for state_seq, sys_p, targets in val_loader:
                state_seq = state_seq.to(device)
                sys_p     = sys_p.to(device)
                targets   = targets.to(device)

                dist_pred, flag_logits = model(state_seq, sys_p)
                loss, _ = MoonRNN.compute_loss(dist_pred, flag_logits, targets)
                val_losses.append(loss.item())

                # Flag accuracy
                flag_pred = (flag_logits > 0).float()
                flag_gt   = targets[..., -2:]
                flag_correct += (flag_pred == flag_gt).sum().item()
                flag_total   += flag_gt.numel()

                # Distance MAE (AU)
                dist_mae = (dist_pred - targets[..., :5]).abs().mean().item()
                dist_maes.append(dist_mae)

        val_loss    = float(np.mean(val_losses))
        flag_acc    = flag_correct / max(1, flag_total)
        dist_mae_ep = float(np.mean(dist_maes))

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["flag_accuracy"].append(flag_acc)
        history["flag_accuracy_train"].append(train_flag_acc)
        history["dist_mae"].append(dist_mae_ep)
        history["epochs"] = epoch

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            model.save(out_dir)

        # Update live status
        status = {
            "status":      "running",
            "epoch":       epoch,
            "total_epochs": epochs,
            "train_loss":  round(train_loss, 6),
            "val_loss":    round(val_loss, 6),
            "flag_acc":    round(flag_acc, 4),
            "elapsed_s":   round(time.time() - t0, 1),
        }
        _write_status(out_dir, status)

        if status_cb:
            status_cb(epoch, epochs, train_loss, val_loss)

        if verbose:
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"flag_acc={flag_acc:.3f}  dist_mae={dist_mae_ep:.5f} AU  "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )

    # Write final training history
    hist_path = os.path.join(out_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    _write_status(out_dir, {
        "status": "complete", "epoch": epochs, "total_epochs": epochs,
        "train_loss": history["train_loss"][-1],
        "val_loss":   history["val_loss"][-1],
        "elapsed_s":  round(time.time() - t0, 1),
    })

    if verbose:
        print(f"[train] Done — best val_loss={best_val:.4f}  artefacts in {out_dir}/", flush=True)

    return history


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MoonRNN on generated dataset")
    parser.add_argument("--data",     required=True,         help="Path to ml_dataset.parquet")
    parser.add_argument("--out",      default="models/",     help="Output directory for artefacts")
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--hidden",   type=int,   default=256)
    parser.add_argument("--layers",   type=int,   default=2)
    parser.add_argument("--rnn_type", default="gru", choices=["gru", "lstm"])
    parser.add_argument("--val_frac", type=float, default=0.20)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    train(
        data_path  = args.data,
        out_dir    = args.out,
        epochs     = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        hidden     = args.hidden,
        layers     = args.layers,
        rnn_type   = args.rnn_type,
        val_frac   = args.val_frac,
        seed       = args.seed,
        verbose    = not args.quiet,
    )
