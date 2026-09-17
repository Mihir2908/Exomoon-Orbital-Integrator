"""
Diagnostic-only experiment (Fix #3): does `habitable` recall improve if it gets its
own dedicated GRU hidden state instead of sharing one with distance regression,
`stable`, and `habitable_from_temp`?

This does NOT modify model.py / train.py / dataset.py, does NOT touch the deployed
models/ directory, and is not a proposed replacement for the multi-task MoonRNN --
purely a test of one specific hypothesis. Reuses make_splits()/MoonDataset unmodified
so the data, split, and normalization are identical to the deployed model's training.

Evaluated teacher-forced (true validation-set state sequence fed at every step, same
as training) -- this isolates the representational-capacity question from the
separate, already-characterized autoregressive-drift problem.

Usage:
    python _diag_habitable_only_model.py train     # trains the single-task model
    python _diag_habitable_only_model.py eval       # compares both models' habitable-only
                                                      # teacher-forced validation metrics
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from exomoon.ml.dataset import (
    SYS_COLS, STATE_COLS, TARGET_DIST_COLS, TARGET_FLAG_COLS,
    SYS_DIM, STATE_DIM, MoonDataset, make_splits, save_normalizer, load_normalizer,
)
from exomoon.ml.model import MoonRNN

HABITABLE_IDX = len(TARGET_DIST_COLS) + TARGET_FLAG_COLS.index("habitable")  # index within OUT_DIM targets
OUT_DIR = "models_habitable_only_diagnostic"
DEPLOYED_DIR = "models"


class HabitableOnlyRNN(nn.Module):
    """Same GRU backbone/hyperparams as MoonRNN, but a single output: habitable_logit."""

    def __init__(self, system_dim=SYS_DIM, state_dim=STATE_DIM, hidden=256, layers=2, rnn_type="gru"):
        super().__init__()
        self.rnn_type, self.hidden, self.layers = rnn_type, hidden, layers
        self.system_dim, self.state_dim = system_dim, state_dim
        input_dim = system_dim + state_dim
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_dim, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, state_seq, sys_params):
        batch, T, _ = state_seq.shape
        sys_exp = sys_params.unsqueeze(1).expand(-1, T, -1)
        rnn_in = torch.cat([state_seq, sys_exp], dim=-1)
        rnn_out, _ = self.rnn(rnn_in)
        return self.head(rnn_out)   # (batch, T, 1) raw logit

    def get_config(self):
        return {"system_dim": self.system_dim, "state_dim": self.state_dim,
                "hidden": self.hidden, "layers": self.layers, "rnn_type": self.rnn_type}

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_dir, "habitable_only_model.pt"))
        with open(os.path.join(out_dir, "habitable_only_model_config.json"), "w") as f:
            json.dump(self.get_config(), f, indent=2)

    @classmethod
    def load(cls, out_dir):
        with open(os.path.join(out_dir, "habitable_only_model_config.json")) as f:
            cfg = json.load(f)
        model = cls(**cfg)
        state = torch.load(os.path.join(out_dir, "habitable_only_model.pt"),
                            map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model


def train(epochs=30, batch_size=64, lr=1e-3, hidden=256, layers=2, patience=10):
    print("[habitable-only] Loading ml_dataset.parquet ...", flush=True)
    train_df, val_df, sys_scaler, state_scaler = make_splits("ml_dataset.parquet")
    print(f"[habitable-only] Train sims: {train_df['sim_id'].nunique()}  "
          f"Val sims: {val_df['sim_id'].nunique()}", flush=True)

    train_ds = MoonDataset(train_df, sys_scaler, state_scaler)
    val_ds   = MoonDataset(val_df,   sys_scaler, state_scaler)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # pos_weight for habitable specifically, computed the same way train.py does
    hab_counts = train_df["habitable"].values
    pos = hab_counts.sum()
    neg = len(hab_counts) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
    print(f"[habitable-only] habitable pos_weight: {pos_weight.item():.3f}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HabitableOnlyRNN(hidden=hidden, layers=layers).to(device)
    pos_weight = pos_weight.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, es_counter = float("inf"), 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for state_seq, sys_p, targets, _dist_mask in train_loader:
            state_seq, sys_p, targets = state_seq.to(device), sys_p.to(device), targets.to(device)
            hab_target = targets[..., HABITABLE_IDX:HABITABLE_IDX + 1]
            optimiser.zero_grad()
            logit = model(state_seq, sys_p)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, hab_target, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, correct, total, tp, fn_ = [], 0, 0, 0, 0
        with torch.no_grad():
            for state_seq, sys_p, targets, _dist_mask in val_loader:
                state_seq, sys_p, targets = state_seq.to(device), sys_p.to(device), targets.to(device)
                hab_target = targets[..., HABITABLE_IDX:HABITABLE_IDX + 1]
                logit = model(state_seq, sys_p)
                loss = nn.functional.binary_cross_entropy_with_logits(logit, hab_target, pos_weight=pos_weight)
                val_losses.append(loss.item())
                pred = (logit > 0).float()
                correct += (pred == hab_target).sum().item()
                total += hab_target.numel()
                tp += ((pred == 1) & (hab_target == 1)).sum().item()
                fn_ += ((pred == 0) & (hab_target == 1)).sum().item()

        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        acc = correct / total
        recall = tp / max(tp + fn_, 1)
        elapsed = time.time() - t0
        marker = ""
        if val_loss < best_val:
            best_val, es_counter = val_loss, 0
            model.save(OUT_DIR)
            save_normalizer(state_scaler, os.path.join(OUT_DIR, "state_normalizer.pkl"))
            save_normalizer(sys_scaler, os.path.join(OUT_DIR, "sys_normalizer.pkl"))
            marker = " *saved*"
        else:
            es_counter += 1
        print(f"  Epoch {ep:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"acc={acc:.3f}  TEACHER-FORCED habitable recall={recall:.3f}  ({elapsed:.0f}s){marker}",
              flush=True)
        if es_counter >= patience:
            print(f"[habitable-only] Early stopping at epoch {ep}.", flush=True)
            break

    print(f"[habitable-only] Done. Best val_loss={best_val:.4f}. Artifacts in {OUT_DIR}/", flush=True)


def eval_compare():
    """Compare teacher-forced habitable recall: deployed multi-task model vs this single-task model."""
    print("[eval] Loading validation split ...", flush=True)
    _, val_df, _, _ = make_splits("ml_dataset.parquet")

    # --- single-task diagnostic model ---
    diag_model = HabitableOnlyRNN.load(OUT_DIR)
    diag_state_scaler = load_normalizer(os.path.join(OUT_DIR, "state_normalizer.pkl"))
    diag_sys_scaler = load_normalizer(os.path.join(OUT_DIR, "sys_normalizer.pkl"))
    diag_ds = MoonDataset(val_df, diag_sys_scaler, diag_state_scaler)
    diag_loader = DataLoader(diag_ds, batch_size=64, shuffle=False, num_workers=0)

    tp = fn_ = tn = fp = 0
    with torch.no_grad():
        for state_seq, sys_p, targets, _ in diag_loader:
            hab_target = targets[..., HABITABLE_IDX:HABITABLE_IDX + 1]
            logit = diag_model(state_seq, sys_p)
            pred = (logit > 0).float()
            tp += ((pred == 1) & (hab_target == 1)).sum().item()
            fn_ += ((pred == 0) & (hab_target == 1)).sum().item()
            tn += ((pred == 0) & (hab_target == 0)).sum().item()
            fp += ((pred == 1) & (hab_target == 0)).sum().item()
    print(f"[SINGLE-TASK diagnostic] TP={tp} FN={fn_} TN={tn} FP={fp}  "
          f"recall={tp/max(tp+fn_,1):.3f}  precision={tp/max(tp+fp,1):.3f}  "
          f"accuracy={(tp+tn)/max(tp+fn_+tn+fp,1):.3f}")

    # --- deployed multi-task model, same validation rows ---
    deployed = MoonRNN.load(DEPLOYED_DIR, rnn_type="gru")
    deployed_sys_scaler, deployed_state_scaler = load_normalizer(os.path.join(DEPLOYED_DIR, "normalizer.pkl"))
    dep_ds = MoonDataset(val_df, deployed_sys_scaler, deployed_state_scaler)
    dep_loader = DataLoader(dep_ds, batch_size=64, shuffle=False, num_workers=0)

    tp = fn_ = tn = fp = 0
    with torch.no_grad():
        for state_seq, sys_p, targets, _ in dep_loader:
            hab_target = targets[..., HABITABLE_IDX:HABITABLE_IDX + 1]
            _, flag_logits = deployed(state_seq, sys_p)
            hab_logit = flag_logits[..., 1:2]   # TARGET_FLAG_COLS = [stable, habitable, habitable_from_temp]
            pred = (hab_logit > 0).float()
            tp += ((pred == 1) & (hab_target == 1)).sum().item()
            fn_ += ((pred == 0) & (hab_target == 1)).sum().item()
            tn += ((pred == 0) & (hab_target == 0)).sum().item()
            fp += ((pred == 1) & (hab_target == 0)).sum().item()
    print(f"[DEPLOYED multi-task]    TP={tp} FN={fn_} TN={tn} FP={fp}  "
          f"recall={tp/max(tp+fn_,1):.3f}  precision={tp/max(tp+fp,1):.3f}  "
          f"accuracy={(tp+tn)/max(tp+fn_+tn+fp,1):.3f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        train()
    elif mode == "eval":
        eval_compare()
    else:
        print("Usage: python _diag_habitable_only_model.py [train|eval]")
