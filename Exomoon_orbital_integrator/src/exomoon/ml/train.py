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
    make_splits, MoonDataset, save_normalizer, load_normalizer,
    SYS_COLS, STATE_COLS, TARGET_COLS, TARGET_FLAG_COLS, TARGET_DIST_COLS, _safe_cols,
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
    patience:   int   = 10,       # early stopping: epochs without min_delta improvement
    min_delta:  float = 1e-4,     # minimum val_loss improvement to reset patience counter
    input_noise_scale: float = 0.0,   # 0.0 = disabled, reproduces prior (no-noise) behaviour exactly
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
                              num_workers=0, pin_memory=False,
                              generator=torch.Generator().manual_seed(seed))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[train] Device: {device}  |  rnn_type={rnn_type}  hidden={hidden}  layers={layers}", flush=True)

    # Per-flag-column BCE pos_weight, derived from the train split's own class balance.
    # habitable=1 is rare (LHS samples Ts/rs_solar/ap_AU independently, so most sampled
    # systems never put the planet near their star's HZ at all) -- without this, plain
    # BCE collapses to predicting habitable=0 almost everywhere.
    flag_counts = train_df[TARGET_FLAG_COLS].values
    pos_counts  = flag_counts.sum(axis=0)
    neg_counts  = flag_counts.shape[0] - pos_counts
    pos_weight_np = neg_counts / np.clip(pos_counts, 1, None)
    flag_pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    if verbose:
        print(f"[train] Flag pos_weight {TARGET_FLAG_COLS}: {pos_weight_np.round(3).tolist()}", flush=True)

    # Input noise injection (training only, never validation): perturbs the first
    # N_DIST teacher-forced state columns by an amount matching the model's own
    # typical autoregressive prediction error (measured per-column MAE on the
    # current normalized features), giving it some exposure to slightly-imperfect
    # inputs during training. Cheap proxy for full scheduled sampling -- doesn't
    # require abandoning the fully-parallel nn.GRU forward pass. input_noise_scale=0.0
    # disables this entirely and reproduces the exact prior (no-noise) behaviour.
    # Recalibrated against models_asinh after replacing moon_star_dist_norm with
    # arcsinh((moon_star_dist - a_inner_au) / hz_width) -- its MAE dropped from 0.084
    # to 0.038 (a real precision gain from compressing the long tail, not just a
    # different run), so the pre-asinh noise array would over-inject noise on that
    # column by >2x. moon_speed/planet_speed (untouched by the asinh change) shifted
    # by only ~2-3%, confirming the difference reflects the feature fix, not run noise.
    # moon_temp_norm (index 2) has no measured MAE yet -- it's a new column (added
    # alongside the second habitability head) -- placeholder reuses the moon_star_dist_norm
    # value since both are arcsinh-compressed HZ-band-position quantities of similar scale;
    # remeasure with _diag_percolumn_mae.py before relying on this for an actual noise run.
    _RAW_NOISE_STD = np.array([0.04774, 0.03837, 0.03837, 0.03575, 0.35333, 0.08633], dtype=np.float32)
    if input_noise_scale > 0:
        n = len(_RAW_NOISE_STD)
        state_noise_std_np = np.zeros(len(STATE_COLS), dtype=np.float32)
        state_noise_std_np[:n] = (_RAW_NOISE_STD * input_noise_scale) / state_scaler.scale_[:n]
        state_noise_std = torch.tensor(state_noise_std_np, dtype=torch.float32, device=device)
        if verbose:
            print(f"[train] Input noise injection enabled (scale={input_noise_scale}): "
                  f"standardized-space std={state_noise_std_np.round(4).tolist()}", flush=True)
    else:
        state_noise_std = None

    torch.manual_seed(seed)
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
            "flag_pos_weight": dict(zip(TARGET_FLAG_COLS, pos_weight_np.round(3).tolist())),
            "input_noise_scale": input_noise_scale,
        },
    }

    best_val      = float("inf")
    es_counter    = 0
    stopped_early = False
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        train_flag_correct, train_flag_total = 0, 0
        for state_seq, sys_p, targets, dist_mask in train_loader:
            state_seq = state_seq.to(device)
            sys_p     = sys_p.to(device)
            targets   = targets.to(device)
            dist_mask = dist_mask.to(device)

            if state_noise_std is not None:
                state_seq = state_seq + torch.randn_like(state_seq) * state_noise_std

            optimiser.zero_grad()
            dist_pred, flag_logits = model(state_seq, sys_p)
            loss, _ = MoonRNN.compute_loss(dist_pred, flag_logits, targets, dist_mask, flag_pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())
            with torch.no_grad():
                flag_pred_t = (flag_logits.detach() > 0).float()
                train_flag_correct += (flag_pred_t == targets[..., -3:]).sum().item()
                train_flag_total   += targets[..., -3:].numel()

        train_loss     = float(np.mean(train_losses))
        train_flag_acc = train_flag_correct / max(1, train_flag_total)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        val_losses, flag_correct, flag_total, dist_maes = [], 0, 0, []
        with torch.no_grad():
            for state_seq, sys_p, targets, dist_mask in val_loader:
                state_seq = state_seq.to(device)
                sys_p     = sys_p.to(device)
                targets   = targets.to(device)
                dist_mask = dist_mask.to(device)

                dist_pred, flag_logits = model(state_seq, sys_p)
                loss, _ = MoonRNN.compute_loss(dist_pred, flag_logits, targets, dist_mask, flag_pos_weight)
                val_losses.append(loss.item())

                # Flag accuracy
                flag_pred = (flag_logits > 0).float()
                flag_gt   = targets[..., -3:]
                flag_correct += (flag_pred == flag_gt).sum().item()
                flag_total   += flag_gt.numel()

                # Distance MAE (AU) -- only over rows where the moon is still stable;
                # post-escape rows are excluded from regression supervision entirely.
                abs_err   = (dist_pred - targets[..., :len(TARGET_DIST_COLS)]).abs()
                valid_n   = dist_mask.sum().item() * dist_pred.shape[-1]
                dist_mae  = (abs_err * dist_mask).sum().item() / max(valid_n, 1.0)
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

        # Patience counter: reset only when improvement exceeds min_delta
        if val_loss < best_val - min_delta:
            es_counter = 0
        else:
            es_counter += 1

        # Save best model (no min_delta threshold — save any strict improvement)
        if val_loss < best_val:
            best_val = val_loss
            model.save(out_dir)

        # Update live status
        status = {
            "status":       "running",
            "epoch":        epoch,
            "total_epochs": epochs,
            "train_loss":   round(train_loss, 6),
            "val_loss":     round(val_loss, 6),
            "flag_acc":     round(flag_acc, 4),
            "elapsed_s":    round(time.time() - t0, 1),
        }
        _write_status(out_dir, status)

        if status_cb:
            status_cb(epoch, epochs, train_loss, val_loss)

        if verbose:
            es_info = f"  [no improvement {es_counter}/{patience}]" if es_counter > 0 else ""
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"flag_acc={flag_acc:.3f}  dist_mae={dist_mae_ep:.5f} AU  "
                f"({time.time()-t0:.0f}s){es_info}",
                flush=True,
            )

        if es_counter >= patience:
            stopped_early = True
            if verbose:
                print(f"[train] Early stopping at epoch {epoch} — no improvement for {patience} epochs.", flush=True)
            break

    history["stopped_early"] = stopped_early

    # Write final training history keyed by model type
    hist_path = os.path.join(out_dir, f"{rnn_type}_training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    _write_status(out_dir, {
        "status":        "complete",
        "epoch":         epoch,
        "total_epochs":  epochs,
        "train_loss":    history["train_loss"][-1],
        "val_loss":      history["val_loss"][-1],
        "elapsed_s":     round(time.time() - t0, 1),
        "stopped_early": stopped_early,
    })

    if verbose:
        print(f"[train] Done — best val_loss={best_val:.4f}  artefacts in {out_dir}/", flush=True)

    return history


def finetune_scheduled_sampling(
    data_path:       str,
    warm_start_dir:  str   = "models/",
    out_dir:         str   = "models_scheduled_sampling/",
    epochs:          int   = 25,
    batch_size:      int   = 64,
    lr:              float = 1e-4,
    seq_stride:      int   = 4,      # downsample 1000 -> 250 steps for the slow loop
    k_end:           int   = 1,      # teacher-forced prefix length at ramp_epochs (1 = only t=0 is real, matches inference)
    ramp_epochs:     int   = 15,     # epochs to linearly shrink the teacher-forced prefix to k_end; held flat after
    rnn_type:        str   = "gru",
    val_frac:        float = 0.20,
    seed:            int   = 42,
    verbose:         bool  = True,
    status_cb               = None,
    patience:        int   = 5,
    min_delta:       float = 1e-3,
) -> dict:
    """
    Fine-tune a warm-started MoonRNN with scheduled sampling, using a shrinking
    teacher-forced prefix (curriculum-learning variant) rather than per-step
    probabilistic mixing (Bengio et al. 2015's original formulation).

    Unlike the fully-parallel nn.GRU forward pass in train(), this unrolls the
    sequence step by step. For a given epoch's prefix length K: steps 0..K-1 of
    the input are the true ground-truth state; steps K..T-1 are the model's own
    (differentiable) prior-step prediction fed back as the next input. K shrinks
    linearly across epochs from the full sequence length (epoch 1 — equivalent to
    ordinary teacher-forced training, a safe warm start) down to k_end=1 (only the
    true initial condition is real, every remaining step is self-generated — this
    exactly matches the real autoregressive-inference condition, unlike a low
    per-step probability which rarely produces more than a few consecutive
    self-generated steps in a row during training).

    This targets a specific failure shape found via ground-truth validation: the
    model doesn't drift gradually -- it tracks correctly for many steps, then jumps
    to a wrong value in a single step and *locks* at that wrong plateau for the
    rest of the rollout, never recovering. A per-step Bernoulli mix (the original
    implementation) almost never exposes the model to a long, uninterrupted
    self-generated stretch during training, so it rarely encounters the actual
    condition under which that lock-in occurs. A shrinking prefix does, by
    construction, once K is small.

    Early stopping is gated to epoch >= ramp_epochs: the model needs real training
    time *at* the K=k_end ceiling, not just to touch it briefly before stopping.
    An earlier per-step-probability run stopped at epoch 8 of a planned 15-epoch
    ramp, never reaching more than roughly half its target sampling intensity.

    Warm-starts from warm_start_dir's existing weights + normalizer.pkl rather than
    a fresh random init -- this is additive fine-tuning on top of the already-
    converged model, not a replacement for it. Artefacts are written to a separate
    out_dir so the warm-start checkpoint in warm_start_dir is never overwritten;
    compare the two before deciding which one to deploy. That comparison should be
    done via the ground-truth-grid diagnostics (run_ground_truth_grid.py +
    compare_ml_vs_ground_truth.py) across multiple systems and both moon directions,
    not by this function's own internal val_loss alone -- the previous run's val_loss
    judged a checkpoint "best" that improved one system/direction while regressing
    another, which val_loss alone could not reveal.

    Sequences are downsampled (every seq_stride-th resampled step) since the
    step-by-step loop loses nn.GRU's batched-sequence kernel fusion and is far
    slower per step than the parallel path. Validation always rolls out fully
    autoregressively (prefix length 1) -- that's the regime this fine-tuning
    phase is meant to fix; the teacher-forced metric from train() already looked
    excellent and did not predict the rollout collapse.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        raise RuntimeError("PyTorch required — pip install torch --index-url https://download.pytorch.org/whl/cpu")

    os.makedirs(out_dir, exist_ok=True)
    _write_status(out_dir, {"status": "running", "epoch": 0, "total_epochs": epochs,
                            "train_loss": None, "val_loss": None})

    device = torch.device("cpu")

    if verbose:
        print(f"[finetune_ss] Warm-starting from {warm_start_dir}/{rnn_type}_model.pt", flush=True)
    model = MoonRNN.load(warm_start_dir, rnn_type=rnn_type, map_location="cpu").to(device)
    sys_scaler, state_scaler = load_normalizer(os.path.join(warm_start_dir, "normalizer.pkl"))

    if verbose:
        print(f"[finetune_ss] Loading {data_path} …", flush=True)
    # Reuses make_splits purely for its train/val sim_id split logic and dropna
    # handling; the scalers it fits fresh are discarded in favour of the existing
    # normalizer.pkl above, since the warm-started weights were trained under that
    # exact feature scale.
    train_df, val_df, _, _ = make_splits(data_path, val_frac=val_frac, seed=seed)

    def _subsample(df):
        return (df.groupby("sim_id", group_keys=False)
                  .apply(lambda g: g.sort_values("t_frac").iloc[::seq_stride]))

    train_df_sub = _subsample(train_df).reset_index(drop=True)
    val_df_sub   = _subsample(val_df).reset_index(drop=True)

    train_ds = MoonDataset(train_df_sub, sys_scaler, state_scaler)
    val_ds   = MoonDataset(val_df_sub,   sys_scaler, state_scaler)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    seq_len = train_ds[0][0].shape[0]
    k_start = seq_len   # epoch 1: full teacher-forced prefix == ordinary training

    if verbose:
        n_train = train_df_sub["sim_id"].nunique()
        n_val   = val_df_sub["sim_id"].nunique()
        print(f"[finetune_ss] Train sims: {n_train}  Val sims: {n_val}  "
              f"seq_len={seq_len}  teacher-forced prefix K: {k_start}->{k_end} "
              f"over {ramp_epochs} epochs (held flat after)  lr={lr}", flush=True)

    def _K_for_epoch(ep: int) -> int:
        if ramp_epochs <= 1:
            return k_end
        frac = min(1.0, (ep - 1) / (ramp_epochs - 1))
        k = k_start - frac * (k_start - k_end)
        return max(k_end, int(round(k)))

    flag_counts = train_df_sub[TARGET_FLAG_COLS].values
    pos_counts  = flag_counts.sum(axis=0)
    neg_counts  = flag_counts.shape[0] - pos_counts
    pos_weight_np = neg_counts / np.clip(pos_counts, 1, None)
    flag_pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)

    state_mean_t  = torch.tensor(state_scaler.mean_,  dtype=torch.float32, device=device)
    state_scale_t = torch.tensor(state_scaler.scale_, dtype=torch.float32, device=device)

    def _norm_state(raw: "torch.Tensor") -> "torch.Tensor":
        # Differentiable equivalent of state_scaler.transform() -- must stay in
        # torch (not sklearn/numpy) so gradients flow back through every step
        # where the model's own prediction was fed forward as the next input.
        return (raw - state_mean_t) / state_scale_t

    def _init_hidden(batch_n: int):
        if rnn_type == "lstm":
            return (torch.zeros(model.layers, batch_n, model.hidden, device=device),
                    torch.zeros(model.layers, batch_n, model.hidden, device=device))
        return torch.zeros(model.layers, batch_n, model.hidden, device=device)

    def _rollout(state_seq: "torch.Tensor", sys_p: "torch.Tensor", K: int):
        """Step-by-step unroll with a shrinking teacher-forced prefix: steps
        0..K-1 use the true ground-truth input, steps K..T-1 self-feed the
        model's own (differentiable) prior-step prediction. K=T reproduces
        ordinary teacher-forced training; K=1 reproduces real autoregressive
        inference exactly (only the true initial condition is real).
        Replicates model.forward()'s per-column activation since model.rnn /
        .head_dist are called directly to keep hidden state h flowing continuously."""
        B, T, _ = state_seq.shape
        h = _init_hidden(B)
        sys_exp = sys_p.unsqueeze(1)
        cur_input = state_seq[:, 0:1, :]
        dist_preds, flag_logits_list = [], []
        for t in range(T):
            rnn_in = torch.cat([cur_input, sys_exp], dim=-1)
            rnn_out, h = model.rnn(rnn_in, h)
            raw_dist = model.head_dist(rnn_out)
            dist_pred_t = torch.cat([
                torch.relu(raw_dist[..., 0:1]),
                raw_dist[..., 1:3],
                torch.relu(raw_dist[..., 3:6]),
            ], dim=-1)
            flag_logits_t = model.head_flags(rnn_out)
            dist_preds.append(dist_pred_t)
            flag_logits_list.append(flag_logits_t)

            if t < T - 1:
                true_next = state_seq[:, t + 1:t + 2, :]
                if t + 1 < K:
                    cur_input = true_next
                else:
                    t_frac_next = true_next[..., -1:]
                    pred_next_raw = torch.cat([dist_pred_t, t_frac_next], dim=-1)
                    cur_input = _norm_state(pred_next_raw)

        return torch.cat(dist_preds, dim=1), torch.cat(flag_logits_list, dim=1)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "val_loss": [], "flag_accuracy": [], "dist_mae": [],
        "hyperparams": {
            "rnn_type": rnn_type, "warm_start_dir": warm_start_dir,
            "lr": lr, "batch_size": batch_size, "epochs": epochs,
            "seq_stride": seq_stride,
            "k_start": k_start, "k_end": k_end,
            "ramp_epochs": ramp_epochs,
            "flag_pos_weight": dict(zip(TARGET_FLAG_COLS, pos_weight_np.round(3).tolist())),
        },
    }

    best_val      = float("inf")
    es_counter    = 0
    stopped_early = False
    t0 = time.time()
    epoch = 0
    for epoch in range(1, epochs + 1):
        cur_K = _K_for_epoch(epoch)
        model.train()
        train_losses = []
        for state_seq, sys_p, targets, dist_mask in train_loader:
            state_seq = state_seq.to(device); sys_p = sys_p.to(device)
            targets   = targets.to(device);   dist_mask = dist_mask.to(device)

            optimiser.zero_grad()
            dist_pred, flag_logits = _rollout(state_seq, sys_p, cur_K)
            loss, _ = MoonRNN.compute_loss(dist_pred, flag_logits, targets, dist_mask, flag_pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # Fully autoregressive validation (prob=1.0) -- the regime this phase targets.
        model.eval()
        val_losses, flag_correct, flag_total, dist_maes = [], 0, 0, []
        with torch.no_grad():
            for state_seq, sys_p, targets, dist_mask in val_loader:
                state_seq = state_seq.to(device); sys_p = sys_p.to(device)
                targets   = targets.to(device);   dist_mask = dist_mask.to(device)

                dist_pred, flag_logits = _rollout(state_seq, sys_p, 1)
                loss, _ = MoonRNN.compute_loss(dist_pred, flag_logits, targets, dist_mask, flag_pos_weight)
                val_losses.append(loss.item())

                flag_pred = (flag_logits > 0).float()
                flag_gt   = targets[..., -3:]
                flag_correct += (flag_pred == flag_gt).sum().item()
                flag_total   += flag_gt.numel()

                abs_err  = (dist_pred - targets[..., :len(TARGET_DIST_COLS)]).abs()
                valid_n  = dist_mask.sum().item() * dist_pred.shape[-1]
                dist_maes.append((abs_err * dist_mask).sum().item() / max(valid_n, 1.0))

        val_loss    = float(np.mean(val_losses))
        flag_acc    = flag_correct / max(1, flag_total)
        dist_mae_ep = float(np.mean(dist_maes))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["flag_accuracy"].append(flag_acc)
        history["dist_mae"].append(dist_mae_ep)
        history["epochs"] = epoch

        # Patience counter: reset only when improvement exceeds min_delta
        if val_loss < best_val - min_delta:
            es_counter = 0
        else:
            es_counter += 1

        # Save best model by autoregressive val_loss (no min_delta threshold —
        # save any strict improvement). Unlike train()'s short single-pass usage,
        # this run can go many epochs, so silently overwriting with a worse epoch
        # is a real risk here.
        if val_loss < best_val:
            best_val = val_loss
            model.save(out_dir)

        elapsed = time.time() - t0
        status = {"status": "running", "epoch": epoch, "total_epochs": epochs,
                  "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6),
                  "flag_acc": round(flag_acc, 4), "elapsed_s": round(elapsed, 1)}
        _write_status(out_dir, status)
        if status_cb:
            status_cb(epoch, epochs, train_loss, val_loss)
        if verbose:
            es_info = f"  [no improvement {es_counter}/{patience}]" if es_counter > 0 else ""
            ramp_info = "" if epoch >= ramp_epochs else "  [ramping, ES disabled]"
            print(
                f"  [SS] Epoch {epoch:3d}/{epochs}  K={cur_K:4d}  train={train_loss:.4f}  "
                f"val(autoregressive)={val_loss:.4f}  flag_acc={flag_acc:.3f}  "
                f"dist_mae={dist_mae_ep:.5f}  ({elapsed:.0f}s){es_info}{ramp_info}",
                flush=True,
            )

        # Early stopping is gated behind the ramp completing: the model needs real
        # training time at the K=k_end ceiling, not just to touch it briefly.
        if epoch >= ramp_epochs and es_counter >= patience:
            stopped_early = True
            if verbose:
                print(f"[finetune_ss] Early stopping at epoch {epoch} — no improvement for {patience} epochs.", flush=True)
            break

    history["stopped_early"] = stopped_early

    hist_path = os.path.join(out_dir, f"{rnn_type}_training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    _write_status(out_dir, {
        "status":        "complete",
        "epoch":         history["epochs"],
        "total_epochs":  epochs,
        "train_loss":    history["train_loss"][-1],
        "val_loss":      history["val_loss"][-1],
        "elapsed_s":     round(time.time() - t0, 1),
        "stopped_early": stopped_early,
    })

    # Self-contained out_dir (model + normalizer) so predict_stability_map() can
    # point at it directly without falling back to warm_start_dir.
    save_normalizer((sys_scaler, state_scaler), os.path.join(out_dir, "normalizer.pkl"))

    if verbose:
        print(f"[finetune_ss] Done — artefacts in {out_dir}/", flush=True)

    return history


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MoonRNN on generated dataset")
    parser.add_argument("--data",      required=True,         help="Path to ml_dataset.parquet")
    parser.add_argument("--out",       default="models/",     help="Output directory for artefacts")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch",     type=int,   default=64)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--hidden",    type=int,   default=256)
    parser.add_argument("--layers",    type=int,   default=2)
    parser.add_argument("--rnn_type",  default="gru", choices=["gru", "lstm"])
    parser.add_argument("--val_frac",  type=float, default=0.20)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--patience",  type=int,   default=10,   help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum val_loss improvement to reset patience counter")
    parser.add_argument("--input_noise_scale", type=float, default=0.0,
                         help="Scale of Gaussian noise injected into teacher-forced training inputs "
                              "(0.0 disables; 1.0 = noise std matches measured per-column MAE)")
    parser.add_argument("--scheduled_sampling", action="store_true",
                         help="Fine-tune with scheduled sampling instead of running train() from scratch. "
                              "Warm-starts from --warm_start_dir; writes to --out (use a separate dir "
                              "from --warm_start_dir to avoid overwriting the warm-start checkpoint).")
    parser.add_argument("--warm_start_dir", default="models/",
                         help="Directory to load existing weights + normalizer.pkl from (scheduled sampling only)")
    parser.add_argument("--seq_stride",     type=int,   default=4,
                         help="Downsample factor for the step-by-step loop (1000/seq_stride steps per sequence)")
    parser.add_argument("--k_end",          type=int,   default=1,
                         help="Teacher-forced prefix length at --ramp_epochs (1 = only t=0 is real, matches inference)")
    parser.add_argument("--ramp_epochs",    type=int,   default=15,
                         help="Epochs to linearly shrink the teacher-forced prefix from full-length to --k_end; "
                              "held flat after. Early stopping is disabled until this many epochs have passed.")
    parser.add_argument("--quiet",     action="store_true")
    args = parser.parse_args()

    if args.scheduled_sampling:
        finetune_scheduled_sampling(
            data_path      = args.data,
            warm_start_dir = args.warm_start_dir,
            out_dir        = args.out,
            epochs         = args.epochs,
            batch_size     = args.batch,
            lr             = args.lr,
            seq_stride     = args.seq_stride,
            k_end          = args.k_end,
            ramp_epochs    = args.ramp_epochs,
            rnn_type       = args.rnn_type,
            val_frac       = args.val_frac,
            seed           = args.seed,
            verbose        = not args.quiet,
        )
    else:
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
            patience   = args.patience,
            min_delta  = args.min_delta,
            input_noise_scale = args.input_noise_scale,
        )
