# models_temphead — PERMANENT BASELINE (DO NOT OVERWRITE)

This directory is the permanent reference checkpoint for the best-performing
MoonRNN model to date. It must never be overwritten, deleted, or modified by
any training run, fine-tuning script, or deployment step.

## What's here

| File | Description |
|------|-------------|
| `gru_model.pt` | Trained GRU weights — the inference artifact |
| `gru_model_config.json` | Architecture config (`system_dim=14, state_dim=7, hidden=256, layers=2`) |
| `normalizer.pkl` | `(sys_scaler, state_scaler)` StandardScaler pair fitted on training data |
| `gru_training_history.json` | Loss/accuracy curves across 30 training epochs |
| `train_status.json` | Final training status (`val_loss=0.208, epoch=30`) |

## Performance (4-gate acceptance evaluation, strict metric)

Metric: `(gt_stable=True AND gt_habitable=True AND ml_stable=True AND ml_habitable=True)`
         / `(gt_stable=True AND gt_habitable=True)`

Inference config: AND criterion + 5% outer-HZ tolerance (ARCSINH_1_OUTER = arcsinh(1.05))

| Gate | Stable recall | Habitable recall |
|------|--------------|-----------------|
| Kepler-452b-v2 prograde | 0.964 | 0.463 |
| Kepler-452b-v2 retrograde | 0.940 | 0.632 |
| Kepler-1229b prograde | 0.849 | 0.205 |
| Kepler-1229b retrograde | 0.950 | 0.602 |
| **Average habitable recall** | | **0.476** |

## Architecture

- `system_dim=14`: SYS_COLS includes `a_inner_au`, `a_outer_au` alongside `rhill_AU`
- `state_dim=7`: `[moon_planet_dist_norm, moon_star_dist_norm, moon_temp_norm, planet_star_dist, moon_speed, planet_speed, t_frac]`
  — no binary `stable`/`habitable` flags in state (removed to eliminate binary attractor feedback loop)
  — `moon_star_dist_norm = arcsinh((moon_star_dist - a_inner) / hz_width)` — arcsinh-encoded HZ position
  — `moon_temp_norm = arcsinh((T_moon - T_cold) / temp_width)` — second independent habitability encoding
- `hidden=256, layers=2, rnn_type=gru`
- `TARGET_FLAG_COLS`: `[stable, habitable, habitable_from_temp]` — 3 BCE heads
- `TARGET_DIST_COLS`: `[moon_planet_dist_norm, moon_star_dist_norm, moon_temp_norm, planet_star_dist, moon_speed, planet_speed]` — 6 MSE targets

## Training provenance

- Trained Jun 22, 2026 on a now-overwritten local dataset
- Dataset: intermediate version of run_ml_dataset.py (pre-Stage-C)
  — Parameter ranges: ms/rs 0.4–2.0 solar, Ts 3000–12000 K, ap_AU 0.2–3.5
  — am_hill: 0.05–0.80 (fixed range, pre-Roche-limit)
  — mm_earth: capped at min(mp × 0.30, 0.50 M_earth)
  — Pure LHS, no conditional HZ sampling
  — freeze-post-escape: froze to first out-of-Hill-sphere value (pre-rhill×(1+1e-6) fix)
- 30 epochs, best val_loss=0.208 at epoch 30, flag_acc≈0.988

## Rules

- All future retraining writes to `models/` (default in train.py) or a new named directory
- This directory is NEVER the `--out` target for any training or fine-tuning run
- To revert production to this baseline: copy all files here to `models/`
- The `models/` directory is the live inference target for the web app and agent service
