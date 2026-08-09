"""
models_geolife.py
=================

This module implements the full GeoLife trajectory‑forecasting pipeline:

    • Loading and preprocessing trajectories from parquet
    • Global normalization (leakage‑safe)
    • Window generation for multi‑step forecasting
    • Baseline predictor
    • GRU / LSTM / TCN models (in Part 2)
    • Training + prediction loops (in Part 2)
    • Metrics + summary tables
    • Full orchestration (in Part 2)

The rewrite emphasizes *engineering‑grade clarity*, explaining:

    • Why each architectural choice exists
    • How data flows through the pipeline
    • Why normalization is global
    • Why leakage is avoided by design
    • Why windowing is structured this way
    • Why the logging system ensures reproducibility
    • How metrics are computed and why they matter

This file is intentionally verbose to support capstone‑level documentation.
"""

# =====================================================================
# Imports
# =====================================================================
import time
import json
import os
import psutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim


# =====================================================================
# GLOBAL CONFIGURATION
# =====================================================================
"""
These configuration knobs define the behavior of the entire pipeline.
They are centralized so experiments remain reproducible and easy to tune.
"""

# Path to the unified GeoLife parquet dataset produced by geolife_pipeline.py.
PARQUET_PATH = (
    r"C:\Users\gb630\OneDrive\USD AAI\USD AAI\AAI-590 CAPSTONE\FINAL PROJECT\Data\geolife.parquet"
)

# Ordered list of feature columns extracted from the parquet file.
# Order matters because:
#   • Models assume this exact layout
#   • Normalization vectors follow this order
#   • Window generator slices based on this order
FEATURES = ["x", "y", "speed", "heading", "accel", "turn_rate"]

# Number of past timesteps fed into the model.
# This defines the receptive field for GRU/LSTM/TCN.
INPUT_WINDOW = 20

# Number of future timesteps to predict.
# Models output FUTURE_STEPS displacement vectors relative to the last input position.
FUTURE_STEPS = 5

# Device selection — CPU is used for reproducibility and simplicity.
DEVICE = "cpu"

# Heartbeat interval for long‑running loops.
HEARTBEAT_INTERVAL = 5.0

# How often to print progress during training/prediction.
PROGRESS_BATCH_INTERVAL = 10

# Threshold (meters) to detect GPS glitches.
# Any displacement > threshold is treated as invalid and filtered out.
GLITCH_THRESHOLD_METERS = 100.0

# Directories for model artifacts and logs.
MODEL_DIR = "./models"
LOG_DIR = os.path.join(MODEL_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# =====================================================================
# LOGGING SYSTEM — Dual Logging + Timestamps
# =====================================================================
"""
The logging system is designed for reproducibility and post‑run analysis.

It writes:
    • Real‑time feedback to stdout
    • A persistent log across all runs
    • A per‑run log for detailed inspection

All logs include human‑readable timestamps.
"""

run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
persistent_log_path = os.path.join(LOG_DIR, "geolife.log")
run_log_path = os.path.join(LOG_DIR, f"run_{run_timestamp}.log")


def log(msg: str) -> None:
    """
    Write high‑level events to stdout AND both log files.

    This function is used for:
        • Phase boundaries (training start/end)
        • Major milestones
        • Errors
        • Summary metrics

    Logging is UTF‑8 safe and append‑only.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {msg}"

    print(entry)

    with open(persistent_log_path, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

    with open(run_log_path, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


# =====================================================================
# Utility: Human‑Friendly ETA
# =====================================================================
def format_eta(seconds: float | None) -> str:
    """
    Convert a raw seconds estimate into a human‑friendly ETA string.

    Used in training/prediction loops to give rough time remaining.
    """
    if seconds is None or seconds <= 0:
        return "ETA: unknown"
    m, s = divmod(int(seconds), 60)
    if m == 0:
        return f"ETA: {s}s"
    return f"ETA: {m}m {s}s"


# =====================================================================
# Load and Preprocess Trajectories
# =====================================================================
def load_trajectories(path: str) -> list[np.ndarray]:
    """
    Load trajectories from a parquet file and apply glitch filtering.

    WHY THIS DESIGN:
    ----------------
    • GeoLife parquet files can be large (millions of rows).
      Reading row‑groups avoids loading everything into memory.

    • Grouping by 'source_file' reconstructs individual trajectories
      exactly as they were recorded.

    • Only FEATURES are extracted — this keeps memory usage low and
      ensures models receive consistent input.

    • GPS glitches (large jumps) are filtered using a displacement
      threshold. This prevents unrealistic velocities from corrupting
      training.

    • Trajectories too short to produce at least one window are discarded.

    RETURNS:
        List of trajectories, each shaped [num_steps, num_features].
    """
    log("[RUN] Loading parquet and building trajectories...")
    pf = pq.ParquetFile(path)
    trajectories: list[np.ndarray] = []

    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        df = tbl.to_pandas()

        # Group rows by source_file — each group is one trajectory.
        for src, traj in df.groupby("source_file"):
            # Extract configured features and cast to float32 for PyTorch.
            t = traj[FEATURES].to_numpy(dtype=np.float32)

            # Compute displacement between consecutive (x, y) positions.
            coords = t[:, 0:2]
            deltas = np.linalg.norm(coords[1:] - coords[:-1], axis=1)

            # Mask keeps first point + any point below glitch threshold.
            mask = np.concatenate([[True], deltas <= GLITCH_THRESHOLD_METERS])
            t_filtered = t[mask]

            # Skip trajectories too short for windowing.
            if len(t_filtered) <= INPUT_WINDOW + FUTURE_STEPS:
                continue

            trajectories.append(t_filtered)

    log(f"[RUN] Total trajectories after filtering: {len(trajectories)}")
    return trajectories


# =====================================================================
# Global Normalization (Leakage‑Safe)
# =====================================================================
def compute_normalization(trajectories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute global mean and std across all trajectories.

    WHY GLOBAL NORMALIZATION:
    -------------------------
    • Normalization must be computed BEFORE splitting into train/val/test
      to avoid leakage.

    • Using global stats ensures:
        - All splits share the same scale
        - Models generalize better
        - No trajectory receives privileged scaling

    RETURNS:
        mean, std — vectors shaped [num_features].
    """
    log("[RUN] Computing global normalization stats...")

    all_data = np.concatenate(trajectories, axis=0)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)

    # Prevent division‑by‑zero.
    std[std == 0] = 1.0

    log(f"[RUN] Normalization mean: {mean}")
    log(f"[RUN] Normalization std:  {std}")
    return mean, std


# =====================================================================
# Split Trajectories (Leakage‑Avoidant)
# =====================================================================
def split_trajectories(
    trajectories: list[np.ndarray],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Split trajectories into train/val/test sets at the trajectory level.

    WHY SPLIT BY TRAJECTORY:
    ------------------------
    Splitting by window would cause leakage:
        • Windows from the same trajectory could appear in multiple splits.
        • Models would implicitly "see" future behavior during training.

    Splitting by trajectory ensures:
        • Train/val/test are independent
        • No temporal leakage
        • Metrics reflect true generalization

    RETURNS:
        train, val, test — lists of trajectories.
    """
    n = len(trajectories)
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train = [trajectories[i] for i in indices[:train_end]]
    val = [trajectories[i] for i in indices[train_end:val_end]]
    test = [trajectories[i] for i in indices[val_end:]]

    log(f"[RUN] Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# =====================================================================
# Window Generator — Multi‑Step Forecasting
# =====================================================================
def stream_multistep_batches(
    trajectories: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
):
    """
    Generator that yields windowed inputs and multi‑step targets.

    WHY THIS DESIGN:
    ----------------
    • Global normalization is applied per trajectory.
    • Sliding window produces INPUT_WINDOW past steps.
    • FUTURE_STEPS future positions are converted to displacements
      relative to the last window position.

    WHY DISPLACEMENTS:
    ------------------
    Predicting absolute coordinates is harder because:
        • Trajectories vary widely in global position
        • Models must learn both motion AND geography

    Predicting displacements:
        • Centers the prediction problem
        • Makes learning easier
        • Improves generalization

    YIELDS:
        windows_tensor: [batch, features, INPUT_WINDOW]
        targets_tensor: [batch, FUTURE_STEPS, 2]
    """
    total_windows = 0
    n_traj = len(trajectories)

    for idx, t in enumerate(trajectories, start=1):
        t_norm = (t - mean) / std

        n = len(t_norm) - INPUT_WINDOW - FUTURE_STEPS
        if n <= 0:
            continue

        windows = []
        targets = []

        for i in range(n):
            # Window: shape [features, time]
            w = t_norm[i:i + INPUT_WINDOW].T

            # Future absolute positions (x, y)
            future_abs = t_norm[i + INPUT_WINDOW:i + INPUT_WINDOW + FUTURE_STEPS, 0:2]

            # Last position in window
            last_pos = t_norm[i + INPUT_WINDOW - 1, 0:2]

            # Convert to displacements
            future_disp = future_abs - last_pos

            windows.append(w)
            targets.append(future_disp)

        batch_windows = len(windows)
        total_windows += batch_windows

        print(
            f"[TRAJ {idx}/{n_traj}] {len(t)} rows → "
            f"{batch_windows} windows (total_windows={total_windows})"
        )

        yield torch.tensor(np.stack(windows)), torch.tensor(np.stack(targets))


# =====================================================================
# Baseline Predictor — Constant Velocity
# =====================================================================
def baseline_predict(windows: np.ndarray) -> np.ndarray:
    """
    Constant‑velocity baseline.

    WHY THIS BASELINE:
    ------------------
    • Provides a simple reference point.
    • Helps evaluate whether ML models actually learn meaningful motion.
    • Uses last two positions to estimate velocity and repeats it.

    RETURNS:
        preds: [batch, FUTURE_STEPS, 2]
    """
    last_pos = windows[:, 0:2, -1]
    prev_pos = windows[:, 0:2, -2]
    disp = last_pos - prev_pos
    return np.tile(disp[:, None, :], (1, FUTURE_STEPS, 1))


def evaluate_baseline(
    trajectories: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate baseline across all trajectories.

    WHY THIS DESIGN:
    ----------------
    • Reuses the same window generator as ML models.
    • Ensures baseline is evaluated under identical conditions.
    • Heartbeats provide observability during long runs.

    RETURNS:
        preds_all, targets_all — stacked arrays for metric computation.
    """
    log("[RUN] Baseline evaluation started...")
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []

    batch_count = 0
    start_time = time.time()
    last_heartbeat = start_time

    for windows_cpu, targets_cpu in stream_multistep_batches(trajectories, mean, std):
        windows_np = windows_cpu.numpy()
        targets_np = targets_cpu.numpy()

        preds_np = baseline_predict(windows_np)

        preds_all.append(preds_np)
        targets_all.append(targets_np)

        batch_count += 1
        now = time.time()

        if now - last_heartbeat >= HEARTBEAT_INTERVAL:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().used / (1024**3)
            print(
                f"[HEARTBEAT] Baseline evaluating... "
                f"batches={batch_count}, elapsed={now - start_time:.1f}s, "
                f"CPU={cpu:.1f}%, RAM={mem:.2f} GB"
            )
            last_heartbeat = now

    total_time = time.time() - start_time
    log(f"[RUN] Baseline evaluation complete — batches={batch_count}, time={total_time:.2f}s")
    return np.vstack(preds_all), np.vstack(targets_all)


# =====================================================================
# Metrics — ADE / MDE / FDE
# =====================================================================
def ade(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    ADE — Average Displacement Error.

    WHY ADE:
    --------
    Measures average Euclidean error across all timesteps.
    Captures overall trajectory accuracy.
    """
    return float(np.linalg.norm(preds - targets, axis=2).mean())


def mde(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    MDE — Maximum Displacement Error.

    WHY MDE:
    --------
    Captures worst‑case error across all timesteps.
    Useful for safety‑critical applications.
    """
    return float(np.linalg.norm(preds - targets, axis=2).max())


def fde(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    FDE — Final Displacement Error.

    WHY FDE:
    --------
    Measures error at the final timestep.
    Critical for trajectory forecasting tasks where the endpoint matters.
    """
    return float(np.linalg.norm(preds[:, -1, :] - targets[:, -1, :], axis=1).mean())


def print_metrics(name: str, preds: np.ndarray, targets: np.ndarray) -> None:
    """
    Compute ADE/MDE/FDE and print + log them in a consistent format.
    """
    msg = (
        f"{name} — ADE: {ade(preds, targets):.3f} m, "
        f"MDE: {mde(preds, targets):.3f} m, "
        f"FDE: {fde(preds, targets):.3f} m"
    )
    print(msg)
    log(f"[RUN] {msg}")


# =====================================================================
# Models — GRU / LSTM / TCN
# =====================================================================
"""
This section defines the three forecasting models used in the pipeline.

All models predict FUTURE_STEPS displacement vectors relative to the
last input position. This framing makes the learning problem easier
because models do not need to learn absolute geographic coordinates.

The three architectures provide a baseline comparison:

    • GRUModel — lightweight recurrent baseline
    • LSTMModel — more expressive recurrent model
    • TCNModel — convolutional model with dilations for long-range context

Each model preserves the original function signatures exactly.
"""


# ---------------------------------------------------------------------
# GRU Model
# ---------------------------------------------------------------------
class GRUModel(nn.Module):
    """
    GRU-based sequence model for multi-step displacement prediction.

    WHY GRU:
    --------
    • GRUs are simpler than LSTMs (fewer gates).
    • They train faster and require fewer parameters.
    • They handle moderate temporal dependencies well.

    ARCHITECTURE:
    -------------
    • Single GRU layer (batch_first=True)
    • Fully connected layer maps last hidden state → FUTURE_STEPS * 2
    • Output reshaped to [batch, FUTURE_STEPS, 2]
    """
    def __init__(self, input_size: int = 6, hidden_size: int = 64, future_steps: int = FUTURE_STEPS):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, future_steps * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_size]
        out, _ = self.gru(x)

        # Use last timestep hidden state as sequence summary.
        out = self.fc(out[:, -1, :])

        # Reshape to [batch, FUTURE_STEPS, 2]
        return out.view(-1, FUTURE_STEPS, 2)


# ---------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------
class LSTMModel(nn.Module):
    """
    LSTM-based sequence model.

    WHY LSTM:
    ---------
    • LSTMs handle longer-range dependencies than GRUs.
    • Useful when motion patterns span many timesteps.

    ARCHITECTURE:
    -------------
    • Single LSTM layer (batch_first=True)
    • Fully connected layer maps last hidden state → FUTURE_STEPS * 2
    """
    def __init__(self, input_size: int = 6, hidden_size: int = 64, future_steps: int = FUTURE_STEPS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, future_steps * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(-1, FUTURE_STEPS, 2)


# ---------------------------------------------------------------------
# TCN Block — Dilated Convolution
# ---------------------------------------------------------------------
class TCNBlock(nn.Module):
    """
    Single TCN block.

    WHY DILATIONS:
    --------------
    • Dilated convolutions expand the receptive field without increasing
      parameter count.
    • This allows the model to see long-range temporal patterns.
    • Dilation schedule (1, 2, 4, ...) doubles receptive field each layer.

    WHY BATCHNORM:
    --------------
    • Stabilizes training.
    • Helps prevent exploding activations.

    WHY NO RESIDUALS HERE:
    ----------------------
    • This is a simplified baseline TCN.
    • Residuals improve stability but are omitted to preserve original
      architecture exactly as provided.

    WHY TRIMMING PADDING:
    ----------------------
    • Dilated convolutions require padding to maintain output length.
    • After convolution, extra padded timesteps are removed so the output
      matches the input length exactly.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pad = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)

        # Trim extra padding so output length matches input length.
        if self.pad > 0:
            out = out[:, :, :-self.pad]

        return self.relu(self.bn(out))


# ---------------------------------------------------------------------
# TCN Model — Multi-Layer Dilated Convolution
# ---------------------------------------------------------------------
class TCNModel(nn.Module):
    """
    Temporal Convolutional Network (TCN).

    WHY TCN:
    --------
    • Convolutions are parallelizable → faster training than RNNs.
    • Dilations allow exponential receptive field growth.
    • TCNs often outperform RNNs on structured temporal data.

    ARCHITECTURE:
    -------------
    • Stack of TCNBlock layers with dilation schedule 1, 2, 4, ...
    • Final FC layer maps last timestep features → FUTURE_STEPS * 2
    """
    def __init__(self, input_size: int = 6, hidden_size: int = 64, levels: int = 3, kernel_size: int = 3):
        super().__init__()

        layers: list[nn.Module] = []
        in_ch = input_size

        for i in range(levels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, hidden_size, kernel_size, dilation))
            in_ch = hidden_size

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, FUTURE_STEPS * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TCN expects [batch, channels, time]
        x = x.transpose(1, 2)

        out = self.tcn(x)

        # Use last timestep features
        out = self.fc(out[:, :, -1])

        return out.view(-1, FUTURE_STEPS, 2)


# =====================================================================
# Training Loop — Shared Across All Models
# =====================================================================
def train_model(
    model: nn.Module,
    model_name: str,
    trajectories: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    epochs: int = 20,
    lr: float = 1e-3,
    max_batches: int = 200,
) -> None:
    """
    Train a model on windowed trajectories.

    WHY THIS DESIGN:
    ----------------
    • Shared loop ensures GRU/LSTM/TCN are trained identically.
    • MSE loss is appropriate because targets are displacement vectors.
    • max_batches caps per-epoch work for predictable runtime.
    • Heartbeats provide observability during long runs.

    TRAINING FLOW:
    --------------
    1. Generate windows via stream_multistep_batches
    2. Normalize inputs
    3. Forward pass
    4. Compute MSE loss
    5. Backprop + optimizer step
    6. Logging + heartbeats
    """
    log(f"[RUN] Training {model_name} started (epochs={epochs}, max_batches={max_batches})")

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    total_batches_est = epochs * max_batches
    global_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        batches = 0
        last_heartbeat = epoch_start

        print(f"[TRAIN] Epoch {epoch}/{epochs} started...")

        for windows_cpu, targets_cpu in stream_multistep_batches(trajectories, mean, std):
            # Convert to device and permute to [batch, time, features]
            windows = windows_cpu.to(DEVICE).permute(0, 2, 1)
            targets = targets_cpu.to(DEVICE)

            preds = model(windows)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1
            now = time.time()

            # Progress logging
            if batches % PROGRESS_BATCH_INTERVAL == 0:
                elapsed_epoch = now - epoch_start
                completed_batches = (epoch - 1) * max_batches + batches
                progress_batches = completed_batches / total_batches_est
                eta_total = (elapsed_epoch * epochs * max_batches / batches) - elapsed_epoch

                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().used / (1024**3)

                print(
                    f"[TRAIN] Epoch {epoch} — "
                    f"Batches: {progress_batches * 100:.1f}% "
                    f"({batches}/{max_batches}), "
                    f"{format_eta(eta_total)}, "
                    f"CPU={cpu:.1f}%, RAM={mem:.2f} GB"
                )

            # Heartbeat logging
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().used / (1024**3)
                elapsed = now - global_start

                print(
                    f"[HEARTBEAT] Training... elapsed={elapsed:.1f}s, "
                    f"CPU={cpu:.1f}%, RAM={mem:.2f} GB"
                )
                last_heartbeat = now

            if batches >= max_batches:
                break

        epoch_time = time.time() - epoch_start
        print(f"[TRAIN] Epoch {epoch} complete — avg loss {total_loss / max_batches:.4f} ({epoch_time:.2f}s)")

    total_time = time.time() - global_start
    log(f"[RUN] Training {model_name} complete — time={total_time:.2f}s")


# =====================================================================
# Prediction Loop — Shared Across All Models
# =====================================================================
def predict_model(
    model: nn.Module,
    model_name: str,
    trajectories: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    phase_name: str = "VAL",
    max_batches: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run prediction for a trained model.

    WHY THIS DESIGN:
    ----------------
    • Uses same window generator as training → consistent evaluation.
    • No gradients (torch.no_grad()) → faster inference.
    • Heartbeats provide observability.
    • max_batches caps runtime.

    RETURNS:
        preds_all, targets_all — stacked arrays for metrics.
    """
    log(f"[RUN] {model_name} prediction ({phase_name}) started...")

    model.to(DEVICE)
    model.eval()

    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []

    batches = 0
    start_time = time.time()
    last_heartbeat = start_time

    print(f"[PREDICT] Starting {phase_name} prediction...")

    with torch.no_grad():
        for windows_cpu, targets_cpu in stream_multistep_batches(trajectories, mean, std):
            windows = windows_cpu.to(DEVICE).permute(0, 2, 1)
            preds = model(windows).cpu().numpy()
            targets = targets_cpu.numpy()

            preds_all.append(preds)
            targets_all.append(targets)

            batches += 1
            now = time.time()

            # Progress logging
            if batches % PROGRESS_BATCH_INTERVAL == 0:
                elapsed = now - start_time
                progress_batches = batches / max_batches
                eta = (elapsed * max_batches / batches) - elapsed

                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().used / (1024**3)

                print(
                    f"[PREDICT] {phase_name} — "
                    f"Batches: {progress_batches * 100:.1f}% "
                    f"({batches}/{max_batches}), "
                    f"{format_eta(eta)}, "
                    f"CPU={cpu:.1f}%, RAM={mem:.2f} GB"
                )

            # Heartbeat logging
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().used / (1024**3)
                elapsed = now - start_time

                print(
                    f"[HEARTBEAT] {phase_name} prediction... "
                    f"elapsed={elapsed:.1f}s, CPU={cpu:.1f}%, RAM={mem:.2f} GB"
                )
                last_heartbeat = now

            if batches >= max_batches:
                break

    total_time = time.time() - start_time
    log(f"[RUN] {model_name} prediction ({phase_name}) complete — batches={batches}, time={total_time:.2f}s")

    return np.vstack(preds_all), np.vstack(targets_all)


# =====================================================================
# Saving + Loading Utilities
# =====================================================================
def save_model(model: nn.Module, name: str) -> None:
    """
    Save model state_dict to disk.

    WHY SAVE:
    ---------
    • Enables reproducibility.
    • Allows later evaluation without retraining.
    """
    path = os.path.join(MODEL_DIR, f"{name}_model.pt")
    torch.save(model.state_dict(), path)
    log(f"[SAVE] {name} model saved → {path}")


def save_normalization(mean: np.ndarray, std: np.ndarray) -> None:
    """
    Save global normalization stats.

    WHY SAVE:
    ---------
    • Ensures inference uses identical scaling as training.
    """
    np.save(os.path.join(MODEL_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(MODEL_DIR, "norm_std.npy"), std)
    log("[SAVE] Normalization stats saved → norm_mean.npy, norm_std.npy")


def save_metadata() -> None:
    """
    Save model metadata (features, window sizes) to JSON.

    WHY SAVE:
    ---------
    • Supports reproducibility.
    • Allows external tools to understand model input format.
    """
    metadata = {
        "features": FEATURES,
        "input_window": INPUT_WINDOW,
        "future_steps": FUTURE_STEPS,
    }
    path = os.path.join(MODEL_DIR, "model_metadata.json")

    with open(path, "w") as f:
        json.dump(metadata, f)

    log(f"[SAVE] Metadata saved → {path}")


def load_model(model_class: type[nn.Module], name: str) -> nn.Module:
    """
    Load a saved model from disk.

    RETURNS:
        Model instance in eval mode on DEVICE.
    """
    path = os.path.join(MODEL_DIR, f"{name}_model.pt")
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_normalization() -> tuple[np.ndarray, np.ndarray]:
    """
    Load global normalization stats from disk.
    """
    mean = np.load(os.path.join(MODEL_DIR, "norm_mean.npy"))
    std = np.load(os.path.join(MODEL_DIR, "norm_std.npy"))
    return mean, std


# =====================================================================
# Final Summary Table
# =====================================================================
def print_summary_table(results: dict[str, dict[str, float]]) -> None:
    """
    Print final metrics for all models in three formats:

        • ASCII table
        • Pretty per-model breakdown
        • JSON dump

    WHY MULTIPLE FORMATS:
    ---------------------
    • ASCII table is easy to scan.
    • Pretty format is human-readable.
    • JSON is machine-readable for downstream tools.
    """
    print("\n================ FINAL SUMMARY (ASCII) ================")
    print("Model       | ADE_VAL | MDE_VAL | FDE_VAL | ADE_TEST | MDE_TEST | FDE_TEST")
    print("--------------------------------------------------------------------------")

    for model, metrics in results.items():
        print(
            f"{model:<11} {metrics['ADE_VAL']:<8.3f} {metrics['MDE_VAL']:<8.3f} "
            f"{metrics['FDE_VAL']:<8.3f} {metrics['ADE_TEST']:<9.3f} "
            f"{metrics['MDE_TEST']:<9.3f} {metrics['FDE_TEST']:<9.3f}"
        )

    print("=======================================================\n")

    print("================ FINAL SUMMARY (PRETTY) ===============")
    for model, metrics in results.items():
        print(f"\n{model}")
        print("  Validation:")
        print(f"    ADE: {metrics['ADE_VAL']:.3f}")
        print(f"    MDE: {metrics['MDE_VAL']:.3f}")
        print(f"    FDE: {metrics['FDE_VAL']:.3f}")
        print("  Test:")
        print(f"    ADE: {metrics['ADE_TEST']:.3f}")
        print(f"    MDE: {metrics['MDE_TEST']:.3f}")
        print(f"    FDE: {metrics['FDE_TEST']:.3f}")

    print("=======================================================\n")

    print("================ FINAL SUMMARY (JSON) =================")
    print(json.dumps(results, indent=4))
    print("=======================================================\n")

    log("[RUN] Final summary table printed.")


# =====================================================================
# Main Orchestration
# =====================================================================
def main() -> None:
    """
    Orchestrate the full GeoLife pipeline.

    PIPELINE:
    ---------
    1. Load trajectories
    2. Compute normalization
    3. Split into train/val/test
    4. Save normalization + metadata
    5. Evaluate baseline
    6. Train + evaluate GRU
    7. Train + evaluate LSTM
    8. Train + evaluate TCN
    9. Save models
    10. Print final summary

    Any exception is logged and re-raised.

