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

# --------------------------------------------------------------------
# Global configuration and constants
# --------------------------------------------------------------------
# Path to the preprocessed GeoLife parquet file. Centralized here so
# the pipeline can be re-pointed to different datasets without touching
# core logic.
PARQUET_PATH = r"C:\Users\gb630\OneDrive\USD AAI\USD AAI\AAI-590 CAPSTONE\FINAL PROJECT\Data\geolife.parquet"

# Features used for modeling. These are engineered motion features:
# - x, y: projected coordinates
# - speed: scalar speed
# - heading: direction of travel
# - accel: linear acceleration
# - turn_rate: angular rate of change
FEATURES = ["x", "y", "speed", "heading", "accel", "turn_rate"]

# Number of timesteps in the input window and number of future steps
# to predict. These define the temporal context and prediction horizon.
INPUT_WINDOW = 20
FUTURE_STEPS = 5

# Device selection. For this project we explicitly use CPU to keep
# runtime predictable and avoid GPU dependency for grading/repro.
DEVICE = "cpu"

# Heartbeat and progress reporting intervals. These control how often
# the pipeline prints system status and training progress.
HEARTBEAT_INTERVAL = 5.0
PROGRESS_BATCH_INTERVAL = 10

# Threshold for detecting GPS glitches. Any jump larger than this
# between consecutive points is treated as a glitch and removed.
GLITCH_THRESHOLD_METERS = 100.0

# Directories for saving models and logs. Created on startup to ensure
# the pipeline can write artifacts without manual setup.
MODEL_DIR = "./models"
LOG_DIR = os.path.join(MODEL_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Logging system (dual logging + timestamps)
# --------------------------------------------------------------------
# We maintain both a persistent log (across runs) and a per-run log.
# This supports reproducibility, debugging, and auditability.
run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
persistent_log_path = os.path.join(LOG_DIR, "geolife.log")
run_log_path = os.path.join(LOG_DIR, f"run_{run_timestamp}.log")


def log(msg):
    """
    Write high-level events to stdout AND both log files.

    This function centralizes logging so that all important events
    (data loading, training start/end, errors, etc.) are captured with
    timestamps. Logs are UTF-8 encoded to avoid encoding issues.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {msg}"

    # stdout for immediate visibility
    print(entry)

    # persistent log (across runs)
    with open(persistent_log_path, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

    # per-run log (specific to this execution)
    with open(run_log_path, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


# --------------------------------------------------------------------
# Utility: human-friendly ETA
# --------------------------------------------------------------------
def format_eta(seconds):
    """
    Convert a raw seconds estimate into a human-friendly string.

    Used in training and prediction loops to give a rough sense of how
    long the remaining work will take.
    """
    if seconds is None or seconds <= 0:
        return "ETA: unknown"
    m, s = divmod(int(seconds), 60)
    if m == 0:
        return f"ETA: {s}s"
    return f"ETA: {m}m {s}s"


# --------------------------------------------------------------------
# Load and preprocess trajectories
# --------------------------------------------------------------------
def load_trajectories(path):
    """
    Load trajectories from a parquet file and apply basic filtering.

    Steps:
    - Read each row group from the parquet file.
    - Group rows by 'source_file' to reconstruct individual trajectories.
    - Convert selected FEATURES to a NumPy array.
    - Compute per-step displacement in (x, y) to detect GPS glitches.
    - Remove points where displacement exceeds GLITCH_THRESHOLD_METERS.
    - Discard trajectories that are too short to produce at least one
      training window (INPUT_WINDOW + FUTURE_STEPS).

    This ensures that downstream models see clean, usable trajectories.
    """
    log("[RUN] Loading parquet and building trajectories...")
    pf = pq.ParquetFile(path)
    trajectories = []

    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        df = tbl.to_pandas()

        for src, traj in df.groupby("source_file"):
            # Extract motion features as float32 for efficiency
            t = traj[FEATURES].to_numpy(dtype=np.float32)

            # Compute displacement between consecutive points
            coords = t[:, 0:2]
            deltas = np.linalg.norm(coords[1:] - coords[:-1], axis=1)

            # Build a mask that keeps points with reasonable displacement
            mask = np.concatenate([[True], deltas <= GLITCH_THRESHOLD_METERS])
            t_filtered = t[mask]

            # Enforce minimum length for windowing
            if len(t_filtered) <= INPUT_WINDOW + FUTURE_STEPS:
                continue

            trajectories.append(t_filtered)

    log(f"[RUN] Total trajectories after filtering: {len(trajectories)}")
    return trajectories


def compute_normalization(trajectories):
    """
    Compute global mean and standard deviation across all trajectories.

    This provides dataset-wide normalization parameters that:
    - Center features around zero.
    - Scale features to unit variance (unless std is zero, in which case
      we set std to 1 to avoid division by zero).

    Global normalization helps models see consistent feature scales
    across different trajectories.
    """
    log("[RUN] Computing global normalization stats...")
    all_data = np.concatenate(trajectories, axis=0)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)
    std[std == 0] = 1.0
    log(f"[RUN] Normalization mean: {mean}")
    log(f"[RUN] Normalization std:  {std}")
    return mean, std


def split_trajectories(trajectories, train_ratio=0.6, val_ratio=0.2):
    """
    Split trajectories into train/validation/test sets at the trajectory level.

    We:
    - Shuffle trajectory indices to avoid ordering bias.
    - Assign 60% to training, 20% to validation, and 20% to test.

    Crucially, entire trajectories are assigned to a single split. This
    avoids leakage where parts of the same trajectory appear in both
    train and test, which would inflate performance metrics.
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

def log_trajectory_length_histogram(trajectories):
    """
    Log a histogram of trajectory lengths to help tune window caps.

    This function is intentionally lightweight:
    - Computes lengths of all trajectories.
    - Buckets them into human-friendly ranges.
    - Logs counts per bucket.
    - Logs min, max, mean, median.

    This helps diagnose memory pressure caused by extremely long trajectories.
    """
    lengths = np.array([len(t) for t in trajectories], dtype=np.int64)

    if len(lengths) == 0:
        log("[RUN] No trajectories available for histogram.")
        return

    bins = [0, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000]
    hist, edges = np.histogram(lengths, bins=bins)

    log("[RUN] Trajectory length histogram (rows per trajectory):")
    for count, left, right in zip(hist, edges[:-1], edges[1:]):
        log(f"  [{left:6d}, {right:6d}) : {count} trajectories")

    log(
        f"[RUN] Trajectory length stats — "
        f"min={lengths.min()}, max={lengths.max()}, "
        f"mean={lengths.mean():.1f}, median={np.median(lengths):.1f}"
    )

# --------------------------------------------------------------------
# Window generator (with per-window normalization)
# --------------------------------------------------------------------
def stream_multistep_batches(trajectories, mean, std, max_windows_per_traj=5000):
    """
    Stream batches of multistep windows from a list of trajectories.

    For each trajectory:
    - Apply global normalization using dataset-wide mean/std.
    - Slide a window of length INPUT_WINDOW across the trajectory.
    - For each window:
        * Apply per-window normalization (mean/std over that window).
          This reduces trajectory-level scale differences and stabilizes
          training, especially for the TCN.
        * Transpose to [features, time] to match model expectations.
        * Compute future absolute positions for the next FUTURE_STEPS.
        * Convert future positions to displacements relative to the last
          observed position in the window.

    This function yields tensors of windows and targets, and is used
    both for training and evaluation. It is streaming to avoid loading
    all windows into memory at once.

    IMPORTANT FIX:
    - A hard cap (max_windows_per_traj) is applied to prevent extremely
      long trajectories (40k–77k windows) from blowing up memory and
      causing native segmentation faults during TCN training.

    PATCH:
    - Added trajectory length histogram logging to help tune caps.
    - Added adaptive per-trajectory caps based on trajectory length.
    - Added logging when caps activate for a given trajectory.
    """
    total_windows = 0
    n_traj = len(trajectories)

    # --------------------------------------------------------------
    # Trajectory length histogram (for cap tuning)
    # --------------------------------------------------------------
    traj_lengths = np.array([len(t) for t in trajectories], dtype=np.int64)
    if n_traj > 0:
        bins = [0, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000]
        hist, bin_edges = np.histogram(traj_lengths, bins=bins)
        log("[RUN] Trajectory length histogram (rows per trajectory):")
        for count, left, right in zip(hist, bin_edges[:-1], bin_edges[1:]):
            log(f"  [{left:6d}, {right:6d}) : {count} trajectories")

        log(
            f"[RUN] Trajectory length stats — "
            f"min={traj_lengths.min()}, max={traj_lengths.max()}, "
            f"mean={traj_lengths.mean():.1f}, median={np.median(traj_lengths):.1f}"
        )

    for idx, t in enumerate(trajectories, start=1):
        # Apply global normalization first
        t_norm = (t - mean) / std

        # Number of possible windows in this trajectory
        n = len(t_norm) - INPUT_WINDOW - FUTURE_STEPS
        if n <= 0:
            continue

        # ----------------------------------------------------------
        # Adaptive per-trajectory cap based on trajectory length
        # ----------------------------------------------------------
        # Very long trajectories get a stricter cap to avoid huge batches.
        effective_cap = max_windows_per_traj
        if n > 20000:
            effective_cap = min(effective_cap, 2000)
        elif n > 10000:
            effective_cap = min(effective_cap, 4000)

        if effective_cap < max_windows_per_traj:
            log(
                f"[CAP] Trajectory {idx}/{n_traj} length={len(t)} rows, "
                f"windows_possible={n}, "
                f"effective_cap={effective_cap} (global_cap={max_windows_per_traj})"
            )

        windows = []
        targets = []

        for i in range(n):
            # HARD SAFETY CAP (per-trajectory, adaptive):
            # Prevent massive trajectories from generating tens of thousands
            # of windows, which leads to memory exhaustion and segfaults.
            if len(windows) >= effective_cap:
                # Log once per trajectory when the cap actually activates.
                log(
                    f"[CAP] Activated for trajectory {idx}/{n_traj}: "
                    f"len(t)={len(t)}, windows_generated={len(windows)}, "
                    f"windows_possible={n}, effective_cap={effective_cap}"
                )
                break

            # Extract a raw window of length INPUT_WINDOW
            w = t_norm[i:i+INPUT_WINDOW].astype(np.float32)  # [INPUT_WINDOW, features]

            # Per-window normalization: center and scale this window
            w_mean = w.mean(axis=0, keepdims=True)
            w_std = w.std(axis=0, keepdims=True)
            w_std[w_std == 0] = 1.0
            w_norm = (w - w_mean) / w_std  # [INPUT_WINDOW, features]

            # Transpose to [features, time] for all models
            w_norm = w_norm.T  # [features, INPUT_WINDOW]

            # Compute future absolute positions (normalized space)
            future_abs = t_norm[i+INPUT_WINDOW:i+INPUT_WINDOW+FUTURE_STEPS, 0:2]
            last_pos = t_norm[i+INPUT_WINDOW-1, 0:2]

            # Convert to displacements relative to last observed position
            future_disp = future_abs - last_pos

            windows.append(w_norm)
            targets.append(future_disp)

        batch_windows = len(windows)
        total_windows += batch_windows

        # Progress print for visibility; not logged to avoid log spam
        print(
            f"[TRAJ {idx}/{n_traj}] {len(t)} rows → "
            f"{batch_windows} windows (total_windows={total_windows})"
        )

        if batch_windows == 0:
            # Skip yielding empty batches to avoid downstream issues
            continue

        yield torch.tensor(np.stack(windows)), torch.tensor(np.stack(targets))


# --------------------------------------------------------------------
# Baseline model (simple heuristic)
# --------------------------------------------------------------------
def baseline_predict(windows):
    """
    Baseline prediction: extrapolate using the last observed displacement.

    We:
    - Take the last two positions in the window.
    - Compute the displacement between them.
    - Repeat that displacement for all FUTURE_STEPS.

    This is a naive constant-velocity baseline used to contextualize
    the performance of learned models.
    """
    last_pos = windows[:, 0:2, -1]
    prev_pos = windows[:, 0:2, -2]
    disp = last_pos - prev_pos
    return np.tile(disp[:, None, :], (1, FUTURE_STEPS, 1))


def evaluate_baseline(trajectories, mean, std):
    """
    Evaluate the baseline model on a given set of trajectories.

    Uses the same window generator as the learned models to ensure
    fair comparison. Logs heartbeat information to track CPU and RAM
    usage during evaluation.
    """
    log("[RUN] Baseline evaluation started...")
    preds_all = []
    targets_all = []

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
                f"batches={batch_count}, elapsed={now-start_time:.1f}s, "
                f"CPU={cpu:.1f}%, RAM={mem:.2f} GB"
            )
            last_heartbeat = now

    total_time = time.time() - start_time
    log(f"[RUN] Baseline evaluation complete — batches={batch_count}, time={total_time:.2f}s")
    return np.vstack(preds_all), np.vstack(targets_all)


# --------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------
def ade(preds, targets):
    """
    Average Displacement Error (ADE).

    Computes the mean Euclidean distance between predicted and target
    positions across all timesteps and all samples.
    """
    return float(np.linalg.norm(preds - targets, axis=2).mean())


def mde(preds, targets):
    """
    Maximum Displacement Error (MDE).

    Computes the maximum Euclidean distance between predicted and target
    positions across all timesteps and all samples. This highlights
    worst-case errors.
    """
    return float(np.linalg.norm(preds - targets, axis=2).max())


def fde(preds, targets):
    """
    Final Displacement Error (FDE).

    Computes the mean Euclidean distance between the final predicted
    position and the final target position for each sample. This is
    particularly important for trajectory forecasting tasks where the
    endpoint matters.
    """
    return float(np.linalg.norm(preds[:, -1, :] - targets[:, -1, :], axis=1).mean())


def print_metrics(name, preds, targets):
    """
    Compute ADE, MDE, and FDE for a given set of predictions and targets,
    then print and log them with a model/phase label.
    """
    msg = (
        f"{name} — ADE: {ade(preds, targets):.3f} m, "
        f"MDE: {mde(preds, targets):.3f} m, "
        f"FDE: {fde(preds, targets):.3f} m"
    )
    print(msg)
    log(f"[RUN] {msg}")


# --------------------------------------------------------------------
# Models: GRU, LSTM, TCN
# --------------------------------------------------------------------
class GRUModel(nn.Module):
    """
    GRU-based sequence model.

    Architecture:
    - Single GRU layer with 64 hidden units.
    - Fully connected layer mapping the final hidden state to
      FUTURE_STEPS * 2 outputs (x,y displacements).

    GRUs are chosen for their efficiency and ability to capture
    temporal dependencies with fewer parameters than LSTMs.
    """
    def __init__(self, input_size=6, hidden_size=64, future_steps=FUTURE_STEPS):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, future_steps * 2)

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.gru(x)
        # Use the final hidden state as a summary of the sequence
        out = self.fc(out[:, -1, :])
        return out.view(-1, FUTURE_STEPS, 2)


class LSTMModel(nn.Module):
    """
    LSTM-based sequence model.

    Architecture:
    - Single LSTM layer with 64 hidden units.
    - Fully connected layer mapping the final hidden state to
      FUTURE_STEPS * 2 outputs.

    LSTMs are included due to their strong performance on sequential
    data with long-range dependencies.
    """
    def __init__(self, input_size=6, hidden_size=64, future_steps=FUTURE_STEPS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, future_steps * 2)

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)
        # Use the final hidden state as a summary of the sequence
        out = self.fc(out[:, -1, :])
        return out.view(-1, FUTURE_STEPS, 2)


class TCNBlock(nn.Module):
    """
    Single block of a Temporal Convolutional Network (TCN).

    Components:
    - 1D convolution with dilation and padding to expand receptive field.
    - Batch normalization for stable training.
    - ReLU activation.
    - Dropout for regularization.
    - Residual connection (with optional 1x1 conv for channel matching).

    The residual path helps gradients flow through deep stacks of
    dilated convolutions, reducing vanishing/exploding issues.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pad = padding

        # Residual path: if channel count changes, use 1x1 conv to match
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        # x: [batch, channels, time]
        out = self.conv(x)
        if self.pad > 0:
            # Remove extra padding introduced by dilation to keep
            # output length aligned with input length.
            out = out[:, :, :-self.pad]

        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection: add original (or downsampled) input
        res = self.downsample(x) if self.downsample is not None else x
        out = out + res
        return self.relu(out)


class TCNModel(nn.Module):
    """
    Enhanced Temporal Convolutional Network for trajectory forecasting.

    Design choices:
    - Input size: 6 features (x, y, speed, heading, accel, turn_rate).
    - Hidden size: 128 channels for richer representation.
    - Levels: 5 TCN blocks with dilations {1, 2, 4, 8, 16}, giving a
      receptive field that covers and exceeds the 20-step input window.
    - Sinusoidal positional encodings: added to inputs to give the
      model awareness of timestep positions (TCN is otherwise
      translation-invariant).
    - Multi-head self-attention: applied over the time dimension after
      the TCN stack to let the model focus on the most informative
      timesteps.
    - Final fully connected layer: maps the attended summary to
      FUTURE_STEPS * 2 displacement outputs.

    This architecture combines convolutional pattern extraction with
    attention-based temporal weighting.
    """
    def __init__(self, input_size=6, hidden_size=128, levels=5, kernel_size=3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.levels = levels
        self.kernel_size = kernel_size

        # Precompute sinusoidal positional encodings for the input window.
        # Stored as a buffer so they move with the model across devices.
        self.register_buffer("pos_encoding", self._build_positional_encoding(INPUT_WINDOW, input_size))

        layers = []
        in_ch = input_size
        for i in range(levels):
            dilation = 2 ** i  # 1, 2, 4, 8, 16 → receptive field >= 20
            layers.append(TCNBlock(in_ch, hidden_size, kernel_size, dilation, dropout=0.2))
            in_ch = hidden_size

        self.tcn = nn.Sequential(*layers)

        # Multi-head attention over time. Embedding dimension matches
        # hidden_size so we can attend over the TCN outputs directly.
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)

        # Final prediction head: maps attended summary to future displacements.
        self.fc = nn.Linear(hidden_size, FUTURE_STEPS * 2)

    def _build_positional_encoding(self, length, dim):
        """
        Build sinusoidal positional encodings as in the original
        Transformer paper. These encodings give the model information
        about the relative position of each timestep in the sequence.
        """
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [length, dim]

    def forward(self, x):
        """
        Forward pass through the TCN model.

        Steps:
        - Add positional encodings to the input sequence.
        - Transpose to [batch, channels, time] for TCN.
        - Pass through the TCN stack.
        - Transpose back to [batch, time, channels] for attention.
        - Apply self-attention over time.
        - Use the last timestep of the attended sequence as a summary.
        - Map summary to FUTURE_STEPS * 2 outputs and reshape.
        """
        # x: [batch, seq_len, features]
        b, seq_len, feat = x.shape
        if seq_len == INPUT_WINDOW and feat == self.input_size:
            # Broadcast positional encodings across the batch
            pe = self.pos_encoding.unsqueeze(0).expand(b, -1, -1)  # [b, seq_len, feat]
            x = x + pe

        # TCN expects [batch, channels, time]
        x = x.transpose(1, 2)  # [b, features, seq_len]
        out = self.tcn(x)      # [b, hidden_size, seq_len]

        # Attention over time: convert to [b, seq_len, hidden_size]
        out_time = out.transpose(1, 2)
        attn_out, _ = self.attn(out_time, out_time, out_time)

        # Use the last timestep as a summary of the attended sequence
        summary = attn_out[:, -1, :]  # [b, hidden_size]

        out = self.fc(summary)        # [b, FUTURE_STEPS*2]
        return out.view(-1, FUTURE_STEPS, 2)


# --------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------
def train_model(model, model_name, trajectories, mean, std, epochs=20, lr=1e-3, max_batches=200):
    """
    Generic training loop for any of the models.

    We now use an adaptive window cap based on model type:
    - Baseline/GRU/LSTM can tolerate more windows.
    - TCN is more memory-hungry (dilated convs + attention), so we
      cap per-trajectory windows more aggressively.
    """
    log(f"[RUN] Training {model_name} started (epochs={epochs}, max_batches={max_batches})")
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Adaptive cap based on model name
    if model_name.upper() == "TCN":
        max_windows_per_traj = 3000
    elif model_name.upper() in ("GRU", "LSTM"):
        max_windows_per_traj = 8000
    else:
        max_windows_per_traj = 10000  # default / baseline

    log(f"[RUN] Using max_windows_per_traj={max_windows_per_traj} for {model_name}")

    total_batches_est = epochs * max_batches
    global_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        batches = 0
        last_heartbeat = epoch_start

        print(f"[TRAIN] Epoch {epoch}/{epochs} started...")

        for windows_cpu, targets_cpu in stream_multistep_batches(
            trajectories, mean, std, max_windows_per_traj=max_windows_per_traj
        ):
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

            if batches % PROGRESS_BATCH_INTERVAL == 0:
                elapsed_epoch = now - epoch_start
                completed_batches = (epoch - 1) * max_batches + batches
                progress_batches = completed_batches / total_batches_est
                eta_total = (elapsed_epoch * epochs * max_batches / batches) - elapsed_epoch
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().used / (1024**3)
                print(
                    f"[TRAIN] Epoch {epoch} — "
                    f"Batches: {progress_batches*100:.1f}% "
                    f"({batches}/{max_batches}), "
                    f"{format_eta(eta_total)}, "
                    f"CPU={cpu:.1f}%, RAM={mem:.2f} GB"
                )

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
        print(f"[TRAIN] Epoch {epoch} complete — avg loss {total_loss/max_batches:.4f} ({epoch_time:.2f}s)")

    total_time = time.time() - global_start
    log(f"[RUN] Training {model_name} complete — time={total_time:.2f}s")


# --------------------------------------------------------------------
# Prediction loop
# --------------------------------------------------------------------
def predict_model(model, model_name, trajectories, mean, std, phase_name="VAL", max_batches=200):
    """
    Generic prediction loop for any of the models.

    Design:
    - Uses the same window generator as training to ensure consistency.
    - Runs in eval mode with no gradient computation.
    - Collects all predictions and targets for metric computation.
    - Prints progress and heartbeat information.

    This loop is used for both validation and test phases.

    PATCH:
    - Added adaptive max_windows_per_traj based on model type.
    - Added logging showing which cap is used.
    - Caps are applied per-trajectory inside stream_multistep_batches,
      with additional logging when they activate.
    """
    log(f"[RUN] {model_name} prediction ({phase_name}) started...")
    model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------------
    # Adaptive window caps based on model type
    # --------------------------------------------------------------
    # TCN is memory-heavy (dilated convs + attention), so it gets the smallest cap.
    # GRU/LSTM can tolerate more windows.
    # Baseline or unknown models get a generous cap.
    if model_name.upper() == "TCN":
        max_windows_per_traj = 3000
    elif model_name.upper() in ("GRU", "LSTM"):
        max_windows_per_traj = 8000
    else:
        max_windows_per_traj = 10000

    log(
        f"[RUN] Using adaptive max_windows_per_traj={max_windows_per_traj} "
        f"for {model_name} during {phase_name} prediction."
    )

    preds_all, targets_all = [], []
    batches = 0
    start_time = time.time()
    last_heartbeat = start_time

    print(f"[PREDICT] Starting {phase_name} prediction...")

    with torch.no_grad():
        # --------------------------------------------------------------
        # Pass adaptive cap into stream_multistep_batches
        # --------------------------------------------------------------
        for windows_cpu, targets_cpu in stream_multistep_batches(
            trajectories, mean, std, max_windows_per_traj=max_windows_per_traj
        ):
            windows = windows_cpu.to(DEVICE).permute(0, 2, 1)
            preds = model(windows).cpu().numpy()
            targets = targets_cpu.numpy()

            preds_all.append(preds)
            targets_all.append(targets)

            batches += 1
            now = time.time()

            if batches % PROGRESS_BATCH_INTERVAL == 0:
                elapsed = now - start_time
                progress_batches = batches / max_batches
                eta = (elapsed * max_batches / batches) - elapsed
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().used / (1024**3)
                print(
                    f"[PREDICT] {phase_name} — "
                    f"Batches: {progress_batches*100:.1f}% "
                    f"({batches}/{max_batches}), "
                    f"{format_eta(eta)}, "
                    f"CPU={cpu:.1f}%, RAM={mem:.2f} GB"
                )

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
                # Cap batches to control runtime
                break

    total_time = time.time() - start_time
    log(
        f"[RUN] {model_name} prediction ({phase_name}) complete — "
        f"batches={batches}, time={total_time:.2f}s"
    )
    return np.vstack(preds_all), np.vstack(targets_all)

# --------------------------------------------------------------------
# Saving + Loading
# --------------------------------------------------------------------
def save_model(model, name):
    """
    Save a model's state_dict to disk under the models directory.

    This supports reproducibility and allows later reuse of trained
    models without retraining.
    """
    path = os.path.join(MODEL_DIR, f"{name}_model.pt")
    torch.save(model.state_dict(), path)
    log(f"[SAVE] {name} model saved → {path}")


def save_normalization(mean, std):
    """
    Save global normalization parameters (mean and std) to disk.

    These are needed to ensure that any future inference uses the same
    normalization as training.
    """
    np.save(os.path.join(MODEL_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(MODEL_DIR, "norm_std.npy"), std)
    log("[SAVE] Normalization stats saved → norm_mean.npy, norm_std.npy")


def save_metadata():
    """
    Save basic model metadata (features, input window, future steps)
    to a JSON file. This documents the configuration used for training
    and supports future inspection or reuse.
    """
    metadata = {
        "features": FEATURES,
        "input_window": INPUT_WINDOW,
        "future_steps": FUTURE_STEPS
    }
    path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f)
    log(f"[SAVE] Metadata saved → {path}")


def load_model(model_class, name):
    """
    Load a saved model from disk and return it in eval mode.

    This is useful for running inference or further evaluation without
    retraining.
    """
    path = os.path.join(MODEL_DIR, f"{name}_model.pt")
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_normalization():
    """
    Load saved normalization parameters (mean and std) from disk.
    """
    mean = np.load(os.path.join(MODEL_DIR, "norm_mean.npy"))
    std = np.load(os.path.join(MODEL_DIR, "norm_std.npy"))
    return mean, std


# --------------------------------------------------------------------
# Final summary table
# --------------------------------------------------------------------
def print_summary_table(results):
    """
    Print a summary table of metrics for all models in multiple formats:
    - ASCII table for quick inspection.
    - Pretty per-model breakdown.
    - JSON for machine-readable logging.

    This function centralizes reporting of final performance metrics.
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
        print(f"  Validation:")
        print(f"    ADE: {metrics['ADE_VAL']:.3f}")
        print(f"    MDE: {metrics['MDE_VAL']:.3f}")
        print(f"    FDE: {metrics['FDE_VAL']:.3f}")
        print(f"  Test:")
        print(f"    ADE: {metrics['ADE_TEST']:.3f}")
        print(f"    MDE: {metrics['MDE_TEST']:.3f}")
        print(f"    FDE: {metrics['FDE_TEST']:.3f}")
    print("=======================================================\n")

    print("================ FINAL SUMMARY (JSON) =================")
    print(json.dumps(results, indent=4))
    print("=======================================================\n")

    log("[RUN] Final summary table printed.")


# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------
def main():
    """
    Main pipeline entry point.

    Steps:
    - Load and preprocess trajectories.
    - Compute global normalization.
    - Split trajectories into train/val/test.
    - Save normalization and metadata.
    - Evaluate baseline on val/test.
    - Train GRU, LSTM, and TCN models.
    - Evaluate each model on val/test.
    - Print final summary table.

    This function orchestrates the entire experimental pipeline.
    """
    log("[RUN] GeoLife pipeline run started.")

    try:
        # Load and preprocess data
        trajectories = load_trajectories(PARQUET_PATH)
        log_trajectory_length_histogram(trajectories)
        mean, std = compute_normalization(trajectories)
        train_traj, val_traj, test_traj = split_trajectories(trajectories)

        # Persist normalization and metadata for reproducibility
        save_normalization(mean, std)
        save_metadata()

        results = {}

        # -------------------------
        # Baseline
        # -------------------------
        print("\n==============================")
        print("Evaluating Baseline (VAL)")
        print("==============================")
        base_val_preds, base_val_targets = evaluate_baseline(val_traj, mean, std)
        print_metrics("Baseline (VAL)", base_val_preds, base_val_targets)

        print("\n==============================")
        print("Evaluating Baseline (TEST)")
        print("==============================")
        base_test_preds, base_test_targets = evaluate_baseline(test_traj, mean, std)
        print_metrics("Baseline (TEST)", base_test_preds, base_test_targets)

        results["Baseline"] = {
            "ADE_VAL": ade(base_val_preds, base_val_targets),
            "MDE_VAL": mde(base_val_preds, base_val_targets),
            "FDE_VAL": fde(base_val_preds, base_val_targets),
            "ADE_TEST": ade(base_test_preds, base_test_targets),
            "MDE_TEST": mde(base_test_preds, base_test_targets),
            "FDE_TEST": fde(base_test_preds, base_test_targets),
        }

        # -------------------------
        # GRU
        # -------------------------
        print("\n==============================")
        print("Starting GRU training...")
        print("==============================")
        gru = GRUModel()
        train_model(gru, "GRU", train_traj, mean, std)
        save_model(gru, "gru")

        gru_val_preds, gru_val_targets = predict_model(gru, "GRU", val_traj, mean, std, phase_name="VAL")
        gru_test_preds, gru_test_targets = predict_model(gru, "GRU", test_traj, mean, std, phase_name="TEST")
        print_metrics("GRU (VAL)", gru_val_preds, gru_val_targets)
        print_metrics("GRU (TEST)", gru_test_preds, gru_test_targets)

        results["GRU"] = {
            "ADE_VAL": ade(gru_val_preds, gru_val_targets),
            "MDE_VAL": mde(gru_val_preds, gru_val_targets),
            "FDE_VAL": fde(gru_val_preds, gru_val_targets),
            "ADE_TEST": ade(gru_test_preds, gru_test_targets),
            "MDE_TEST": mde(gru_test_preds, gru_test_targets),
            "FDE_TEST": fde(gru_test_preds, gru_test_targets),
        }

        # -------------------------
        # LSTM
        # -------------------------
        print("\n==============================")
        print("Starting LSTM training...")
        print("==============================")
        lstm = LSTMModel()
        train_model(lstm, "LSTM", train_traj, mean, std)
        save_model(lstm, "lstm")

        lstm_val_preds, lstm_val_targets = predict_model(lstm, "LSTM", val_traj, mean, std, phase_name="VAL")
        lstm_test_preds, lstm_test_targets = predict_model(lstm, "LSTM", test_traj, mean, std, phase_name="TEST")
        print_metrics("LSTM (VAL)", lstm_val_preds, lstm_val_targets)
        print_metrics("LSTM (TEST)", lstm_test_preds, lstm_test_targets)

        results["LSTM"] = {
            "ADE_VAL": ade(lstm_val_preds, lstm_val_targets),
            "MDE_VAL": mde(lstm_val_preds, lstm_val_targets),
            "FDE_VAL": fde(lstm_val_preds, lstm_val_targets),
            "ADE_TEST": ade(lstm_test_preds, lstm_test_targets),
            "MDE_TEST": mde(lstm_test_preds, lstm_test_targets),
            "FDE_TEST": fde(lstm_test_preds, lstm_test_targets),
        }

        # -------------------------
        # TCN
        # -------------------------
        print("\n==============================")
        print("Starting TCN training...")
        print("==============================")
        tcn = TCNModel()
        # TCN is deeper and benefits from more batches per epoch.
        train_model(tcn, "TCN", train_traj, mean, std, max_batches=400)
        save_model(tcn, "tcn")

        tcn_val_preds, tcn_val_targets = predict_model(tcn, "TCN", val_traj, mean, std, phase_name="VAL", max_batches=400)
        tcn_test_preds, tcn_test_targets = predict_model(tcn, "TCN", test_traj, mean, std, phase_name="TEST", max_batches=400)
        print_metrics("TCN (VAL)", tcn_val_preds, tcn_val_targets)
        print_metrics("TCN (TEST)", tcn_test_preds, tcn_test_targets)

        results["TCN"] = {
            "ADE_VAL": ade(tcn_val_preds, tcn_val_targets),
            "MDE_VAL": mde(tcn_val_preds, tcn_val_targets),
            "FDE_VAL": fde(tcn_val_preds, tcn_val_targets),
            "ADE_TEST": ade(tcn_test_preds, tcn_test_targets),
            "MDE_TEST": mde(tcn_test_preds, tcn_test_targets),
            "FDE_TEST": fde(tcn_test_preds, tcn_test_targets),
        }

        # -------------------------
        # Final summary
        # -------------------------
        print_summary_table(results)
        log("[RUN] GeoLife pipeline run completed successfully.")

    except Exception as e:
        # Any unhandled exception is logged before being re-raised.
        log(f"[ERROR] Exception during run: {e}")
        raise


if __name__ == "__main__":
    main()
