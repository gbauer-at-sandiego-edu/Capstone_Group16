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
# Core configuration knobs for the entire pipeline. Changing these
# affects data processing, model behavior, and training dynamics.

# Path to the GeoLife parquet dataset. This is the single data source
# for all experiments in this module.
PARQUET_PATH = r"C:\Users\gb630\OneDrive\USD AAI\USD AAI\AAI-590 CAPSTONE\FINAL PROJECT\Data\geolife.parquet"

# Ordered list of feature columns expected in the parquet file.
# The order is important because models and normalization assume this
# exact layout. Any change here must be reflected in model input sizes.
FEATURES = ["x", "y", "speed", "heading", "accel", "turn_rate"]

# Number of past timesteps used as input to the models. Defines the
# length of each window and the temporal context the models see.
INPUT_WINDOW = 20

# Number of future timesteps to predict. Models output FUTURE_STEPS
# displacement vectors relative to the last input position.
FUTURE_STEPS = 5

# Device selection. Using CPU keeps behavior simple and reproducible.
# This can be changed to "cuda" if GPU acceleration is desired.
DEVICE = "cpu"

# Heartbeat interval (seconds) for long-running loops. Heartbeats give
# coarse-grained progress and resource usage without flooding logs.
HEARTBEAT_INTERVAL = 5.0

# How often (in batches) to print progress during training/prediction.
PROGRESS_BATCH_INTERVAL = 10

# Threshold (meters) to detect and filter out GPS glitches. Any jump
# between consecutive points larger than this is treated as invalid.
GLITCH_THRESHOLD_METERS = 100.0

# Directories for model artifacts and logs. Keeping these centralized
# makes it easy to archive runs and inspect results.
MODEL_DIR = "./models"
LOG_DIR = os.path.join(MODEL_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =====================================================================
# LOGGING SYSTEM (Dual Logging + Timestamps)
# =====================================================================
# Design goals:
# - Real-time feedback via stdout
# - Persistent log across runs
# - Per-run log for detailed analysis
# - Human-readable timestamps

# Unique timestamp for this run, used to name the per-run log file.
run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

# Persistent log accumulates all runs.
persistent_log_path = os.path.join(LOG_DIR, "geolife.log")

# Per-run log contains only entries from this execution.
run_log_path = os.path.join(LOG_DIR, f"run_{run_timestamp}.log")


def log(msg: str) -> None:
    """
    Write high-level events to stdout AND both log files (UTF-8 safe).

    Intended for:
      - Phase boundaries (start/end of training, prediction, etc.)
      - Major milestones
      - Errors and important status messages
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {msg}"

    # stdout for immediate visibility
    print(entry)

    # persistent log (UTF-8)
    with open(persistent_log_path, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

    # per-run log (UTF-8)
    with open(run_log_path, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


# =====================================================================
# Utility: human-friendly ETA
# =====================================================================
def format_eta(seconds: float | None) -> str:
    """
    Convert a raw seconds estimate into a human-friendly ETA string.

    Used in training/prediction loops to give rough time remaining.
    """
    if seconds is None or seconds <= 0:
        return "ETA: unknown"
    m, s = divmod(int(seconds), 60)
    if m == 0:
        return f"ETA: {s}s"
    return f"ETA: {m}m {s}s"


# =====================================================================
# Load and preprocess trajectories
# =====================================================================
def load_trajectories(path: str) -> list[np.ndarray]:
    """
    Load trajectories from a parquet file and apply glitch filtering.

    Architecture:
      - Read parquet row-group by row-group to avoid loading everything
        into memory at once.
      - Group rows by 'source_file' to reconstruct individual trajectories.
      - Extract only configured FEATURES.
      - Compute per-step displacement to detect GPS glitches.
      - Filter out steps where displacement exceeds GLITCH_THRESHOLD_METERS.
      - Discard trajectories too short to support windowing.

    Returns:
      List of filtered trajectories, each as a NumPy array of shape
      [num_steps, num_features].
    """
    log("[RUN] Loading parquet and building trajectories...")
    pf = pq.ParquetFile(path)
    trajectories: list[np.ndarray] = []

    for rg in range(pf.num_row_groups):
        # Read one row-group at a time to control memory usage.
        tbl = pf.read_row_group(rg)
        df = tbl.to_pandas()

        # Group by source_file so each group represents one trajectory.
        for src, traj in df.groupby("source_file"):
            # Extract only the configured features and cast to float32
            # for PyTorch compatibility and memory efficiency.
            t = traj[FEATURES].to_numpy(dtype=np.float32)

            # Compute displacement between consecutive positions (x, y).
            coords = t[:, 0:2]
            deltas = np.linalg.norm(coords[1:] - coords[:-1], axis=1)

            # Build a mask that keeps the first point and any subsequent
            # point whose displacement is below the glitch threshold.
            mask = np.concatenate([[True], deltas <= GLITCH_THRESHOLD_METERS])
            t_filtered = t[mask]

            # Skip trajectories that cannot produce at least one window
            # of INPUT_WINDOW plus FUTURE_STEPS.
            if len(t_filtered) <= INPUT_WINDOW + FUTURE_STEPS:
                continue

            trajectories.append(t_filtered)

    log(f"[RUN] Total trajectories after filtering: {len(trajectories)}")
    return trajectories


def compute_normalization(trajectories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute global mean and std across all trajectories.

    Architectural choice:
      - Use global normalization so train/val/test share the same scale.
      - This avoids leakage because normalization is computed before
        splitting and applied consistently.

    Returns:
      (mean, std) vectors of shape [num_features].
    """
    log("[RUN] Computing global normalization stats...")
    # Concatenate all trajectories along time axis to compute stats.
    all_data = np.concatenate(trajectories, axis=0)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)

    # Avoid division-by-zero during normalization.
    std[std == 0] = 1.0

    log(f"[RUN] Normalization mean: {mean}")
    log(f"[RUN] Normalization std:  {std}")
    return mean, std


def split_trajectories(
    trajectories: list[np.ndarray],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Split trajectories into train/val/test sets at the trajectory level.

    Design decision:
      - Splitting by trajectory (not by window) avoids leakage where
        windows from the same trajectory appear in multiple splits.

    Returns:
      (train, val, test) lists of trajectories.
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
# Window generator
# =====================================================================
def stream_multistep_batches(
    trajectories: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
):
    """
    Generator that yields windowed inputs and multi-step targets.

    Processing steps:
      - Apply global normalization to each trajectory.
      - Slide a window of length INPUT_WINDOW across the trajectory.
      - For each window:
          * Extract features and transpose to [features, time] for TCN.
          * Compute FUTURE_STEPS absolute positions.
          * Convert absolute positions to displacements relative to the
            last position in the window.

    Yields:
      (windows_tensor, targets_tensor) per trajectory, where:
        windows_tensor: [batch_size, features, INPUT_WINDOW]
        targets_tensor: [batch_size, FUTURE_STEPS, 2]
    """
    total_windows = 0
    n_traj = len(trajectories)

    for idx, t in enumerate(trajectories, start=1):
        # Global normalization ensures consistent scaling across splits.
        t_norm = (t - mean) / std

        # Number of possible windows for this trajectory.
        n = len(t_norm) - INPUT_WINDOW - FUTURE_STEPS
        if n <= 0:
            continue

        windows = []
        targets = []

        for i in range(n):
            # Extract window and transpose to [features, time].
            w = t_norm[i:i + INPUT_WINDOW].T

            # Future absolute positions for the next FUTURE_STEPS.
            future_abs = t_norm[i + INPUT_WINDOW:i + INPUT_WINDOW + FUTURE_STEPS, 0:2]

            # Last position in the window is the reference point.
            last_pos = t_norm[i + INPUT_WINDOW - 1, 0:2]

            # Displacements relative to last window position.
            future_disp = future_abs - last_pos

            windows.append(w)
            targets.append(future_disp)

        batch_windows = len(windows)
        total_windows += batch_windows

        # stdout-only progress (not logged to avoid log noise).
        print(
            f"[TRAJ {idx}/{n_traj}] {len(t)} rows → "
            f"{batch_windows} windows (total_windows={total_windows})"
        )

        yield torch.tensor(np.stack(windows)), torch.tensor(np.stack(targets))


# =====================================================================
# Baseline
# =====================================================================
def baseline_predict(windows: np.ndarray) -> np.ndarray:
    """
    Simple constant-velocity baseline.

    Logic:
      - Use last two positions in the window.
      - Compute displacement between them.
      - Assume this displacement repeats for all FUTURE_STEPS.
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

    Architecture:
      - Reuse the same window generator as models.
      - Apply baseline_predict to each batch.
      - Accumulate predictions and targets for metrics.
      - Print heartbeats with CPU/RAM usage for observability.
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
            # heartbeat prints only (NOT logged)
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
# Metrics
# =====================================================================
def ade(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Average Displacement Error (ADE):
    Mean Euclidean distance across all timesteps and samples.
    """
    return float(np.linalg.norm(preds - targets, axis=2).mean())


def mde(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Maximum Displacement Error (MDE):
    Maximum Euclidean distance across all timesteps and samples.
    """
    return float(np.linalg.norm(preds - targets, axis=2).max())


def fde(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Final Displacement Error (FDE):
    Mean Euclidean distance at the final timestep across samples.
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
# Models
# =====================================================================
class GRUModel(nn.Module):
    """
    GRU-based sequence model for multi-step displacement prediction.

    Architecture:
      - Single GRU layer (batch_first=True).
      - Fully connected layer mapping last hidden state to FUTURE_STEPS*2.
      - Output reshaped to [batch, FUTURE_STEPS, 2].
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
        return out.view(-1, FUTURE_STEPS, 2)


class LSTMModel(nn.Module):
    """
    LSTM-based sequence model, analogous to GRUModel but with LSTM cells.
    """
    def __init__(self, input_size: int = 6, hidden_size: int = 64, future_steps: int = FUTURE_STEPS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, future_steps * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(-1, FUTURE_STEPS, 2)


class TCNBlock(nn.Module):
    """
    Single TCN block:
      - Dilated 1D convolution
      - BatchNorm
      - ReLU
      - No residual here (simplified baseline TCN)
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


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network (simplified):

    Architecture:
      - Stack of TCNBlock layers with increasing dilation.
      - Final FC layer on last timestep features.
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
        # TCN expects [batch, channels, time].
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # Use last timestep features.
        out = self.fc(out[:, :, -1])
        return out.view(-1, FUTURE_STEPS, 2)


# =====================================================================
# Training loop
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

    Design:
      - Shared training loop for all models (GRU/LSTM/TCN).
      - Uses MSE loss on displacement predictions.
      - Progress + heartbeat logging for observability.
      - max_batches caps per-epoch work for predictable runtime.
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
            # Convert to device and permute to [batch, time, features].
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

            # Progress logging every PROGRESS_BATCH_INTERVAL batches.
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

            # Heartbeat logging for long runs.
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().used / (1024**3)
                elapsed = now - global_start
                print(
                    f"[HEARTBEAT] Training... elapsed={elapsed:.1f}s, "
                    f"CPU={cpu:.1f}%, RAM={mem:.2f} GB"
                )
                last_heartbeat = now

            # Cap batches per epoch.
            if batches >= max_batches:
                break

        epoch_time = time.time() - epoch_start
        print(f"[TRAIN] Epoch {epoch} complete — avg loss {total_loss / max_batches:.4f} ({epoch_time:.2f}s)")

    total_time = time.time() - global_start
    log(f"[RUN] Training {model_name} complete — time={total_time:.2f}s")


# =====================================================================
# Prediction loop
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
    Run prediction for a trained model on given trajectories.

    Shared architecture:
      - Same window generator as training.
      - No gradients (torch.no_grad()).
      - Progress + heartbeat logging.
      - max_batches caps runtime.
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
# Saving + Loading
# =====================================================================
def save_model(model: nn.Module, name: str) -> None:
    """
    Save model state_dict to disk under MODEL_DIR.
    """
    path = os.path.join(MODEL_DIR, f"{name}_model.pt")
    torch.save(model.state_dict(), path)
    log(f"[SAVE] {name} model saved → {path}")


def save_normalization(mean: np.ndarray, std: np.ndarray) -> None:
    """
    Save global normalization stats as NumPy arrays.
    """
    np.save(os.path.join(MODEL_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(MODEL_DIR, "norm_std.npy"), std)
    log("[SAVE] Normalization stats saved → norm_mean.npy, norm_std.npy")


def save_metadata() -> None:
    """
    Save model metadata (features, window sizes) to JSON for reproducibility.
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
    Load a saved model from disk and return it in eval mode on DEVICE.
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
# Final summary table
# =====================================================================
def print_summary_table(results: dict[str, dict[str, float]]) -> None:
    """
    Print final metrics for all models in three formats:
      - ASCII table
      - Pretty per-model breakdown
      - JSON dump
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
# Main
# =====================================================================
def main() -> None:
    """
    Orchestrate the full GeoLife pipeline:

      - Load and preprocess trajectories
      - Compute and save normalization + metadata
      - Split into train/val/test
      - Evaluate baseline
      - Train + evaluate GRU, LSTM, TCN
      - Save models
      - Print final summary

    Any exception is logged and re-raised.
    """
    log("[RUN] GeoLife pipeline run started.")

    try:
        trajectories = load_trajectories(PARQUET_PATH)
        mean, std = compute_normalization(trajectories)
        train_traj, val_traj, test_traj = split_trajectories(trajectories)

        save_normalization(mean, std)
        save_metadata()

        results: dict[str, dict[str, float]] = {}

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
        train_model(tcn, "TCN", train_traj, mean, std)
        save_model(tcn, "tcn")

        tcn_val_preds, tcn_val_targets = predict_model(tcn, "TCN", val_traj, mean, std, phase_name="VAL")
        tcn_test_preds, tcn_test_targets = predict_model(tcn, "TCN", test_traj, mean, std, phase_name="TEST")
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
        log(f"[ERROR] Exception during run: {e}")
        raise


if __name__ == "__main__":
    main()
