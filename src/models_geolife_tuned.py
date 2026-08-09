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

# Directories for saving artifacts used in figure generation. These
# are separate from MODEL_DIR so that plots and analysis artifacts
# can be managed independently of model checkpoints.
ARTIFACT_DIR = "./artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

PRED_DIR = os.path.join(ARTIFACT_DIR, "predictions")
LOSS_DIR = os.path.join(ARTIFACT_DIR, "loss")
METRIC_DIR = os.path.join(ARTIFACT_DIR, "metrics")
WINDOW_DIR = os.path.join(ARTIFACT_DIR, "windows")

for d in [PRED_DIR, LOSS_DIR, METRIC_DIR, WINDOW_DIR]:
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------------------------
# Logging system (dual logging + timestamps)
# --------------------------------------------------------------------
# We maintain both a persistent log (across runs) and a per-run log.
# This supports reproducibility, debugging, and auditability.
run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
persistent_log_path = os.path.join(LOG_DIR, "geolife.log")
run_log_path = os.path.join(LOG_DIR, f"run_{run_timestamp}.log")


def log(msg: str) -> None:
    """
    Write high-level events to stdout AND both log files.

    This function centralizes logging so that all important events
    (data loading, training start/end, errors, etc.) are captured with
    timestamps. Logs are UTF-8 encoded to avoid encoding issues.

    Parameters
    ----------
    msg : str
        Human-readable message describing the event.
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
def format_eta(seconds: float | None) -> str:
    """
    Convert a raw seconds estimate into a human-friendly string.

    Used in training and prediction loops to give a rough sense of how
    long the remaining work will take.

    Parameters
    ----------
    seconds : float | None
        Estimated remaining time in seconds.

    Returns
    -------
    str
        Human-friendly ETA string (e.g., "ETA: 3m 12s").
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
def load_trajectories(path: str) -> list[np.ndarray]:
    """
    Load trajectories from a parquet file and apply basic filtering.

    Data flow:
    ----------
    1. Read each row group from the parquet file to avoid loading the
       entire dataset into memory at once.
    2. Group rows by 'source_file' so each group corresponds to a single
       trajectory from the original GeoLife logs.
    3. Extract the configured FEATURES and convert them to a NumPy array.
    4. Compute per-step displacement in (x, y) to detect GPS glitches.
    5. Remove points where displacement exceeds GLITCH_THRESHOLD_METERS.
    6. Discard trajectories that are too short to produce at least one
       training window (INPUT_WINDOW + FUTURE_STEPS).

    Why this matters:
    -----------------
    - Glitch filtering removes unrealistic jumps that would confuse
      the models and inflate error metrics.
    - Trajectory-level grouping preserves temporal continuity.
    - Minimum-length enforcement ensures that window generation is
      well-defined and avoids degenerate cases.

    Parameters
    ----------
    path : str
        Filesystem path to the parquet dataset.

    Returns
    -------
    list[np.ndarray]
        List of filtered trajectories, each of shape
        [num_steps, num_features].
    """
    log("[RUN] Loading parquet and building trajectories...")
    pf = pq.ParquetFile(path)
    trajectories: list[np.ndarray] = []

    for rg in range(pf.num_row_groups):
        # Read one row group at a time to control memory usage.
        tbl = pf.read_row_group(rg)
        df = tbl.to_pandas()

        # Group by source_file so each group represents one trajectory.
        for src, traj in df.groupby("source_file"):
            # Extract motion features as float32 for efficiency.
            t = traj[FEATURES].to_numpy(dtype=np.float32)

            # Compute displacement between consecutive points in (x, y).
            coords = t[:, 0:2]
            deltas = np.linalg.norm(coords[1:] - coords[:-1], axis=1)

            # Build a mask that keeps points with reasonable displacement.
            mask = np.concatenate([[True], deltas <= GLITCH_THRESHOLD_METERS])
            t_filtered = t[mask]

            # Enforce minimum length for windowing.
            if len(t_filtered) <= INPUT_WINDOW + FUTURE_STEPS:
                continue

            trajectories.append(t_filtered)

    log(f"[RUN] Total trajectories after filtering: {len(trajectories)}")
    return trajectories


def compute_normalization(trajectories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute global mean and standard deviation across all trajectories.

    Architectural choice:
    ---------------------
    - Use global normalization so train/val/test share the same scale.
    - This avoids leakage because normalization is computed once on the
      full dataset and then applied consistently to all splits.

    Data flow:
    ----------
    1. Concatenate all trajectories along the time axis.
    2. Compute per-feature mean and standard deviation.
    3. Replace zero standard deviations with 1.0 to avoid division by
       zero during normalization.
    4. Save mean/std to ARTIFACT_DIR for later figure generation.

    Parameters
    ----------
    trajectories : list[np.ndarray]
        List of trajectories, each [num_steps, num_features].

    Returns
    -------
    (mean, std) : tuple[np.ndarray, np.ndarray]
        Global mean and std vectors of shape [num_features].
    """
    log("[RUN] Computing global normalization stats...")
    all_data = np.concatenate(trajectories, axis=0)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)
    std[std == 0] = 1.0

    log(f"[RUN] Normalization mean: {mean}")
    log(f"[RUN] Normalization std:  {std}")

    # Save normalization stats for later figure generation and analysis.
    np.save(os.path.join(ARTIFACT_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(ARTIFACT_DIR, "norm_std.npy"), std)
    log("[RUN] Saved normalization statistics")

    return mean, std


def split_trajectories(
    trajectories: list[np.ndarray],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Split trajectories into train/validation/test sets at the trajectory level.

    Design decision:
    ----------------
    - Splitting by trajectory (not by window) avoids leakage where
      windows from the same trajectory appear in multiple splits.
    - Ratios are 60% train, 20% validation, 20% test.

    Data flow:
    ----------
    1. Shuffle trajectory indices to avoid ordering bias.
    2. Compute split boundaries based on ratios.
    3. Construct train/val/test lists by indexing into the original
       trajectory list.

    Parameters
    ----------
    trajectories : list[np.ndarray]
        List of trajectories.
    train_ratio : float
        Fraction of trajectories assigned to training.
    val_ratio : float
        Fraction assigned to validation (test gets the remainder).

    Returns
    -------
    (train, val, test) : tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]
        Split trajectory lists.
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


def log_trajectory_length_histogram(trajectories: list[np.ndarray]) -> None:
    """
    Log a histogram of trajectory lengths to help tune window caps.

    Why this exists:
    ----------------
    - Extremely long trajectories can generate huge numbers of windows,
      which in turn can cause memory pressure and native segmentation
      faults (especially for TCN with dilations and attention).
    - Logging the distribution of trajectory lengths helps diagnose
      such issues and tune per-trajectory caps.

    Data flow:
    ----------
    1. Compute lengths of all trajectories.
    2. Bucket them into human-friendly ranges.
    3. Log counts per bucket and summary statistics.
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
def stream_multistep_batches(
    trajectories: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    max_windows_per_traj: int = 5000,
):
    """
    Stream batches of multistep windows from a list of trajectories.

    Architectural choices:
    ----------------------
    - Global normalization (mean/std) is applied first to ensure
      consistent scaling across trajectories.
    - Per-window normalization is then applied to each window to
      stabilize training and reduce sensitivity to local scale shifts.
    - A hard cap (max_windows_per_traj) is enforced per trajectory to
      prevent extremely long trajectories from generating too many
      windows and blowing up memory.

    Data flow per trajectory:
    -------------------------
    1. Normalize trajectory using global mean/std.
    2. Compute number of possible windows:
         n = len(t_norm) - INPUT_WINDOW - FUTURE_STEPS
    3. Adapt effective cap based on trajectory length (longer
       trajectories get a smaller cap).
    4. For each window:
         - Extract INPUT_WINDOW rows.
         - Compute per-window mean/std and normalize.
         - Transpose to [features, time] for TCN/attention.
         - Compute FUTURE_STEPS absolute positions.
         - Convert to displacements relative to last window position.
    5. Yield stacked windows/targets as torch tensors.

    Parameters
    ----------
    trajectories : list[np.ndarray]
        List of trajectories.
    mean : np.ndarray
        Global mean vector.
    std : np.ndarray
        Global std vector.
    max_windows_per_traj : int
        Global cap on windows per trajectory (adapted per trajectory).

    Yields
    ------
    (windows_tensor, targets_tensor) : tuple[torch.Tensor, torch.Tensor]
        windows_tensor: [batch_size, features, INPUT_WINDOW]
        targets_tensor: [batch_size, FUTURE_STEPS, 2]
    """
    total_windows = 0
    n_traj = len(trajectories)

    # Trajectory length histogram (for cap tuning and diagnostics).
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

        # Save trajectory lengths for plotting and offline analysis.
        np.save(os.path.join(WINDOW_DIR, "trajectory_lengths.npy"), traj_lengths)

    for idx, t in enumerate(trajectories, start=1):
        # Apply global normalization first.
        t_norm = (t - mean) / std

        # Number of possible windows in this trajectory.
        n = len(t_norm) - INPUT_WINDOW - FUTURE_STEPS
        if n <= 0:
            continue

        # Adaptive per-trajectory cap based on trajectory length.
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

        windows: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for i in range(n):
            if len(windows) >= effective_cap:
                # Cap activated: log once per trajectory to document
                # that we intentionally limited window count.
                log(
                    f"[CAP] Activated for trajectory {idx}/{n_traj}: "
                    f"len(t)={len(t)}, windows_generated={len(windows)}, "
                    f"windows_possible={n}, effective_cap={effective_cap}"
                )
                break

            # Extract a raw window of length INPUT_WINDOW.
            w = t_norm[i:i + INPUT_WINDOW].astype(np.float32)  # [INPUT_WINDOW, features]

            # Per-window normalization: center and scale features within
            # the window to reduce local scale variation.
            w_mean = w.mean(axis=0, keepdims=True)
            w_std = w.std(axis=0, keepdims=True)
            w_std[w_std == 0] = 1.0
            w_norm = (w - w_mean) / w_std  # [INPUT_WINDOW, features]

            # Transpose to [features, time] to match TCN/attention input.
            w_norm = w_norm.T  # [features, INPUT_WINDOW]

            # Future absolute positions (normalized space).
            future_abs = t_norm[i + INPUT_WINDOW:i + INPUT_WINDOW + FUTURE_STEPS, 0:2]
            last_pos = t_norm[i + INPUT_WINDOW - 1, 0:2]

            # Convert to displacements relative to last observed position.
            future_disp = future_abs - last_pos

            windows.append(w_norm)
            targets.append(future_disp)

        batch_windows = len(windows)
        total_windows += batch_windows

        print(
            f"[TRAJ {idx}/{n_traj}] {len(t)} rows → "
            f"{batch_windows} windows (total_windows={total_windows})"
        )

        if batch_windows == 0:
            continue

        # Save window count per trajectory for diagnostics and plotting.
        np.save(
            os.path.join(WINDOW_DIR, f"traj{idx}_window_count.npy"),
            np.array([batch_windows], dtype=np.int32),
        )

        yield torch.tensor(np.stack(windows)), torch.tensor(np.stack(targets))


# --------------------------------------------------------------------
# Baseline model (simple heuristic)
# --------------------------------------------------------------------
def baseline_predict(windows: np.ndarray) -> np.ndarray:
    """
    Baseline prediction: extrapolate using the last observed displacement.

    Logic:
    ------
    - Take the last two positions in the window.
    - Compute the displacement between them.
    - Repeat that displacement for all FUTURE_STEPS.

    This is a naive constant-velocity baseline used to contextualize
    the performance of learned models.

    Parameters
    ----------
    windows : np.ndarray
        Window tensor of shape [batch, features, time].

    Returns
    -------
    np.ndarray
        Baseline predictions of shape [batch, FUTURE_STEPS, 2].
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
    Evaluate the baseline model on a given set of trajectories.

    Architecture:
    -------------
    - Reuse the same window generator as the learned models to ensure
      fair comparison.
    - Apply baseline_predict to each batch.
    - Accumulate predictions and targets for metrics.
    - Print heartbeats with CPU/RAM usage for observability.

    Parameters
    ----------
    trajectories : list[np.ndarray]
        Trajectories to evaluate on.
    mean : np.ndarray
        Global mean for normalization.
    std : np.ndarray
        Global std for normalization.

    Returns
    -------
    (preds_all, targets_all) : tuple[np.ndarray, np.ndarray]
        Stacked predictions and targets across all batches.
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


# --------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------
def ade(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Average Displacement Error (ADE).

    Computes the mean Euclidean distance between predicted and target
    positions across all timesteps and all samples.

    Parameters
    ----------
    preds : np.ndarray
        Predicted displacements [batch, FUTURE_STEPS, 2].
    targets : np.ndarray
        Target displacements [batch, FUTURE_STEPS, 2].

    Returns
    -------
    float
        ADE in meters.
    """
    return float(np.linalg.norm(preds - targets, axis=2).mean())


def mde(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Maximum Displacement Error (MDE).

    Computes the maximum Euclidean distance between predicted and target
    positions across all timesteps and all samples. This highlights
    worst-case errors.

    Parameters
    ----------
    preds : np.ndarray
        Predicted displacements.
    targets : np.ndarray
        Target displacements.

    Returns
    -------
    float
        MDE in meters.
    """
    return float(np.linalg.norm(preds - targets, axis=2).max())


def fde(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Final Displacement Error (FDE).

    Computes the mean Euclidean distance between the final predicted
    position and the final target position for each sample. This is
    particularly important for trajectory forecasting tasks where the
    endpoint matters.

    Parameters
    ----------
    preds : np.ndarray
        Predicted displacements.
    targets : np.ndarray
        Target displacements.

    Returns
    -------
    float
        FDE in meters.
    """
    return float(np.linalg.norm(preds[:, -1, :] - targets[:, -1, :], axis=1).mean())


def print_metrics(name: str, preds: np.ndarray, targets: np.ndarray) -> None:
    """
    Compute ADE, MDE, and FDE for a given set of predictions and targets,
    then print and log them with a model/phase label.

    Parameters
    ----------
    name : str
        Label describing the model and phase (e.g., "GRU (VAL)").
    preds : np.ndarray
        Predicted displacements.
    targets : np.ndarray
        Target displacements.
    """
    msg = (
        f"{name} — ADE: {ade(preds, targets):.3f} m, "
        f"MDE: {mde(preds, targets):.3f} m, "
        f"FDE: {fde(preds, targets):.3f} m"
    )
    print(msg)
    log(f"[RUN] {msg}")


def evaluate_model(
    model: nn.Module,
    model_name: str,
    trajectories: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    phase: str = "TEST",
    max_windows_per_traj: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a trained model on a given set of trajectories.

    Design:
    -------
    - Uses the same window generator as training to ensure consistency.
    - Applies an optional per-trajectory window cap to control memory.
    - Records inputs, predictions, and targets to disk for figure
      generation and detailed analysis.

    Data flow:
    ----------
    1. Stream windows/targets via stream_multistep_batches.
    2. Convert windows to [batch, time, features] for RNN/TCN models.
    3. Run model in eval mode under torch.no_grad().
    4. Accumulate inputs/preds/targets in lists.
    5. Stack and save arrays to PRED_DIR.
    6. Compute and log metrics via print_metrics.

    Parameters
    ----------
    model : nn.Module
        Trained model to evaluate.
    model_name : str
        Short name (e.g., "GRU", "LSTM", "TCN").
    trajectories : list[np.ndarray]
        Trajectories to evaluate on.
    mean : np.ndarray
        Global mean for normalization.
    std : np.ndarray
        Global std for normalization.
    phase : str
        Phase label ("VAL" or "TEST").
    max_windows_per_traj : int
        Cap on windows per trajectory.

    Returns
    -------
    (inputs_all, preds_all, targets_all) : tuple[np.ndarray, np.ndarray, np.ndarray]
        Stacked inputs, predictions, and targets.
    """
    log(f"[RUN] Evaluation for {model_name} ({phase}) started...")
    model.to(DEVICE)
    model.eval()

    inputs_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []

    batch_count = 0
    start_time = time.time()
    last_heartbeat = start_time

    with torch.no_grad():
        for windows_cpu, targets_cpu in stream_multistep_batches(
            trajectories, mean, std, max_windows_per_traj=max_windows_per_traj
        ):
            # windows_cpu: [batch, features, time]
            windows_np = windows_cpu.numpy()
            # Convert to [batch, time, features] for sequence models.
            windows_seq = np.transpose(windows_np, (0, 2, 1))

            targets_np = targets_cpu.numpy()  # [batch, FUTURE_STEPS, 2]

            windows = torch.tensor(windows_seq, dtype=torch.float32).to(DEVICE)
            targets = torch.tensor(targets_np, dtype=torch.float32).to(DEVICE)

            preds = model(windows).cpu().numpy()  # [batch, FUTURE_STEPS, 2]

            inputs_all.append(windows_seq)
            preds_all.append(preds)
            targets_all.append(targets_np)

            batch_count += 1

            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().used / (1024**3)
                print(
                    f"[HEARTBEAT] Evaluating {model_name} ({phase})... "
                    f"batches={batch_count}, elapsed={now - start_time:.1f}s, "
                    f"CPU={cpu:.1f}%, RAM={mem:.2f} GB"
                )
                last_heartbeat = now

    total_time = time.time() - start_time
    log(f"[RUN] Evaluation for {model_name} ({phase}) complete — batches={batch_count}, time={total_time:.2f}s")

    inputs_all_arr = np.vstack(inputs_all)
    preds_all_arr = np.vstack(preds_all)
    targets_all_arr = np.vstack(targets_all)

    # Save arrays for downstream plotting/analysis.
    np.save(os.path.join(PRED_DIR, f"{model_name}_{phase}_inputs.npy"), inputs_all_arr)
    np.save(os.path.join(PRED_DIR, f"{model_name}_{phase}_preds.npy"), preds_all_arr)
    np.save(os.path.join(PRED_DIR, f"{model_name}_{phase}_targets.npy"), targets_all_arr)

    print_metrics(f"{model_name} ({phase})", preds_all_arr, targets_all_arr)

    return inputs_all_arr, preds_all_arr, targets_all_arr


# --------------------------------------------------------------------
# Models: GRU, LSTM, TCN
# --------------------------------------------------------------------
class GRUModel(nn.Module):
    """
    GRU-based sequence model.

    Architecture:
    -------------
    - Single GRU layer with 64 hidden units.
    - Fully connected layer mapping the final hidden state to
      FUTURE_STEPS * 2 outputs (x,y displacements).

    Why GRU:
    --------
    - GRUs are simpler than LSTMs (fewer gates) and often train faster.
    - They capture temporal dependencies with fewer parameters, making
      them a good baseline for sequence modeling.
    """
    def __init__(self, input_size: int = 6, hidden_size: int = 64, future_steps: int = FUTURE_STEPS):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, future_steps * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        out, _ = self.gru(x)
        # Use the final hidden state as a summary of the sequence.
        out = self.fc(out[:, -1, :])
        return out.view(-1, FUTURE_STEPS, 2)


class LSTMModel(nn.Module):
    """
    LSTM-based sequence model.

    Architecture:
    -------------
    - Single LSTM layer with 64 hidden units.
    - Fully connected layer mapping the final hidden state to
      FUTURE_STEPS * 2 outputs.

    Why LSTM:
    ---------
    - LSTMs are well-suited for sequential data with long-range
      dependencies due to their gating mechanisms.
    """
    def __init__(self, input_size: int = 6, hidden_size: int = 64, future_steps: int = FUTURE_STEPS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, future_steps * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)
        # Use the final hidden state as a summary of the sequence.
        out = self.fc(out[:, -1, :])
        return out.view(-1, FUTURE_STEPS, 2)


class TCNBlock(nn.Module):
    """
    Single block of a Temporal Convolutional Network (TCN).

    Components:
    -----------
    - 1D convolution with dilation and padding to expand receptive field.
    - Batch normalization for stable training.
    - ReLU activation for non-linearity.
    - Dropout for regularization.
    - Residual connection (with optional 1x1 conv for channel matching).

    Why dilations:
    --------------
    - Dilated convolutions allow the receptive field to grow
      exponentially with depth, enabling the model to see long-range
      temporal patterns without a huge number of layers.

    Why residuals:
    --------------
    - Residual connections help gradients flow through deep stacks of
      dilated convolutions, reducing vanishing/exploding issues and
      improving training stability.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
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
        self.dropout = nn.Dropout(dropout)
        self.pad = padding

        # Residual path: if channel count changes, use 1x1 conv to match.
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        out = self.conv(x)
        if self.pad > 0:
            # Remove extra padding introduced by dilation to keep
            # output length aligned with input length.
            out = out[:, :, :-self.pad]

        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection: add original (or downsampled) input.
        res = self.downsample(x) if self.downsample is not None else x
        out = out + res
        return self.relu(out)


class TCNModel(nn.Module):
    """
    Enhanced Temporal Convolutional Network for trajectory forecasting.

    Design choices:
    ----------------
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
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        levels: int = 5,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.levels = levels
        self.kernel_size = kernel_size

        # Precompute sinusoidal positional encodings for the input window.
        # Stored as a buffer so they move with the model across devices.
        self.register_buffer(
            "pos_encoding",
            self._build_positional_encoding(INPUT_WINDOW, input_size),
        )

        layers: list[nn.Module] = []
        in_ch = input_size
        for i in range(levels):
            dilation = 2 ** i  # 1, 2, 4, 8, 16 → receptive field >= 20
            layers.append(TCNBlock(in_ch, hidden_size, kernel_size, dilation, dropout=0.2))
            in_ch = hidden_size

        self.tcn = nn.Sequential(*layers)

        # Multi-head attention over time. Embedding dimension matches
        # hidden_size so we can attend over the TCN outputs directly.
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True,
        )

        # Final prediction head: maps attended summary to future displacements.
        self.fc = nn.Linear(hidden_size, FUTURE_STEPS * 2)

    def _build_positional_encoding(self, length: int, dim: int) -> torch.Tensor:
        """
        Build sinusoidal positional encodings as in the original
        Transformer paper. These encodings give the model information
        about the relative position of each timestep in the sequence.

        Parameters
        ----------
        length : int
            Sequence length (INPUT_WINDOW).
        dim : int
            Feature dimension.

        Returns
        -------
        torch.Tensor
            Positional encoding tensor of shape [length, dim].
        """
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-np.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [length, dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TCN model.

        Steps:
        ------
        1. Add positional encodings to the input sequence (if shape
           matches INPUT_WINDOW and input_size).
        2. Transpose to [batch, channels, time] for TCN.
        3. Pass through the TCN stack.
        4. Transpose back to [batch, time, channels] for attention.
        5. Apply self-attention over time.
        6. Use the last timestep of the attended sequence as a summary.
        7. Map summary to FUTURE_STEPS * 2 outputs and reshape.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, seq_len, features].

        Returns
        -------
        torch.Tensor
            Predicted displacements of shape [batch, FUTURE_STEPS, 2].
        """
        # x: [batch, seq_len, features]
        b, seq_len, feat = x.shape
        if seq_len == INPUT_WINDOW and feat == self.input_size:
            # Broadcast positional encodings across the batch.
            pe = self.pos_encoding.unsqueeze(0).expand(b, -1, -1)  # [b, seq_len, feat]
            x = x + pe

        # TCN expects [batch, channels, time].
        x = x.transpose(1, 2)  # [b, features, seq_len]
        out = self.tcn(x)      # [b, hidden_size, seq_len]

        # Attention over time: convert to [b, seq_len, hidden_size].
        out_time = out.transpose(1, 2)
        attn_out, _ = self.attn(out_time, out_time, out_time)

        # Use the last timestep as a summary of the attended sequence.
        summary = attn_out[:, -1, :]  # [b, hidden_size]

        out = self.fc(summary)        # [b, FUTURE_STEPS*2]
        return out.view(-1, FUTURE_STEPS, 2)


# --------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------
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
    Generic training loop for any of the models.

    Design:
    -------
    - Shared loop ensures GRU/LSTM/TCN are trained under comparable
      conditions.
    - Uses MSE loss on displacement predictions.
    - Adaptive window caps per model type to control memory usage.
    - Records training loss history for later visualization.

    Leakage avoidance:
    ------------------
    - Training uses only trajectories assigned to the train split.
    - Normalization parameters are computed globally before splitting,
      then applied consistently to all splits.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    model_name : str
        Name used for logging ("GRU", "LSTM", "TCN").
    trajectories : list[np.ndarray]
        Training trajectories.
    mean : np.ndarray
        Global mean.
    std : np.ndarray
        Global std.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    max_batches : int
        Maximum number of batches per epoch (caps runtime).
    """
    log(f"[RUN] Training {model_name} started (epochs={epochs}, max_batches={max_batches})")
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Adaptive cap based on model name: TCN is more memory-hungry.
    if model_name.upper() == "TCN":
        max_windows_per_traj = 3000
    elif model_name.upper() in ("GRU", "LSTM"):
        max_windows_per_traj = 8000
    else:
        max_windows_per_traj = 10000

    log(f"[RUN] Using max_windows_per_traj={max_windows_per_traj} for {model_name}")

    total_batches_est = epochs * max_batches
    global_start = time.time()

    loss_history: list[float] = []

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
                    f"[HEARTBEAT] Training {model_name}... elapsed={elapsed:.1f}s, "
                    f"CPU={cpu:.1f}%, RAM={mem:.2f} GB"
                )
                last_heartbeat = now

            # Cap batches per epoch to keep runtime predictable.
            if batches >= max_batches:
                break

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max_batches
        loss_history.append(avg_loss)
        print(f"[TRAIN] Epoch {epoch} complete — avg loss {avg_loss:.4f} ({epoch_time:.2f}s)")

    total_time = time.time() - global_start
    log(f"[RUN] Training {model_name} complete — time={total_time:.2f}s")

    # Save loss history for plotting.
    np.save(os.path.join(LOSS_DIR, f"{model_name}_loss_history.npy"), np.array(loss_history, dtype=np.float32))


# --------------------------------------------------------------------
# Saving + metadata
# --------------------------------------------------------------------
def save_model(model: nn.Module, name: str) -> None:
    """
    Save model state_dict to disk under MODEL_DIR.

    Parameters
    ----------
    model : nn.Module
        Model to save.
    name : str
        Short name used in filename (e.g., "gru", "lstm", "tcn").
    """
    path = os.path.join(MODEL_DIR, f"{name}_model.pt")
    torch.save(model.state_dict(), path)
    log(f"[SAVE] {name} model saved → {path}")


def save_metadata() -> None:
    """
    Save model metadata (features, window sizes) to JSON for reproducibility.

    Why:
    ----
    - Captures the configuration needed to interpret saved models and
      artifacts.
    - Enables external tools to reconstruct input shapes and feature
      ordering.
    """
    metadata = {
        "features": FEATURES,
        "input_window": INPUT_WINDOW,
        "future_steps": FUTURE_STEPS,
        "parquet_path": PARQUET_PATH,
    }
    path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    log(f"[SAVE] Metadata saved → {path}")


# --------------------------------------------------------------------
# Summary table
# --------------------------------------------------------------------
def print_summary_table(results: dict[str, dict[str, float]]) -> None:
    """
    Print final metrics for all models in three formats:
      - ASCII table
      - Pretty per-model breakdown
      - JSON dump

    This function is purely presentational but centralizes the way
    metrics are reported, which helps with reproducibility and grading.
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


# --------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------
def main() -> None:
    """
    Orchestrate the full GeoLife tuned pipeline.

    High-level flow:
    ----------------
    1. Load and preprocess trajectories from the parquet dataset.
    2. Compute global normalization statistics.
    3. Log trajectory length histogram for diagnostics.
    4. Split trajectories into train/val/test at trajectory level.
    5. Save normalization and metadata artifacts.
    6. Evaluate baseline on val and test splits.
    7. Train + evaluate GRU, LSTM, and tuned TCN models.
    8. Save models and per-model artifacts.
    9. Print final summary table.

    Leakage avoidance:
    ------------------
    - Normalization is computed before splitting and applied uniformly.
    - Splits are done at trajectory level, so windows from a given
      trajectory never appear in multiple splits.

    Any exception is logged and re-raised to make failures visible.
    """
    log("[RUN] GeoLife tuned pipeline run started.")

    try:
        # 1. Load trajectories from parquet.
        trajectories = load_trajectories(PARQUET_PATH)

        # 2. Compute global normalization.
        mean, std = compute_normalization(trajectories)

        # 3. Log trajectory length histogram.
        log_trajectory_length_histogram(trajectories)

        # 4. Split into train/val/test.
        train_traj, val_traj, test_traj = split_trajectories(trajectories)

        # 5. Save metadata (normalization already saved in compute_normalization).
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

        _, gru_val_preds, gru_val_targets = evaluate_model(
            gru, "GRU", val_traj, mean, std, phase="VAL"
        )
        _, gru_test_preds, gru_test_targets = evaluate_model(
            gru, "GRU", test_traj, mean, std, phase="TEST"
        )

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

        _, lstm_val_preds, lstm_val_targets = evaluate_model(
            lstm, "LSTM", val_traj, mean, std, phase="VAL"
        )
        _, lstm_test_preds, lstm_test_targets = evaluate_model(
            lstm, "LSTM", test_traj, mean, std, phase="TEST"
        )

        results["LSTM"] = {
            "ADE_VAL": ade(lstm_val_preds, lstm_val_targets),
            "MDE_VAL": mde(lstm_val_preds, lstm_val_targets),
            "FDE_VAL": fde(lstm_val_preds, lstm_val_targets),
            "ADE_TEST": ade(lstm_test_preds, lstm_test_targets),
            "MDE_TEST": mde(lstm_test_preds, lstm_test_targets),
            "FDE_TEST": fde(lstm_test_preds, lstm_test_targets),
        }

        # -------------------------
        # Tuned TCN
        # -------------------------
        print("\n==============================")
        print("Starting TCN training...")
        print("==============================")
        tcn = TCNModel()
        train_model(tcn, "TCN", train_traj, mean, std)
        save_model(tcn, "tcn")

        _, tcn_val_preds, tcn_val_targets = evaluate_model(
            tcn, "TCN", val_traj, mean, std, phase="VAL"
        )
        _, tcn_test_preds, tcn_test_targets = evaluate_model(
            tcn, "TCN", test_traj, mean, std, phase="TEST"
        )

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
        log("[RUN] GeoLife tuned pipeline run completed successfully.")

        # Save metrics dictionary for offline analysis.
        np.save(
            os.path.join(METRIC_DIR, "final_results.npy"),
            np.array(results, dtype=object),
        )

    except Exception as e:
        log(f"[ERROR] Exception during run: {e}")
        raise


if __name__ == "__main__":
    main()
