import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ARTIFACT_DIR = "./artifacts"
PRED_DIR = os.path.join(ARTIFACT_DIR, "predictions")
LOSS_DIR = os.path.join(ARTIFACT_DIR, "loss")
METRIC_DIR = os.path.join(ARTIFACT_DIR, "metrics")
WINDOW_DIR = os.path.join(ARTIFACT_DIR, "windows")

sns.set(style="whitegrid")


def load_loss_curve(model_name):
    path = os.path.join(LOSS_DIR, f"{model_name}_loss.npy")
    return np.load(path)


def plot_loss_curve(model_name):
    loss = load_loss_curve(model_name)
    plt.figure(figsize=(10, 5))
    plt.plot(loss, label=f"{model_name} Loss", color="blue")
    plt.title(f"Training Loss Curve for {model_name}")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def load_predictions(model_name, phase):
    preds = np.load(os.path.join(PRED_DIR, f"{model_name}_{phase}_preds.npy"))
    targets = np.load(os.path.join(PRED_DIR, f"{model_name}_{phase}_targets.npy"))
    return preds, targets


def plot_error_distribution(model_name, phase):
    preds, targets = load_predictions(model_name, phase)
    errors = np.linalg.norm(preds - targets, axis=2).flatten()

    plt.figure(figsize=(10, 5))
    sns.histplot(errors, bins=50, kde=True, color="green")
    plt.title(f"Error Distribution for {model_name} ({phase})")
    plt.xlabel("Displacement Error (meters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_scatter_pred_vs_true(model_name, phase):
    preds, targets = load_predictions(model_name, phase)

    plt.figure(figsize=(8, 8))
    plt.scatter(targets[:, -1, 0], targets[:, -1, 1],
                s=10, alpha=0.5, label="True Final Position", color="black")
    plt.scatter(preds[:, -1, 0], preds[:, -1, 1],
                s=10, alpha=0.5, label="Predicted Final Position", color="red")

    plt.title(f"Predicted vs True Final Displacements ({model_name}, {phase})")
    plt.xlabel("X Displacement")
    plt.ylabel("Y Displacement")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_trajectory_overlay(model_name, phase, sample_index=0):
    preds, targets = load_predictions(model_name, phase)

    true_traj = targets[sample_index]
    pred_traj = preds[sample_index]

    plt.figure(figsize=(8, 6))
    plt.plot(true_traj[:, 0], true_traj[:, 1],
             marker="o", label="True Future Path", color="black")
    plt.plot(pred_traj[:, 0], pred_traj[:, 1],
             marker="x", label="Predicted Future Path", color="red")

    plt.title(f"Trajectory Overlay for {model_name} ({phase})")
    plt.xlabel("X Displacement")
    plt.ylabel("Y Displacement")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_window_histogram():
    lengths = np.load(os.path.join(WINDOW_DIR, "trajectory_lengths.npy"))

    plt.figure(figsize=(10, 5))
    sns.histplot(lengths, bins=50, color="purple")
    plt.title("Trajectory Length Distribution")
    plt.xlabel("Trajectory Length (samples)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_model_comparison_table():
    import json

    models = ["Baseline", "GRU", "LSTM", "TCN"]
    rows = []

    for m in models:
        path = os.path.join(METRIC_DIR, f"{m}_epoch1.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            rows.append([m, data["avg_loss"], data["epoch_time"]])

    import pandas as pd
    df = pd.DataFrame(rows, columns=["Model", "Avg Loss (Epoch 1)", "Epoch Time (s)"])
    print(df)


if __name__ == "__main__":
    # Example usage
    plot_loss_curve("LSTM")
    plot_error_distribution("LSTM", "TEST")
    plot_scatter_pred_vs_true("LSTM", "TEST")
    plot_trajectory_overlay("LSTM", "TEST", sample_index=10)
    plot_window_histogram()
    plot_model_comparison_table()
