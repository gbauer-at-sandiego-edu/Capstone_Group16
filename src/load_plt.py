# load_plt.py
import pandas as pd

def load_plt_file(path):
    df = pd.read_csv(
        path,
        skiprows=6,
        header=None,
        names=["lat", "lon", "unused", "alt", "days", "date", "time"]
    )

    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"])

    dt = df["timestamp"].diff().dt.total_seconds()
    df = df[dt > 0]

    return df[["timestamp", "lat", "lon", "alt"]]
