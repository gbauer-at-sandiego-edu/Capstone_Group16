import numpy as np
import pandas as pd
from pyproj import Proj

proj = Proj(proj="utm", zone=48, ellps="WGS84")

numeric_cols = [
    "lat", "lon", "alt", "x", "y",
    "dx", "dy", "dt",
    "speed", "heading", "accel", "turn_rate"
]

def enforce_schema(df):
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df.reset_index(drop=True)

def process_file(path):
    from your_notebook_imports import load_plt_file  # adjust import

    df = load_plt_file(path)

    df = (
        df.set_index("timestamp")
          .resample("1s")
          .interpolate(method="time")
    )
    df.reset_index(inplace=True)
    df = df.drop_duplicates("timestamp")

    x, y = proj(df["lon"].values, df["lat"].values)
    df["x"] = x
    df["y"] = y
    df = df.dropna(subset=["x", "y"])

    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["dt"] = df["timestamp"].diff().dt.total_seconds()
    df = df[df["dt"] > 0]

    df["speed"] = np.sqrt(df["dx"]**2 + df["dy"]**2) / df["dt"]
    df["heading"] = np.arctan2(df["dy"], df["dx"])
    df["accel"] = df["speed"].diff() / df["dt"]
    df["turn_rate"] = df["heading"].diff() / df["dt"]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df["source_file"] = path.name

    df = enforce_schema(df)

    return ("OK", df, path.name)
