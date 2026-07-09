# geolife_pipeline.py
# GeoLife PLT → UTM → kinematics → parquet chunk processor
# Standalone Windows-safe multiprocessing script

import os
import time
import psutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from multiprocessing import Pool, cpu_count
from pyproj import Proj

from load_plt import load_plt_file  # must be in same directory as this module

# -------------------------------
# 1. Paths and projection
# -------------------------------
DATA_ROOT = Path(
    r"C:\Users\gb630\OneDrive\USD AAI\USD AAI\AAI-590 CAPSTONE\FINAL PROJECT\DATA"
)

CHUNK_DIR = DATA_ROOT / "chunks"
ERROR_DIR = DATA_ROOT / "errors"
FINAL_PARQUET = DATA_ROOT / "geolife.parquet"

CHUNK_DIR.mkdir(exist_ok=True)
ERROR_DIR.mkdir(exist_ok=True)

proj = Proj(proj="utm", zone=48, ellps="WGS84")

numeric_cols = [
    "lat", "lon", "alt", "x", "y",
    "dx", "dy", "dt",
    "speed", "heading", "accel", "turn_rate"
]

# -------------------------------
# 1b. Stable Parquet Schema
# -------------------------------
PARQUET_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("ns")),
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("alt", pa.float64()),
    ("x", pa.float64()),
    ("y", pa.float64()),
    ("dx", pa.float64()),
    ("dy", pa.float64()),
    ("dt", pa.float64()),
    ("speed", pa.float64()),
    ("heading", pa.float64()),
    ("accel", pa.float64()),
    ("turn_rate", pa.float64()),
    ("source_file", pa.string()),
])

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    # enforce numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # enforce string column
    if "source_file" in df.columns:
        df["source_file"] = df["source_file"].astype("string")

    return df.reset_index(drop=True)

# -------------------------------
# 2. Worker function
# -------------------------------
def process_file(path_str: str):
    path = Path(path_str)
    try:
        df = load_plt_file(path)

        # 1-second resampling
        df = (
            df.set_index("timestamp")
              .resample("1s")
              .interpolate(method="time")
        )
        df.reset_index(inplace=True)
        df = df.drop_duplicates("timestamp")

        # UTM projection
        x, y = proj(df["lon"].values, df["lat"].values)
        df["x"] = x
        df["y"] = y
        df = df.dropna(subset=["x", "y"])

        # Kinematics
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

        # Skip empty chunks
        if df.empty:
            err_path = ERROR_DIR / f"{path.stem}_empty.txt"
            with open(err_path, "w") as f:
                f.write("Empty dataframe after processing.")
            return ("ERROR", None, path.name)

        # Write chunk
        chunk_path = CHUNK_DIR / f"{path.name}.parquet"
        tbl = pa.Table.from_pandas(df).cast(PARQUET_SCHEMA)
        pq.write_table(tbl, chunk_path)

        return ("OK", str(chunk_path), path.name)

    except Exception as e:
        err_path = ERROR_DIR / f"{path.stem}_error.txt"
        with open(err_path, "w") as f:
            f.write(str(e))
        return ("ERROR", None, path.name)

# -------------------------------
# 3. Main pipeline
# -------------------------------
def run_pipeline():
    plt_files = list(DATA_ROOT.rglob("*.plt"))
    print("PLT files discovered:", len(plt_files))

    start_time = time.time()
    logical_cpus = os.cpu_count()
    physical_cpus = logical_cpus // 2 if logical_cpus else 1
    workers = max(1, physical_cpus)

    print("System Info:")
    print(f"  Logical CPUs:       {logical_cpus}")
    print(f"  Physical CPUs (est): {physical_cpus}")
    print(f"  Worker Processes:    {workers}")
    print("--------------------------------------------------")

    results = []
    with Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_file, map(str, plt_files)), 1):
            status, chunk_path, name = res

            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(plt_files) - i) / rate if rate > 0 else 0

            cpu_usage = psutil.cpu_percent(interval=0.1)
            mem_usage = psutil.virtual_memory().percent

            print(
                f"Processed {i}/{len(plt_files)} | "
                f"Elapsed: {elapsed/60:.2f} min | "
                f"ETA: {eta/60:.2f} min | "
                f"CPU: {cpu_usage}% | "
                f"Mem: {mem_usage}%",
                end="\r"
            )

            results.append(res)

    print("\nChunk writing complete.")

    chunk_files = [Path(cp) for status, cp, name in results if status == "OK"]
    errors = [name for status, cp, name in results if status == "ERROR"]

    print("Errors:", len(errors))

    if not chunk_files:
        print("No chunks produced; aborting final parquet.")
        return

    print("Concatenating parquet chunks...")

    writer = pq.ParquetWriter(FINAL_PARQUET, PARQUET_SCHEMA)

    batch_size = 250
    for i in range(0, len(chunk_files), batch_size):
        batch = chunk_files[i:i+batch_size]
        for cf in batch:
            tbl = pq.read_table(cf).cast(PARQUET_SCHEMA)
            writer.write_table(tbl)

    writer.close()

    total_time = (time.time() - start_time) / 60
    print("Parquet file written:", FINAL_PARQUET)
    print("Total time:", round(total_time, 2), "minutes")

# -------------------------------
# 4. Entry point
# -------------------------------
if __name__ == "__main__":
    run_pipeline()
