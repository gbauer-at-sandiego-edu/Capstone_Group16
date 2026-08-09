"""
geolife_pipeline.py
====================

This module converts raw GeoLife .plt GPS logs into a unified,
analysis‑ready parquet dataset. It is intentionally engineered for:

    • Deterministic preprocessing
    • Windows‑safe multiprocessing (spawn mode)
    • Chunked parquet writing to avoid memory blow‑up
    • Stable schema enforcement for downstream ML pipelines
    • Reproducible kinematic feature generation

The output parquet file is consumed by the trajectory forecasting
models in `models_geolife_tuned.py`.

Every function in this module is designed to be:
    • Pure (no hidden global state mutations)
    • Serializable (safe for multiprocessing)
    • Schema‑stable (no dtype drift across chunks)
    • Fault‑tolerant (errors logged per‑file)

This file is heavily commented to explain *how* and *why* each step
exists, not just *what* it does.
"""

import os
import time
import psutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from multiprocessing import Pool
from pyproj import Proj

# Local import: loader for GeoLife .plt files
from load_plt import load_plt_file


# ======================================================================
# 1. PATHS AND PROJECTION SETUP
# ======================================================================
# Using Path() ensures OS‑independent behavior and avoids string‑based
# path bugs. This root directory contains:
#     • raw PLT files
#     • chunk output directory
#     • error logs
#     • final concatenated parquet
DATA_ROOT = Path(
    r"C:\Users\gb630\OneDrive\USD AAI\USD AAI\AAI-590 CAPSTONE\FINAL PROJECT\DATA"
)

# Chunk directory:
#     Each processed PLT file becomes one parquet chunk.
#     This avoids loading all data into memory at once.
CHUNK_DIR = DATA_ROOT / "chunks"

# Error directory:
#     Any file that fails processing gets a text log here.
ERROR_DIR = DATA_ROOT / "errors"

# Final parquet file:
#     All chunks are concatenated into this single dataset.
FINAL_PARQUET = DATA_ROOT / "geolife.parquet"

# Ensure directories exist so the pipeline is idempotent.
CHUNK_DIR.mkdir(exist_ok=True)
ERROR_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# UTM Projection Setup
# ----------------------------------------------------------------------
# GeoLife data was collected in Beijing, which lies in UTM Zone 48.
# UTM is chosen because:
#     • It converts lat/lon → meters
#     • It avoids distortions inherent in spherical coordinates
#     • It makes dx/dy meaningful for kinematic calculations
proj = Proj(proj="utm", zone=48, ellps="WGS84")

# Columns that must be numeric. Enforcing numeric types prevents:
#     • Parquet schema mismatches
#     • Downstream ML crashes due to dtype drift
numeric_cols = [
    "lat", "lon", "alt", "x", "y",
    "dx", "dy", "dt",
    "speed", "heading", "accel", "turn_rate"
]


# ======================================================================
# 1b. STABLE PARQUET SCHEMA
# ======================================================================
# A stable schema ensures:
#     • All chunks have identical column types
#     • The final parquet file is consistent
#     • ML pipelines can rely on float64 kinematic features
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
    """
    Enforce numeric and string types on the dataframe.

    WHY THIS EXISTS:
    ----------------
    GeoLife PLT files occasionally contain malformed numeric values
    (e.g., empty strings, corrupted altitudes). Parquet requires
    consistent column types across all chunks, and ML pipelines expect
    float64 for all kinematic features.

    This function ensures:
        • All numeric columns are float64
        • source_file is a string
        • Index is reset to avoid accidental index leakage

    Returns:
        A schema‑stable dataframe.
    """
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    if "source_file" in df.columns:
        df["source_file"] = df["source_file"].astype("string")

    return df.reset_index(drop=True)


# ======================================================================
# 2. WORKER FUNCTION (RUNS IN MULTIPROCESSING POOL)
# ======================================================================
def process_file(path_str: str):
    """
    Process a single GeoLife .plt file into a parquet chunk.

    WHY THIS FUNCTION IS SELF‑CONTAINED:
    ------------------------------------
    Windows uses "spawn" multiprocessing, meaning:
        • Each worker starts fresh (no shared globals)
        • Functions must be fully serializable
        • No hidden state can leak between workers

    PIPELINE STEPS:
    ---------------
    1. Load PLT file → DataFrame
    2. Resample timestamps to 1‑second intervals
    3. Project lat/lon → UTM (meters)
    4. Compute kinematic features:
         dx, dy, dt, speed, heading, accel, turn_rate
    5. Enforce schema
    6. Write parquet chunk

    Returns:
        ("OK", chunk_path, filename) or ("ERROR", None, filename)
    """
    path = Path(path_str)

    try:
        # --------------------------------------------------------------
        # Load raw PLT file
        # --------------------------------------------------------------
        # load_plt_file returns:
        #     timestamp, lat, lon, alt
        df = load_plt_file(path)

        # --------------------------------------------------------------
        # 1‑second resampling
        # --------------------------------------------------------------
        # GeoLife timestamps are irregular. Resampling:
        #     • normalizes temporal spacing
        #     • simplifies kinematic calculations
        #     • ensures dt ≈ 1s for most rows
        df = (
            df.set_index("timestamp")
              .resample("1s")
              .interpolate(method="time")
        )
        df.reset_index(inplace=True)
        df = df.drop_duplicates("timestamp")

        # --------------------------------------------------------------
        # UTM projection (meters)
        # --------------------------------------------------------------
        # Converts spherical coordinates → planar metric coordinates.
        x, y = proj(df["lon"].values, df["lat"].values)
        df["x"] = x
        df["y"] = y
        df = df.dropna(subset=["x", "y"])

        # --------------------------------------------------------------
        # Kinematic feature computation
        # --------------------------------------------------------------
        # dx, dy: displacement between consecutive points (meters)
        df["dx"] = df["x"].diff()
        df["dy"] = df["y"].diff()

        # dt: time delta in seconds
        df["dt"] = df["timestamp"].diff().dt.total_seconds()

        # Remove rows with non‑positive dt (duplicate timestamps)
        df = df[df["dt"] > 0]

        # speed: meters per second
        df["speed"] = np.sqrt(df["dx"]**2 + df["dy"]**2) / df["dt"]

        # heading: direction of travel (radians)
        df["heading"] = np.arctan2(df["dy"], df["dx"])

        # accel: change in speed over time
        df["accel"] = df["speed"].diff() / df["dt"]

        # turn_rate: change in heading over time
        df["turn_rate"] = df["heading"].diff() / df["dt"]

        # Remove infinities and NaNs
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Track which file this row came from
        df["source_file"] = path.name

        # Enforce stable schema
        df = enforce_schema(df)

        # --------------------------------------------------------------
        # Skip empty chunks
        # --------------------------------------------------------------
        if df.empty:
            err_path = ERROR_DIR / f"{path.stem}_empty.txt"
            with open(err_path, "w") as f:
                f.write("Empty dataframe after processing.")
            return ("ERROR", None, path.name)

        # --------------------------------------------------------------
        # Write parquet chunk
        # --------------------------------------------------------------
        chunk_path = CHUNK_DIR / f"{path.name}.parquet"

        # Convert DataFrame → Arrow Table → cast to stable schema
        tbl = pa.Table.from_pandas(df).cast(PARQUET_SCHEMA)
        pq.write_table(tbl, chunk_path)

        return ("OK", str(chunk_path), path.name)

    except Exception as e:
        # Any exception is logged to an error file for debugging.
        err_path = ERROR_DIR / f"{path.stem}_error.txt"
        with open(err_path, "w") as f:
            f.write(str(e))
        return ("ERROR", None, path.name)


# ======================================================================
# 3. MAIN PIPELINE ORCHESTRATION
# ======================================================================
def run_pipeline():
    """
    Main orchestration function.

    Responsibilities:
        • Discover all PLT files recursively
        • Spawn Windows‑safe multiprocessing pool
        • Process each PLT file → parquet chunk
        • Track progress, ETA, CPU, memory usage
        • Concatenate all chunks into final parquet file

    WHY THIS DESIGN:
    ----------------
    • Avoids loading all PLT files into memory
    • Avoids loading all processed data into memory
    • Robust to individual file failures
    • Safe on Windows (spawn mode)
    • Streaming parquet writer avoids memory blow‑up
    """
    plt_files = list(DATA_ROOT.rglob("*.plt"))
    print("PLT files discovered:", len(plt_files))

    start_time = time.time()

    # Windows uses "spawn" multiprocessing, meaning logical CPU count
    # often overstates usable parallelism. We conservatively estimate
    # physical cores to avoid oversubscription.
    logical_cpus = os.cpu_count()
    physical_cpus = logical_cpus // 2 if logical_cpus else 1
    workers = max(1, physical_cpus)

    print("System Info:")
    print(f"  Logical CPUs:        {logical_cpus}")
    print(f"  Physical CPUs (est): {physical_cpus}")
    print(f"  Worker Processes:    {workers}")
    print("--------------------------------------------------")

    results = []

    # --------------------------------------------------------------
    # Multiprocessing pool
    # --------------------------------------------------------------
    # Using imap_unordered:
    #     • results return as soon as workers finish
    #     • avoids blocking on slow files
    #     • improves throughput
    with Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_file, map(str, plt_files)), 1):
            status, chunk_path, name = res

            # Progress tracking
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

    # --------------------------------------------------------------
    # Separate successes and errors
    # --------------------------------------------------------------
    chunk_files = [Path(cp) for status, cp, name in results if status == "OK"]
    errors = [name for status, cp, name in results if status == "ERROR"]

    print("Errors:", len(errors))

    if not chunk_files:
        print("No chunks produced; aborting final parquet.")
        return

    # --------------------------------------------------------------
    # Concatenate parquet chunks
    # --------------------------------------------------------------
    print("Concatenating parquet chunks...")

    # ParquetWriter allows streaming writes without loading all chunks.
    writer = pq.ParquetWriter(FINAL_PARQUET, PARQUET_SCHEMA)

    batch_size = 250  # write chunks in batches to reduce I/O overhead
    for i in range(0, len(chunk_files), batch_size):
        batch = chunk_files[i:i+batch_size]
        for cf in batch:
            tbl = pq.read_table(cf).cast(PARQUET_SCHEMA)
            writer.write_table(tbl)

    writer.close()

    total_time = (time.time() - start_time) / 60
    print("Parquet file written:", FINAL_PARQUET)
    print("Total time:", round(total_time, 2), "minutes")


# ======================================================================
# 4. ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    run_pipeline()
