# geolife_pipeline.py
# GeoLife PLT → UTM → kinematics → parquet chunk processor
# Standalone Windows-safe multiprocessing script
#
# This module converts raw GeoLife .plt GPS logs into a unified,
# analysis-ready parquet dataset. It is designed for:
#   - deterministic preprocessing
#   - Windows-safe multiprocessing
#   - chunked parquet writing to avoid memory blow-up
#   - stable schema enforcement for downstream ML pipelines
#
# The output parquet file is consumed by the trajectory forecasting
# models in models_geolife.py.

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

# Local import: loader for GeoLife .plt files
# This must be in the same directory as this module.
from load_plt import load_plt_file

# ================================================================
# 1. PATHS AND PROJECTION SETUP
# ================================================================
# DATA_ROOT is the root directory containing raw PLT files and
# output directories. Using Path() ensures OS-independent behavior.
DATA_ROOT = Path(
    r"C:\Users\gb630\OneDrive\USD AAI\USD AAI\AAI-590 CAPSTONE\FINAL PROJECT\DATA"
)

# CHUNK_DIR: where each processed PLT file is written as a parquet chunk.
# ERROR_DIR: where error logs for failed PLT files are written.
# FINAL_PARQUET: the final concatenated parquet file.
CHUNK_DIR = DATA_ROOT / "chunks"
ERROR_DIR = DATA_ROOT / "errors"
FINAL_PARQUET = DATA_ROOT / "geolife.parquet"

# Ensure directories exist. This avoids runtime errors and makes the
# pipeline idempotent.
CHUNK_DIR.mkdir(exist_ok=True)
ERROR_DIR.mkdir(exist_ok=True)

# UTM projection (zone 48 for Beijing region where GeoLife was collected).
# UTM is chosen because:
#   - It provides metric coordinates (meters)
#   - It avoids distortions inherent in lat/lon for distance calculations
proj = Proj(proj="utm", zone=48, ellps="WGS84")

# Columns that must be numeric. Enforcing numeric types prevents
# downstream parquet schema mismatches and ML model crashes.
numeric_cols = [
    "lat", "lon", "alt", "x", "y",
    "dx", "dy", "dt",
    "speed", "heading", "accel", "turn_rate"
]

# ================================================================
# 1b. STABLE PARQUET SCHEMA
# ================================================================
# A stable schema ensures:
#   - All chunks have identical column types
#   - The final concatenated parquet file is consistent
#   - Downstream ML pipelines can rely on fixed dtypes
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

    Why this matters:
    - PLT files sometimes contain malformed numeric values.
    - Parquet requires consistent column types across chunks.
    - ML pipelines expect float64 for all kinematic features.

    This function ensures the dataframe conforms to PARQUET_SCHEMA.
    """
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    if "source_file" in df.columns:
        df["source_file"] = df["source_file"].astype("string")

    return df.reset_index(drop=True)

# ================================================================
# 2. WORKER FUNCTION (RUNS IN MULTIPROCESSING POOL)
# ================================================================
def process_file(path_str: str):
    """
    Process a single GeoLife .plt file into a parquet chunk.

    Steps:
    1. Load PLT file into a dataframe.
    2. Resample to 1-second intervals (GeoLife timestamps are irregular).
    3. Project lat/lon → UTM (meters).
    4. Compute kinematic features:
         dx, dy, dt, speed, heading, accel, turn_rate
    5. Enforce schema and remove invalid rows.
    6. Write parquet chunk.

    This function is intentionally self-contained so it can run safely
    inside Windows multiprocessing (spawn mode).
    """
    path = Path(path_str)
    try:
        # Load raw PLT file (lat, lon, alt, timestamp)
        df = load_plt_file(path)

        # ----------------------------------------------------------
        # 1-second resampling
        # ----------------------------------------------------------
        # GeoLife timestamps are irregular. Resampling:
        #   - normalizes temporal spacing
        #   - simplifies kinematic calculations
        #   - ensures consistent dt = 1s for most rows
        df = (
            df.set_index("timestamp")
              .resample("1s")
              .interpolate(method="time")
        )
        df.reset_index(inplace=True)
        df = df.drop_duplicates("timestamp")

        # ----------------------------------------------------------
        # UTM projection (meters)
        # ----------------------------------------------------------
        # Using UTM allows dx/dy to be interpreted directly as meters.
        x, y = proj(df["lon"].values, df["lat"].values)
        df["x"] = x
        df["y"] = y
        df = df.dropna(subset=["x", "y"])

        # ----------------------------------------------------------
        # Kinematic feature computation
        # ----------------------------------------------------------
        # dx, dy: displacement between consecutive points
        # dt: time delta in seconds
        df["dx"] = df["x"].diff()
        df["dy"] = df["y"].diff()
        df["dt"] = df["timestamp"].diff().dt.total_seconds()

        # Remove rows with non-positive dt (duplicate timestamps)
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

        # ----------------------------------------------------------
        # Skip empty chunks
        # ----------------------------------------------------------
        if df.empty:
            err_path = ERROR_DIR / f"{path.stem}_empty.txt"
            with open(err_path, "w") as f:
                f.write("Empty dataframe after processing.")
            return ("ERROR", None, path.name)

        # ----------------------------------------------------------
        # Write parquet chunk
        # ----------------------------------------------------------
        chunk_path = CHUNK_DIR / f"{path.name}.parquet"
        tbl = pa.Table.from_pandas(df).cast(PARQUET_SCHEMA)
        pq.write_table(tbl, chunk_path)

        return ("OK", str(chunk_path), path.name)

    except Exception as e:
        # Any exception is logged to an error file for debugging.
        err_path = ERROR_DIR / f"{path.stem}_error.txt"
        with open(err_path, "w") as f:
            f.write(str(e))
        return ("ERROR", None, path.name)

# ================================================================
# 3. MAIN PIPELINE
# ================================================================
def run_pipeline():
    """
    Main orchestration function.

    Responsibilities:
    - Discover all PLT files recursively.
    - Spawn a Windows-safe multiprocessing pool.
    - Process each PLT file into a parquet chunk.
    - Track progress, ETA, CPU, and memory usage.
    - Concatenate all chunks into a final parquet file.

    This design:
    - avoids loading all PLT files into memory
    - avoids loading all processed data into memory
    - is robust to individual file failures
    - is safe on Windows (spawn mode)
    """
    plt_files = list(DATA_ROOT.rglob("*.plt"))
    print("PLT files discovered:", len(plt_files))

    start_time = time.time()

    # Windows uses "spawn" multiprocessing, so we avoid cpu_count()
    # assumptions and estimate physical cores conservatively.
    logical_cpus = os.cpu_count()
    physical_cpus = logical_cpus // 2 if logical_cpus else 1
    workers = max(1, physical_cpus)

    print("System Info:")
    print(f"  Logical CPUs:       {logical_cpus}")
    print(f"  Physical CPUs (est): {physical_cpus}")
    print(f"  Worker Processes:    {workers}")
    print("--------------------------------------------------")

    results = []

    # ----------------------------------------------------------
    # Multiprocessing pool
    # ----------------------------------------------------------
    # Using imap_unordered:
    #   - results return as soon as workers finish
    #   - avoids blocking on slow files
    #   - improves throughput
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

    # ----------------------------------------------------------
    # Separate successes and errors
    # ----------------------------------------------------------
    chunk_files = [Path(cp) for status, cp, name in results if status == "OK"]
    errors = [name for status, cp, name in results if status == "ERROR"]

    print("Errors:", len(errors))

    if not chunk_files:
        print("No chunks produced; aborting final parquet.")
        return

    # ----------------------------------------------------------
    # Concatenate parquet chunks
    # ----------------------------------------------------------
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

# ================================================================
# 4. ENTRY POINT
# ================================================================
if __name__ == "__main__":
    run_pipeline()
