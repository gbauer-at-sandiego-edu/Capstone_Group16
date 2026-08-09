"""
load_plt.py
===========

This helper module loads a single GeoLife `.plt` file and converts it
into a clean, timestamp‑indexed pandas DataFrame.

GeoLife PLT files follow a fixed structure:

    • First 6 lines = metadata (ignored)
    • Remaining lines contain:
        lat, lon, unused, altitude, days_since_1899, date, time

This loader is intentionally minimal and deterministic because it is
called inside multiprocessing workers in `geolife_pipeline.py`.

Key design goals:
    • Avoid schema drift
    • Remove malformed timestamps
    • Remove duplicate or backward‑moving timestamps
    • Return only the columns needed for downstream kinematic processing
"""

import pandas as pd


def load_plt_file(path):
    """
    Load a single GeoLife .plt file into a clean DataFrame.

    WHY THIS FUNCTION EXISTS:
    -------------------------
    GeoLife PLT files are not standard CSVs. They contain:
        • 6 header lines of metadata
        • No column headers
        • A mix of numeric and string fields
        • A timestamp split across two columns (date + time)
        • A "days" column representing days since 1899 (unused)

    This loader:
        1. Skips the metadata header
        2. Assigns explicit column names
        3. Constructs a proper pandas datetime timestamp
        4. Removes malformed timestamps
        5. Sorts chronologically
        6. Removes duplicate timestamps
        7. Removes non‑forward time movement (dt <= 0)
        8. Returns only the fields needed for downstream processing

    Parameters
    ----------
    path : str or Path
        Path to the .plt file.

    Returns
    -------
    DataFrame with columns:
        ["timestamp", "lat", "lon", "alt"]

    These columns form the minimal, clean input required by the
    geolife_pipeline → UTM projection → kinematic feature generator.
    """

    # ------------------------------------------------------------------
    # Read PLT file
    # ------------------------------------------------------------------
    # skiprows=6:
    #     The first 6 lines contain metadata such as:
    #         "Geolife trajectory", "Altitude is in feet", etc.
    #     These lines are not part of the tabular data.
    #
    # header=None + names=[...]:
    #     PLT files do not contain column headers, so we assign them.
    #
    # "unused":
    #     GeoLife includes a placeholder column that is always zero.
    #     We keep it only long enough to parse the file; it is dropped later.
    df = pd.read_csv(
        path,
        skiprows=6,
        header=None,
        names=["lat", "lon", "unused", "alt", "days", "date", "time"]
    )

    # ------------------------------------------------------------------
    # Construct timestamp column
    # ------------------------------------------------------------------
    # GeoLife stores date and time separately. We combine them into a
    # single pandas datetime. Using errors="coerce" ensures that any
    # malformed timestamps become NaT and are removed.
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    # Remove rows with invalid timestamps
    df = df.dropna(subset=["timestamp"])

    # ------------------------------------------------------------------
    # Sort chronologically and remove duplicates
    # ------------------------------------------------------------------
    # Sorting ensures deterministic ordering.
    # Duplicate timestamps occur in some GeoLife logs due to GPS jitter.
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"])

    # ------------------------------------------------------------------
    # Remove non‑forward time movement
    # ------------------------------------------------------------------
    # GeoLife occasionally contains timestamp glitches where time moves
    # backward or repeats. These break kinematic calculations.
    #
    # We compute dt between consecutive timestamps and keep only rows
    # where dt > 0.
    dt = df["timestamp"].diff().dt.total_seconds()
    df = df[dt > 0]

    # ------------------------------------------------------------------
    # Return only the columns needed downstream
    # ------------------------------------------------------------------
    # The pipeline will compute:
    #     • UTM projection
    #     • dx, dy, dt
    #     • speed, heading, accel, turn_rate
    #
    # So we return only the minimal raw fields required.
    return df[["timestamp", "lat", "lon", "alt"]]
