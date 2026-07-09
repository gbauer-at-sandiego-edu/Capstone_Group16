# AAI-590 Capstone Project — Short-Horizon GPS Trajectory Forecasting

This repository contains the source code, notebooks, and documentation for the AAI‑590 Capstone Project at the University of San Diego. The project explores short‑horizon trajectory forecasting using only GPS timestamps and coordinates, with the goal of predicting an agent’s future position 1–5 seconds ahead.

## Project Overview

Modern mobility systems often rely on GPS as the only consistently available signal, especially in transportation analytics, mobility‑as‑a‑service platforms, and safety‑critical monitoring systems. This project investigates whether accurate short‑term motion predictions can be made using only sparse GPS inputs and derived kinematic features.

The project uses the Microsoft GeoLife GPS Trajectory Dataset, which provides high‑resolution movement data suitable for exploratory analysis, feature engineering, and sequence‑based forecasting models.

## Repository Structure

```
AAI590-Capstone/
│
├── notebooks/
│   ├── Capstone_Enhanced_Pipeline.ipynb
│   └── (future) Week4_Modeling.ipynb
│
├── data/
│   ├── raw/
│   │   └── (empty)  # Raw GeoLife data not included
│   ├── processed/
│   │   └── (optional) sample trajectory CSVs
│   └── README.md    # Dataset documentation and citations
│
├── src/
│   ├── processing.py
│   ├── geolife_pipeline_enhanced.py
│   └── load_plt.py
│
├── reports/
│   ├── AAI590 Capstone Project Template.docx
│   └── (future) Final_Report/
│
├── README.md
└── .gitignore
```

## Data Location

Download the GeoLife dataset from Microsoft Research and place all `.plt` files under:

```
data/raw/
```

This directory is intentionally empty in the repository.

## Required Code Updates for Repo Execution

The preprocessing modules in `src/` contain several variables and imports that must be updated to run correctly inside this repository.

### 1. Update DATA_ROOT (critical)

In `src/geolife_pipeline_enhanced.py`, replace the hard‑coded Windows path:

```python
DATA_ROOT = Path(
    r"C:\Users\gb630\OneDrive\USD AAI\USD AAI\AAI-590 CAPSTONE\FINAL PROJECT\DATA"
)
```

with the repo‑correct path:

```python
DATA_ROOT = Path("data/raw")
```

This ensures the pipeline discovers `.plt` files inside the repository.

### 2. Update Output Directories (recommended)

In `src/geolife_pipeline_enhanced.py`, change:

```python
CHUNK_DIR = DATA_ROOT / "chunks"
ERROR_DIR = DATA_ROOT / "errors"
FINAL_PARQUET = DATA_ROOT / "geolife.parquet"
```

to:

```python
CHUNK_DIR = Path("data/processed/chunks")
ERROR_DIR = Path("data/processed/errors")
FINAL_PARQUET = Path("data/processed/geolife.parquet")
```

This keeps raw and processed data properly separated.

### 3. Fix Import in processing.py (critical)

In `src/processing.py`, replace the placeholder import:

```python
from your_notebook_imports import load_plt_file
```

with the correct repo‑local import:

```python
from load_plt import load_plt_file
```

This ensures the module can load PLT files correctly.

### 4. Projection Zone (optional)

Both `processing.py` and `geolife_pipeline_enhanced.py` use:

```python
proj = Proj(proj="utm", zone=48, ellps="WGS84")
```

Zone 48N is correct for Beijing (most GeoLife data).  
No change required unless multi‑zone support is desired.

## Preprocessing Modules

### load_plt.py
Loads and cleans a single `.plt` file, converts timestamps, removes duplicates, enforces monotonic time, and returns:

```
timestamp, lat, lon, alt
```

### processing.py
Resamples trajectories to 1‑second intervals, projects coordinates into UTM, computes kinematic features, enforces numeric schema, and returns a fully processed DataFrame.

### geolife_pipeline_enhanced.py
A multiprocessing pipeline that:

- Discovers all `.plt` files under `data/raw/`
- Processes each file using `load_plt.py` and `processing.py`
- Writes parquet chunks
- Logs errors and empty outputs
- Concatenates all chunks into a final unified parquet dataset
- Produces a summary report with row counts, timing, and error statistics

This is the main entry point for large‑scale preprocessing.

## How to Use This Repository

### 1. Download the GeoLife Dataset

Place all `.plt` files under:

```
data/raw/
```

### 2. Run the Enhanced Pipeline Notebook

Open:

```
notebooks/Capstone_Enhanced_Pipeline.ipynb
```

This notebook demonstrates:

- Loading PLT files
- Applying preprocessing functions
- Inspecting kinematic features
- Visualizing trajectories and motion patterns

### 3. Run the Full Multiprocessing Pipeline

From the repository root:

```
python src/geolife_pipeline_enhanced.py
```

Ensure `DATA_ROOT` inside the module points to:

```
data/raw
```

### 4. Reports

The `reports/` directory contains the working capstone document and will later include the final report and supporting materials.

## References (APA 7)

Zheng, Y., Li, Q., Chen, Y., Xie, X., & Ma, W. Y. (2008). Understanding mobility based on GPS data. Proceedings of the 10th International Conference on Ubiquitous Computing, 312–321. [https://doi.org/10.1145/1409635.1409677](https://doi.org/10.1145/1409635.1409677)

Zheng, Y., Xie, X., & Ma, W. Y. (2010). GeoLife: A collaborative social networking service among user, location and trajectory. IEEE Data Engineering Bulletin, 33(2), 32–39.

Microsoft Research Asia. (2012). GeoLife GPS Trajectory Dataset (Version 1.3). [https://www.microsoft.com/en-us/research/project/geolife-building-social-networks-using-human-location-history/](https://www.microsoft.com/en-us/research/project/geolife-building-social-networks-using-human-location-history/)
