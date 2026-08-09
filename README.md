---

# 📘 **AAI‑590 Capstone Project — Short‑Horizon GPS Trajectory Forecasting**

This repository contains the full source code, preprocessing pipeline, trained models, artifacts, and visualization tools for the **AAI‑590 Capstone Project** at the **University of San Diego**.

The project investigates whether accurate **short‑horizon (1–5 second)** motion predictions can be made using only **GPS coordinates** and derived **kinematic features**, using the **Microsoft GeoLife GPS Trajectory Dataset**.

---

# 🚀 **Project Overview**

Modern mobility systems often rely on GPS as the only consistently available signal — especially in transportation analytics, mobility‑as‑a‑service platforms, and safety‑critical monitoring systems.

This project explores whether short‑term trajectory forecasting can be achieved using:

- Raw GPS timestamps  
- Latitude/longitude  
- Derived motion features (speed, heading, acceleration, turn rate)  
- Sequence‑based forecasting models  

We evaluate:

- A **constant‑velocity baseline**
- A **GRU sequence model**
- An **LSTM sequence model**
- A **tuned Temporal Convolutional Network (TCN)** with:
  - Dilated convolutions  
  - Residual connections  
  - Sinusoidal positional encodings  
  - Multi‑head self‑attention  

---

# 📂 **Repository Structure**

```
Capstone_Group16/
│
├── notebooks/
│   └── Capstone_Enhanced_Pipeline.ipynb  # GeoLife Dataset EDA
│   └── Visualization.ipynb        # Full visualization suite (loss curves, KDE, scatter, dashboards)
│
├── data/
│   ├── raw/                       # Raw GeoLife .plt files (not included in repo)
│   └── processed/                 # Processed parquet + optional CSVs
│
├── src/
│   ├── geolife_pipeline.py        # Preprocessing pipeline (PLT → parquet)
│   ├── load_plt.py                # PLT loader + timestamp cleanup
│   ├── models_geolife.py          # Baseline GRU/LSTM/TCN pipeline
│   ├── models_geolife_tuned.py    # Tuned TCN pipeline (dilations + attention)
│   ├── visualization.py           # Optional standalone plotting utilities
│   │
│   ├── models/
│   │   ├── gru_model.pt           # Trained GRU weights
│   │   ├── lstm_model.pt          # Trained LSTM weights
│   │   ├── tcn_model.pt           # Trained tuned TCN weights
│   │   ├── GRU_metadata.json
│   │   ├── LSTM_metadata.json
│   │   ├── TCN_metadata.json
│   │   └── logs/                  # Training logs (kept for reproducibility)
│   │
│   └── artifacts/
│       ├── predictions/           # Saved inputs/preds/targets for all models
│       ├── loss/                  # Loss curves for each model
│       ├── metrics/               # Final ADE/FDE/MDE metrics
│       └── windows/               # Window counts + diagnostics
│
├── reports/
│   ├── AAI590 Capstone Project Template.docx
│   └── Final_Report/              # Final capstone report (in progress)
│
└── README.md
```

---

# 🛠️ **Preprocessing Pipeline**

The preprocessing pipeline converts raw GeoLife `.plt` files into a unified parquet dataset.

### Steps:
1. Load PLT file  
2. Clean timestamps  
3. Remove duplicates  
4. Enforce monotonic time  
5. Project lat/lon → UTM  
6. Compute kinematic features  
7. Save parquet chunks  
8. Merge into final `geolife.parquet`

### Output:
A clean dataset with:

- `x, y` projected coordinates  
- `speed`  
- `heading`  
- `accel`  
- `turn_rate`  

Stored at:

```
src/data/processed/geolife.parquet
```

---

# 🤖 **Modeling Pipeline**

All models predict **5 future displacement steps** based on a **20‑step input window**.

### Models Included

| Model | Description |
|-------|-------------|
| **Baseline** | Constant‑velocity extrapolation |
| **GRU** | Lightweight recurrent model |
| **LSTM** | Long‑range recurrent model |
| **TCN** | Dilated convolutional network |
| **Tuned TCN** | TCN + residuals + positional encodings + attention |

### Training Features

- Global normalization  
- Per‑window normalization  
- Glitch filtering  
- Window caps to prevent memory blow‑ups  
- Heartbeat logging (CPU/RAM)  
- Loss curve saving  
- Prediction artifact saving  

---

# 📈 **Visualization Suite**

The notebook `Visualization.ipynb` provides:

### Training Diagnostics
- Loss curves (early convergence + full)
- Window distribution histograms

### Error Analysis
- KDE error distributions
- Scatter plots of final displacement
- Residual error vectors
- Multi‑panel trajectory dashboards

### Model Comparison
- Side‑by‑side GRU vs LSTM vs TCN
- ADE / FDE / MDE dashboard

All visualizations use a **USD Torero theme** for consistent styling.

---

# 📊 **Metrics**

Each model is evaluated using:

- **ADE** — Average Displacement Error  
- **FDE** — Final Displacement Error  
- **MDE** — Maximum Displacement Error  

Metrics are saved under:

```
src/artifacts/metrics/
```

---

# 📦 **Model Artifacts**

Trained model weights are included in the repo for reproducibility:

```
src/models/gru_model.pt
src/models/lstm_model.pt
src/models/tcn_model.pt
```

Metadata files describe:

- Feature ordering  
- Input window size  
- Future step count  

---

# ▶️ **How to Run**

### 1. Preprocess GeoLife Data
Place `.plt` files under:

```
data/raw/
```

Run:

```bash
python src/geolife_pipeline.py
```

### 2. Train Models

```bash
python src/models_geolife.py
python src/models_geolife_tuned.py
```

### 3. Visualize Results

Open:

```
notebooks/Visualization.ipynb
```

---

# 📚 **References**

  [github.com](https://github.com/gbauer-at-sandiego-edu/Capstone_Group16)

---

# 🎓 **About**

This project was completed as part of the **AAI‑590 Capstone** in the  
**Master of Science in Applied Artificial Intelligence** program  
at the **University of San Diego**.

---