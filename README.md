
---

# 🚀 Short‑Horizon GPS Trajectory Forecasting  
### Deep Sequence Models for Real‑Time Mobility Analytics  
**GRU • LSTM • Temporal Convolutional Network (TCN)**  
*AAI‑590 Capstone — University of San Diego*

---

## 📘 Abstract

Short‑horizon trajectory forecasting is essential for mobility systems that operate with limited or degraded sensor availability. This project evaluates whether accurate 1–5 second motion predictions can be produced using only GPS coordinates and derived kinematic features from the Microsoft GeoLife dataset. A unified forecasting pipeline was developed using engineered motion variables—including normalized positional deltas, speed, heading, acceleration, and turn rate—and three deep sequence models: a GRU, an LSTM, and a Temporal Convolutional Network (TCN) with dilated convolutions, residual connections, positional encodings, and multi‑head attention.

The system incorporates global normalization, per‑window normalization, adaptive window caps, and a streaming window generator to ensure stable training on long and heterogeneous trajectories. Models were evaluated using displacement‑based metrics (ADE, FDE, MDE) computed in normalized coordinate space.

Results show that all learned models outperform a constant‑velocity baseline, with the tuned LSTM achieving the strongest overall performance across validation and test sets. After addressing memory and preprocessing constraints, the tuned TCN achieved competitive ADE and FDE values but retained higher maximum displacement errors. These findings demonstrate that accurate short‑term motion forecasting is feasible using only GPS‑derived features and lightweight temporal models, providing a strong foundation for real‑time mobility analytics and future work in sampling‑rate robustness, transformer architectures, and deployable inference modules.

---

## 📂 Repository Structure

```
Capstone_Group16/
│
├── notebooks/
│   ├── Capstone_Enhanced_Pipeline.ipynb
│   └── Visualization.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── geolife_pipeline.py
│   ├── load_plt.py
│   ├── models_geolife.py
│   ├── models_geolife_tuned.py
│   ├── visualization.py
│   │
│   ├── models/
│   │   ├── gru_model.pt
│   │   ├── lstm_model.pt
│   │   ├── tcn_model.pt
│   │   ├── GRU_metadata.json
│   │   ├── LSTM_metadata.json
│   │   ├── TCN_metadata.json
│   │   └── logs/
│   │
│   └── artifacts/
│       ├── predictions/
│       ├── loss/
│       ├── metrics/
│       └── windows/
│
├── reports/
│   ├── AAI590 Capstone Project Template.docx
│   └── Final_Report/
│
└── README.md
```

---

## 🛠️ Preprocessing Pipeline

The GeoLife dataset contains over 17,000 trajectories sampled at 1–5 second intervals.  
Preprocessing includes:

1. Timestamp cleanup  
2. Duplicate removal  
3. Monotonic time enforcement  
4. Global normalization  
5. Per‑window normalization  
6. Kinematic feature derivation  
7. Sliding‑window extraction  
8. Adaptive window caps for long trajectories  
9. Parquet output generation  

**Important:**  
All modeling is performed in **normalized latitude/longitude space**.  
Metrics reflect **normalized displacement**, not physical meters.

---

## 🤖 Modeling Pipeline

### **Input Features (6 total)**  
- Δlatitude (normalized)  
- Δlongitude (normalized)  
- Speed  
- Heading  
- Acceleration  
- Turn rate  

### **Forecasting Task**  
- **Input:** 20‑step window  
- **Output:** 5 future displacement steps (10 values)

### **Models Evaluated**
| Model | Description |
|-------|-------------|
| Constant‑Velocity Baseline | Extrapolates last observed displacement |
| GRU | 64‑unit recurrent model |
| LSTM | 64‑unit recurrent model |
| TCN | 5 dilated conv layers + residuals + positional encodings + multi‑head attention |

### **Training Details**
- Optimizer: **Adam (lr = 1e‑3)**  
- Loss: **MSE**  
- Epochs: **20**  
- Window generator: **streaming**, memory‑safe  
- Normalization: **global + per‑window**  
- Dataset split: **60/20/20** by trajectory  

---

# 🧠 Model Cards

## 📘 GRU Model Card

**Model Type:** Gated Recurrent Unit (GRU)  
**Hidden Size:** 64  
**Purpose:** Lightweight baseline recurrent model for short‑horizon displacement forecasting.

### Inputs
- 20‑step window  
- Features: Δlat, Δlon, speed, heading, acceleration, turn rate  

### Outputs
- 5 future displacement steps (10 values)

### Performance
- **Val ADE:** 0.012  
- **Test ADE:** 0.014  
- Outperforms constant‑velocity baseline  
- Stable convergence

### Limitations
- Slightly worse temporal stability than LSTM  
- Sensitive to irregular sampling intervals  

---

## 📘 LSTM Model Card

**Model Type:** Long Short‑Term Memory (LSTM)  
**Hidden Size:** 64  
**Purpose:** Capture long‑range temporal dependencies for stable short‑horizon forecasting.

### Performance
- **Val ADE:** 0.006  
- **Test ADE:** 0.009  
- **Best overall performance across ADE, FDE, MDE**  
- Tightest error distribution  

### Limitations
- Slightly higher inference latency than GRU  
- Sensitive to heading discontinuities  

---

## 📘 TCN Model Card

**Model Type:** Dilated Temporal Convolutional Network + Multi‑Head Attention  
**Layers:** 5 dilated conv blocks (dilations: 1, 2, 4, 8, 16)  
**Attention:** 4‑head self‑attention  

### Performance (Pre‑Tuning)
- **Val ADE:** 0.014  
- **Test ADE:** 0.048  
- **Val MDE:** 0.765  

### Performance (Tuned)
- **Val ADE:** 0.0024  
- **Test ADE:** 0.0025  
- Competitive ADE/FDE  
- **Higher MDE** than LSTM  

### Limitations
- Sensitive to long trajectories  
- Requires careful preprocessing  
- Higher worst‑case displacement errors  

---

## 📊 Results

### **Pre‑Tuning Results**
| Model | Val ADE | Test ADE | Val MDE | Test MDE |
|-------|---------|----------|---------|----------|
| Baseline | — | — | **8.448** | — |
| GRU | 0.012 | 0.014 | — | — |
| LSTM | **0.006** | **0.009** | — | — |
| TCN (pre‑tuning) | 0.014 | 0.048 | 0.765 | — |

### **Tuned TCN Results**
| Model | Val ADE | Test ADE | MDE Behavior |
|-------|---------|----------|--------------|
| TCN (tuned) | **0.0024** | **0.0025** | Higher than LSTM |

### Key Findings
- **LSTM achieved the strongest overall performance**  
- **GRU performed competitively**  
- **TCN required preprocessing fixes**  
- After tuning, **TCN achieved excellent ADE/FDE**, but **higher MDE**  
- All models showed **smooth, monotonic convergence**  

---

## 📈 Visualization Suite

The `Visualization.ipynb` notebook provides:

- Training loss curves  
- KDE displacement‑error distributions  
- Predicted vs. true displacement scatter plots  
- ADE/FDE/MDE dashboards  
- Window diagnostics  
- USD‑themed visualizations  

---

## 📦 Model Artifacts

Trained weights and metadata are included:

```
src/models/gru_model.pt
src/models/lstm_model.pt
src/models/tcn_model.pt
```

Metadata files specify:

- Feature ordering  
- Window size  
- Forecast horizon  

---

## ▶️ How to Run

### 1. Preprocess GeoLife Data
Place `.plt` files in:

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

## 🔮 Future Work

- Heading unwrapping  
- Timestamp interpolation  
- Sampling‑rate degradation experiments  
- Transformer‑based architectures  
- Mode‑specific modeling  
- Real‑time inference module with latency profiling  

---

## 🎓 About

This project was completed as part of **AAI‑590 Capstone** in the  
**Master of Science in Applied Artificial Intelligence** program  
at the **University of San Diego**.

---