<div align="center">
  <h1>🌾 CropSense AI</h1>
  <p><b>Climate-Resilient Crop Yield Predictor & Smart Farm Advisor</b></p>
  <p>
    <a href="#-quick-start">🚀 Quick Start</a> •
    <a href="#-how-it-works">⚙️ How it Works</a> •
    <a href="#-user-inputs">📝 Inputs</a> •
    <a href="#-project-structure">📂 Structure</a> •
    <a href="#-model-architecture">🧠 Model</a>
  </p>
</div>

---

## ✨ Overview

Welcome to **CropSense AI**! This application is designed to provide highly accurate crop yield predictions and personalized farm advisory by fusing satellite imagery concepts, historical weather data, and local soil conditions. 

🌍 **Real-Time Data**: Automatically fetches live weather data from Open-Meteo (100% free, no API key required).  
🧠 **Advanced AI**: Uses a multimodal fusion of CNNs, BiLSTMs, and MLPs.  
💡 **Smart Advisory**: Provides a 9-category actionable agronomic report based on your farm's specifics.

---

## 🚀 Quick Start

Get the application running in just a few steps!

<details>
<summary><b>🛠️ Step-by-Step Installation</b> (Click to expand)</summary>

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd cropsense_final
   ```

2. **Install the dependencies**:
   Make sure you have Python installed, then run:
   ```bash
   # Install PyTorch (CPU version recommended for standard testing)
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   
   # Install remaining requirements
   pip install -r requirements.txt
   ```

3. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

4. **View the App**: Open your browser and navigate to 🌐 **[http://localhost:8501](http://localhost:8501)**
</details>

---

## ⚙️ How it Works

Curious about the magic behind the scenes? Here is our fully automated data pipeline:

<details>
<summary><b>🔍 View Pipeline Diagram</b></summary>

```text
🧑‍🌾 User types city name
      │
      ▼  open-meteo.com (geocoding)
  📍 Latitude / Longitude
      │
      ├──▶  archive-api.open-meteo.com 📅
      │     Past 84 days → 12 weekly rows → BiLSTM input
      │
      └──▶  api.open-meteo.com 🌤️
            7-day forecast + current conditions
                  │
                  ▼
         🌡️ Climate stress · drought · heat indices
         🍂 Season auto-detected (Kharif/Rabi/Zaid)
                  │
            + 🧪 Soil values (pH, N, P, K, OC, moisture)
                  │
                  ▼
    ┌─────────────────────────────────────────┐
    │   🧠 Multimodal Fusion Model            │
    │   CNN (satellite) ──────────┐           │
    │   BiLSTM (12wk weather) ───▶ Fusion ──▶ 🌾 Yield (t/ha)
    │   MLP (soil features) ──────┘           │
    └─────────────────────────────────────────┘
                  │
                  ▼
        📋 Personalised farm advisory
        (9 agronomic categories)
```
</details>

---

## 📝 User Inputs

We keep it incredibly simple. You only need to provide **3 things**—the AI handles the rest!

| Input | Fields | Description |
|:---:|---|---|
| 📍 **Location** | City name | Any city worldwide (free text). Geocoded automatically! |
| 🌾 **Crop + Area** | Crop type & Hectares | What are you growing and how much land do you have? |
| 🧪 **Soil Test** | pH, N, P, K, OC, Moisture | Values from your latest soil health card or lab report. |

✨ *Everything else, including complex climate features and stress indices, is fetched automatically.*

---

## 📂 Project Structure

Navigate the codebase with ease:

<details>
<summary><b>🗺️ View Directory Tree</b></summary>

```text
cropsense_final/
├── 🚀 app.py                  ← Main Streamlit dashboard
├── 📦 requirements.txt        ← Python dependencies
├── 📖 README.md               ← You are here!
└── 📁 src/
    ├── 🌤️ weather.py          ← Open-Meteo real-time data fetcher
    ├── 💡 advisor.py          ← Rule-based recommendation engine
    ├── 🎲 data_generator.py   ← Synthetic training data generation
    ├── 🧹 preprocessing.py    ← Feature engineering & data scalers
    ├── 🏋️ trainer.py          ← Model training loop & metrics
    └── 📁 models/
        └── 🧠 fusion_model.py ← CNN + BiLSTM + MLP architecture
```
</details>

---

## 🌤️ Weather APIs Used

We rely on **Open-Meteo** for highly reliable, worldwide weather data. Best of all: **Zero registration. Zero API keys. 100% Free.**

| API Service | URL | What it fetches |
|---|---|---|
| 📍 **Geocoding** | `geocoding-api.open-meteo.com` | City Name → GPS Coordinates |
| 📅 **Archive** | `archive-api.open-meteo.com` | Past 84 days of daily weather |
| 🔮 **Forecast** | `api.open-meteo.com` | 7-day outlook + current conditions |

---

## 🧠 Model Architecture

The core of CropSense AI is a robust multimodal neural network:

<details>
<summary><b>🔬 View Architecture Details</b></summary>

```text
🛰️ Satellite CNN (6 bands, 32×32)  ───────→  64-d embedding
🌧️ Weather BiLSTM (12 wk × 9 feat) ───────→ 128-d embedding  
🧪 Soil MLP (16 features)          ───────→  64-d embedding
                                                  │
                                            (concat 256-d)
                                                  │
                                                  ▼
                                     🔗 Fusion MLP (256→128→64→1)
                                                  │
                                                  ▼
                                       📊 Yield Prediction (t/ha)
```

> **Note on Training Data**: The current version is trained on a synthetically generated dataset (1,200 samples) to demonstrate capability. For a production deployment, simply replace `generate_dataset()` with actual data sources (e.g., Sentinel-2, NASA POWER, FAO).
</details>

---
<div align="center">
  <i>Built with ❤️ for modern, climate-resilient agriculture.</i>
</div>
