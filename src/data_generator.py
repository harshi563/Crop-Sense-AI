"""
data_generator.py
Generates synthetic multi-modal training data for the fusion model.
In production replace with real Sentinel-2, NASA POWER, and FAO data.
"""
import numpy as np
import pandas as pd
from typing import Dict

np.random.seed(42)

CROPS = ["Wheat", "Rice", "Maize", "Soybean", "Cotton"]
YIELD_RANGE = {"Wheat":(1.5,5.5),"Rice":(2.0,7.0),"Maize":(2.0,8.0),
               "Soybean":(0.8,3.5),"Cotton":(0.5,2.5)}
IMG_SIZE, IMG_BANDS = 32, 6
WX_WEEKS, WX_FEATS = 12, 5


def generate_satellite_patches(n: int, health: np.ndarray) -> np.ndarray:
    out = np.zeros((n, IMG_SIZE, IMG_SIZE, IMG_BANDS), dtype=np.float32)
    for i in range(n):
        h = float(health[i])
        red  = np.clip(np.random.uniform(0.02, 0.12 - h*0.07, (IMG_SIZE,IMG_SIZE)), 0,1)
        nir  = np.clip(np.random.uniform(0.10 + h*0.30, 0.45 + h*0.35, (IMG_SIZE,IMG_SIZE)), 0,1)
        blue = np.clip(np.random.uniform(0.02, 0.06, (IMG_SIZE,IMG_SIZE)), 0,1)
        green= np.clip(np.random.uniform(0.03, 0.10, (IMG_SIZE,IMG_SIZE)), 0,1)
        s1   = np.clip(np.random.uniform(0.05, 0.20, (IMG_SIZE,IMG_SIZE)), 0,1)
        s2   = np.clip(np.random.uniform(0.02, 0.12, (IMG_SIZE,IMG_SIZE)), 0,1)
        out[i] = np.stack([blue,green,red,nir,s1,s2], axis=-1)
    return out


def generate_weather_sequences(n: int, stress: np.ndarray) -> np.ndarray:
    out = np.zeros((n, WX_WEEKS, WX_FEATS), dtype=np.float32)
    for i in range(n):
        s = float(stress[i])
        base = np.random.uniform(18, 30)
        for w in range(WX_WEEKS):
            tm  = base + np.random.normal(0, 2+s*3)
            tx  = tm + np.random.uniform(4,9)
            rn  = max(0, np.random.exponential(20)*(1-s*0.65))
            hu  = np.clip(50+rn*0.4+np.random.normal(0,7), 20, 98)
            sl  = np.clip(16+np.random.normal(0,2)-s*2, 5, 25)
            out[i,w] = [tm, tx, rn, hu, sl]
    return out


def generate_soil_features(n: int, fertility: np.ndarray) -> pd.DataFrame:
    f = fertility
    return pd.DataFrame({
        "soil_ph":             np.clip(5.5+f*1.5+np.random.normal(0,0.3,n), 4.5, 8.5),
        "soil_moisture_pct":   np.clip(15+f*35+np.random.normal(0,5,n),    5,  65),
        "organic_carbon_pct":  np.clip(0.3+f*2.5+np.random.normal(0,0.2,n),0.1,3.5),
        "nitrogen_kg_ha":      np.clip(20+f*200+np.random.normal(0,15,n),  5,  280),
        "phosphorus_kg_ha":    np.clip(5+f*60+np.random.normal(0,5,n),     1,  80),
        "potassium_kg_ha":     np.clip(60+f*200+np.random.normal(0,20,n),  20, 320),
        "clay_pct":            np.clip(np.random.uniform(10,45,n),         5,  60),
        "sand_pct":            np.clip(np.random.uniform(20,55,n),         10, 80),
        "silt_pct":            np.clip(np.random.uniform(10,40,n),         5,  50),
        "bulk_density_g_cm3":  np.clip(1.6-f*0.4+np.random.normal(0,0.1,n),1.0,2.0),
        "cec_meq_100g":        np.clip(5+f*35+np.random.normal(0,3,n),     2,  50),
    }).astype(np.float32)


def generate_dataset(n: int = 1000) -> Dict:
    health    = np.random.beta(2, 2, n)
    fertility = np.random.beta(2, 1.5, n)
    stress    = np.random.beta(1.5, 3, n)

    crops = np.random.choice(CROPS, n)
    yields = np.zeros(n, dtype=np.float32)
    for i, c in enumerate(crops):
        lo, hi = YIELD_RANGE[c]
        y = lo + (hi-lo)*(0.4*health[i]+0.35*fertility[i]-0.25*stress[i])
        yields[i] = float(np.clip(y + np.random.normal(0, 0.25), lo*0.6, hi*1.1))

    return {
        "sat":      generate_satellite_patches(n, health),
        "wx":       generate_weather_sequences(n, stress),
        "soil":     generate_soil_features(n, fertility),
        "yields":   yields,
        "n":        n,
    }


def compute_vi(sat: np.ndarray) -> np.ndarray:
    """Compute NDVI and EVI from satellite patches."""
    red = sat[:,:,:,2]; nir = sat[:,:,:,3]; blue = sat[:,:,:,0]
    ndvi = (nir-red)/(nir+red+1e-6)
    evi  = 2.5*(nir-red)/(nir+6*red-7.5*blue+1+1e-6)
    return np.stack([ndvi.mean((1,2)), ndvi.std((1,2)),
                     evi.mean((1,2)),  nir.mean((1,2))], axis=1).astype(np.float32)
