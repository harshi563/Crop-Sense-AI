"""
preprocessing.py
Feature engineering, scaling, and PyTorch DataLoaders.
"""
import numpy as np
import pandas as pd
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib


# ── Weather enrichment ──────────────────────────────────────────

def enrich_weather(wx: np.ndarray) -> np.ndarray:
    """
    (n, T, 5) → (n, T, 9)
    Adds: temp_anomaly | cumulative_rain | drought_index | heat_stress
    """
    n, T, F = wx.shape
    out = np.zeros((n, T, F+4), dtype=np.float32)
    out[:,:,:F] = wx
    for i in range(n):
        tm, rn, hu = wx[i,:,0], wx[i,:,2], wx[i,:,3]
        out[i,:,F]   = tm - tm.mean()
        out[i,:,F+1] = np.cumsum(rn) / (rn.sum() + 1e-6)
        out[i,:,F+2] = 1 - (rn / (rn.max()+1e-6)) * (hu / 100)
        out[i,:,F+3] = np.clip(tm - 35, 0, None) / 10
    return out


# ── Soil enrichment ─────────────────────────────────────────────

def enrich_soil(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().astype(np.float32)
    df["np_ratio"]       = df["nitrogen_kg_ha"] / (df["phosphorus_kg_ha"] + 1e-6)
    df["ph_dev"]         = (df["soil_ph"] - 6.5).abs()
    df["som"]            = df["organic_carbon_pct"] * 1.724
    df["texture_score"]  = 1.0 - (
        (df["clay_pct"]-33).abs() +
        (df["sand_pct"]-33).abs() +
        (df["silt_pct"]-33).abs()
    ) / 100
    df["fertility_idx"]  = (
        np.clip((df["nitrogen_kg_ha"]-20)/260,0,1)*0.30 +
        np.clip((df["organic_carbon_pct"]-0.3)/3.2,0,1)*0.25 +
        np.clip(df["soil_moisture_pct"]/65,0,1)*0.20 +
        np.clip(1-df["ph_dev"]/4,0,1)*0.15 +
        np.clip(df["texture_score"],0,1)*0.10
    )
    return df.astype(np.float32)


# ── Scaler ──────────────────────────────────────────────────────

class Scaler:
    """Holds separate scalers for each modality."""
    def __init__(self):
        self.soil = StandardScaler()
        self.wx   = StandardScaler()
        self.sat  = MinMaxScaler()
        self.y    = StandardScaler()
        self._fit = False

    def fit_transform(self, soil, wx, sat, y):
        n = len(y)
        T, Fw = wx.shape[1], wx.shape[2]
        H, W, C = sat.shape[1], sat.shape[2], sat.shape[3]

        soil_s = self.soil.fit_transform(soil).astype(np.float32)
        wx_s   = self.wx.fit_transform(wx.reshape(-1,Fw)).reshape(n,T,Fw).astype(np.float32)
        sat_s  = self.sat.fit_transform(sat.reshape(-1,1)).reshape(n,H,W,C).astype(np.float32)
        y_s    = self.y.fit_transform(y.reshape(-1,1)).ravel().astype(np.float32)
        self._fit = True
        return soil_s, wx_s, sat_s, y_s   # ORDER: soil, wx, sat, y

    def transform_single(self, soil, wx, sat):
        """Scale a single sample (each input is already batched with n=1)."""
        assert self._fit
        n = soil.shape[0]
        T, Fw = wx.shape[1], wx.shape[2]
        H, W, C = sat.shape[1], sat.shape[2], sat.shape[3]
        soil_s = self.soil.transform(soil).astype(np.float32)
        wx_s   = self.wx.transform(wx.reshape(-1,Fw)).reshape(n,T,Fw).astype(np.float32)
        sat_s  = self.sat.transform(sat.reshape(-1,1)).reshape(n,H,W,C).astype(np.float32)
        return soil_s, wx_s, sat_s

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.y.inverse_transform(y_scaled.reshape(-1,1)).ravel()

    def save(self, path): joblib.dump(self, path)

    @staticmethod
    def load(path): return joblib.load(path)


# ── PyTorch Dataset ─────────────────────────────────────────────

class AgriDataset(Dataset):
    def __init__(self, sat, wx, soil, y):
        # sat: (n,H,W,C) → permute to (n,C,H,W) for CNN
        self.sat  = torch.from_numpy(sat).permute(0,3,1,2)
        self.wx   = torch.from_numpy(wx)
        self.soil = torch.from_numpy(soil)
        self.y    = torch.from_numpy(y).unsqueeze(1)

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.sat[i], self.wx[i], self.soil[i], self.y[i]


def make_loaders(soil_s, wx_s, sat_s, y_s, batch=32):
    idx = np.arange(len(y_s))
    tr, te = train_test_split(idx, test_size=0.15, random_state=42)
    tr, va = train_test_split(tr,  test_size=0.15, random_state=42)

    def _dl(ids, shuffle):
        ds = AgriDataset(sat_s[ids], wx_s[ids], soil_s[ids], y_s[ids])
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0)

    return _dl(tr, True), _dl(va, False), _dl(te, False)
