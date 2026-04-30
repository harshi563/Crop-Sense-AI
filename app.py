"""
CropSense AI  ·  Production v3
================================
Accessibility-first redesign:
  • High-contrast inputs (WCAG AA compliant)
  • Large labels with units and optimal-range hints
  • Live value readout on every slider
  • Keyboard navigable, focus-visible throughout
  • Clear 3-step workflow with visual progress
"""

from __future__ import annotations
import json, logging, sys, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title = "CropSense AI — Smart Farm Advisor",
    page_icon  = "🌾",
    layout     = "wide",
    initial_sidebar_state = "expanded",
    menu_items = {"About": "CropSense AI — CNN+BiLSTM+MLP · Weather: open-meteo.com"},
)

# ─────────────────────────────────────────────────────────────
#  STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

/* ── Base ─────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #F2EFE6; }

/* ── Sidebar shell ────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: #0D2B1A !important;
  border-right: 1px solid rgba(255,255,255,0.07);
}
/* Kill the default Streamlit label colour override inside sidebar */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span:not(.badge):not(.step-n):not(.soil-badge) {
  color: #C8E6C0 !important;
}
section[data-testid="stSidebar"] hr {
  border-color: rgba(255,255,255,0.10) !important;
  margin: 10px 0 !important;
}

/* ── Input cards — bright white boxes inside the dark sidebar ─ */
.input-card {
  background: #FFFFFF;
  border-radius: 14px;
  padding: 16px 18px;
  margin-bottom: 14px;
  border: 2px solid #E2EFE2;
  box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}
.input-card.done  { border-color: #52B788; }
.input-card.active{ border-color: #1E6B3A; box-shadow: 0 0 0 3px rgba(30,107,58,0.15), 0 2px 8px rgba(0,0,0,0.12); }

/* Card header row */
.card-hd {
  display: flex; align-items: center; gap: 10px; margin-bottom: 14px;
}
.step-circle {
  width: 28px; height: 28px; border-radius: 50%;
  background: #1E6B3A; color: #fff;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.78rem; font-weight: 700; flex-shrink: 0;
  font-family: 'Inter', sans-serif;
}
.step-circle.done  { background: #52B788; }
.card-title {
  font-size: 0.92rem; font-weight: 700;
  color: #0D2B1A !important; letter-spacing: 0.2px;
}
.card-sub { font-size: 0.72rem; color: #5A7A60 !important; margin-top: 1px; }

/* ── Text input inside card ──────────────────────────────── */
.input-card .stTextInput input {
  background: #F7FBF7 !important;
  border: 2px solid #B8D8B8 !important;
  border-radius: 10px !important;
  color: #0D2B1A !important;
  font-size: 1rem !important;
  font-weight: 500 !important;
  padding: 11px 14px !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
.input-card .stTextInput input:focus {
  border-color: #1E6B3A !important;
  box-shadow: 0 0 0 3px rgba(30,107,58,0.18) !important;
  outline: none !important;
}
.input-card .stTextInput input::placeholder { color: #8FA88F !important; font-weight:400 !important; }

/* ── Selectbox inside card ────────────────────────────────── */
.input-card .stSelectbox > div > div {
  background: #F7FBF7 !important;
  border: 2px solid #B8D8B8 !important;
  border-radius: 10px !important;
  color: #0D2B1A !important;
  font-size: 0.95rem !important;
  font-weight: 500 !important;
}
.input-card .stSelectbox > div > div:focus-within {
  border-color: #1E6B3A !important;
  box-shadow: 0 0 0 3px rgba(30,107,58,0.18) !important;
}

/* ── Number input inside card ─────────────────────────────── */
.input-card .stNumberInput input {
  background: #F7FBF7 !important;
  border: 2px solid #B8D8B8 !important;
  border-radius: 10px !important;
  color: #0D2B1A !important;
  font-size: 0.95rem !important;
  font-weight: 500 !important;
}
.input-card .stNumberInput input:focus {
  border-color: #1E6B3A !important;
  box-shadow: 0 0 0 3px rgba(30,107,58,0.18) !important;
}

/* ── Slider row (custom label + value badge above native slider) */
.slider-row {
  margin-bottom: 16px;
}
.slider-label-row {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 5px;
}
.slider-label {
  font-size: 0.84rem; font-weight: 600;
  color: #0D2B1A !important;
  display: flex; align-items: center; gap: 6px;
}
.slider-label .licon { font-size: 1rem; }
.slider-value-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.9rem; font-weight: 500;
  background: #1E6B3A; color: #FFFFFF;
  padding: 3px 11px; border-radius: 20px;
  min-width: 60px; text-align: center;
  box-shadow: 0 2px 6px rgba(30,107,58,0.3);
}
.slider-hint {
  font-size: 0.68rem; color: #6A8A6A !important;
  margin-top: 2px; display: flex; justify-content: space-between;
}
.slider-hint .optimal { color: #1E6B3A !important; font-weight: 600; }

/* Native slider track + thumb styling */
.input-card .stSlider [data-baseweb="slider"] {
  margin-top: 2px;
}
/* Slider track */
.input-card .stSlider div[data-testid="stSlider"] > div > div > div {
  background: #C8E6C0 !important;
  height: 6px !important;
  border-radius: 3px !important;
}
/* Filled portion */
.input-card .stSlider div[data-testid="stSlider"] > div > div > div > div:first-child {
  background: #1E6B3A !important;
}
/* Thumb */
.input-card .stSlider div[data-testid="stSlider"] div[role="slider"] {
  background: #1E6B3A !important;
  border: 3px solid #FFFFFF !important;
  width: 20px !important; height: 20px !important;
  box-shadow: 0 2px 8px rgba(30,107,58,0.4) !important;
}
.input-card .stSlider div[data-testid="stSlider"] div[role="slider"]:focus {
  box-shadow: 0 0 0 4px rgba(30,107,58,0.35), 0 2px 8px rgba(30,107,58,0.4) !important;
  outline: none !important;
}
/* Hide default Streamlit slider value tooltip */
.input-card .stSlider [data-testid="stThumbValue"] { display: none !important; }

/* ── Buttons ─────────────────────────────────────────────── */
div.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #1A5C33, #2A8A4E) !important;
  color: #FFFFFF !important;
  border: none !important; border-radius: 12px !important;
  font-weight: 700 !important; font-size: 0.92rem !important;
  padding: 13px 20px !important; width: 100% !important;
  box-shadow: 0 4px 16px rgba(30,107,58,0.40) !important;
  letter-spacing: 0.2px !important;
  transition: all 0.2s !important;
}
div.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #14502C, #1E7A43) !important;
  box-shadow: 0 6px 22px rgba(30,107,58,0.50) !important;
  transform: translateY(-2px) !important;
}
div.stButton > button[kind="primary"]:focus-visible {
  outline: 3px solid #52B788 !important;
  outline-offset: 3px !important;
}
div.stButton > button[kind="primary"]:disabled {
  background: #4A6A54 !important; color: rgba(255,255,255,0.45) !important;
  box-shadow: none !important; transform: none !important; cursor: not-allowed !important;
}
div.stDownloadButton > button {
  background: #EAF7F0 !important; color: #1A5C33 !important;
  border: 2px solid #52B788 !important; border-radius: 10px !important;
  font-weight: 600 !important; padding: 10px 18px !important;
}
div.stDownloadButton > button:hover {
  background: #D5F0E3 !important;
}
div.stDownloadButton > button:focus-visible {
  outline: 3px solid #1E6B3A !important; outline-offset: 2px !important;
}

/* ── Sidebar step divider ─────────────────────────────────── */
.step-done-badge {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.72rem; color: #52B788 !important; font-weight: 600;
  background: rgba(82,183,136,0.12);
  border: 1px solid rgba(82,183,136,0.3);
  padding: 2px 10px; border-radius: 20px; margin-left: auto;
}

/* ── Sidebar credit ──────────────────────────────────────── */
.sb-credit {
  font-size: 0.68rem; color: #4A7A54 !important; line-height: 1.85;
  padding: 10px 14px; background: rgba(255,255,255,0.04);
  border-radius: 10px; margin-top: 8px;
  border: 1px solid rgba(255,255,255,0.07);
}
.sb-credit b { color: #7BC88A !important; }

/* ── Crop selector pills ──────────────────────────────────── */
.crop-label { font-size: 0.84rem; font-weight: 600; color: #0D2B1A !important; margin-bottom: 8px; }

/* ── Hero ────────────────────────────────────────────────── */
.hero {
  background: linear-gradient(135deg, #0B2818 0%, #163A22 50%, #0E2415 100%);
  border-radius: 18px; padding: 28px 36px 24px;
  margin-bottom: 22px; position: relative; overflow: hidden;
}
.hero::before {
  content: ''; position: absolute; top: -60px; right: -60px;
  width: 260px; height: 260px; border-radius: 50%;
  background: radial-gradient(circle, rgba(80,200,100,0.07) 0%, transparent 70%);
}
.hero-title {
  font-family: 'Playfair Display', serif; font-size: 2.1rem; font-weight: 700;
  color: #E0F2D0; margin: 0 0 6px; letter-spacing: -0.5px; position: relative;
}
.hero-sub { font-size: 0.86rem; color: #7BC88A; margin: 0; position: relative; }
.hero-tags { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 7px; position: relative; }
.htag {
  background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14);
  color: #A8D8A8; font-size: 0.63rem; font-family: 'JetBrains Mono', monospace;
  padding: 4px 12px; border-radius: 20px; letter-spacing: 0.5px; display: inline-block;
}

/* ── Weather card ────────────────────────────────────────── */
.wx-card {
  background: linear-gradient(135deg, #163A22 0%, #0E2415 100%);
  border-radius: 16px; padding: 20px 24px; color: #fff;
  margin-bottom: 20px; position: relative; overflow: hidden;
}
.wx-card::before {
  content: ''; position: absolute; top: -40px; right: -40px;
  width: 180px; height: 180px; border-radius: 50%;
  background: radial-gradient(circle, rgba(80,200,100,0.06) 0%, transparent 70%);
}
.wx-ey { font-size: 0.63rem; letter-spacing: 1.2px; text-transform: uppercase; color: #5DBF72; font-weight: 500; margin-bottom: 6px; }
.wx-city { font-family: 'Playfair Display', serif; font-size: 1.3rem; color: #DDF0D8; margin: 0 0 2px; font-weight: 600; }
.wx-meta { font-size: 0.75rem; color: #7BC88A; margin-bottom: 16px; }
.wx-temp-row { display: flex; align-items: flex-end; gap: 14px; flex-wrap: wrap; }
.wx-temp { font-family: 'JetBrains Mono', monospace; font-size: 3.2rem; color: #E0F2D0; line-height: 1; font-weight: 500; }
.wx-cond { font-size: 0.8rem; color: #7BC88A; margin-top: 4px; }
.wx-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-top: 16px; }
.wx-stat { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.10); border-radius: 10px; padding: 10px 13px; }
.wx-stat b { display: block; font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; color: #DDF0D8; font-weight: 500; }
.wx-stat span { font-size: 0.7rem; color: #7BC88A; }
.wx-sr { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.stag { background: rgba(0,0,0,0.20); border: 1px solid rgba(255,255,255,0.10); border-radius: 20px; padding: 3px 12px; font-size: 0.69rem; color: #A8D8A8; font-family: 'JetBrains Mono', monospace; }
.wx-ab { background: rgba(255,140,0,0.12); border: 1px solid rgba(255,140,0,0.28); border-radius: 9px; padding: 9px 13px; margin-top: 10px; font-size: 0.78rem; color: #FFD8A0; }
.wx-src { font-size: 0.6rem; color: #3A6B45; margin-top: 10px; font-style: italic; }

/* ── KPI grid ────────────────────────────────────────────── */
.kg { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 20px; }
.kc { background: #FFF; border: 1px solid #D8E8D8; border-radius: 14px; padding: 16px 18px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); transition: transform 0.2s, box-shadow 0.2s; }
.kc:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.10); }
.kc.ac { border-top: 3px solid #1E6B3A; }
.kv { font-family: 'Playfair Display', serif; font-size: 2rem; color: #0B2818; line-height: 1; font-weight: 600; }
.ku { font-size: 0.82rem; color: #667766; }
.kl { font-size: 0.65rem; color: #889988; text-transform: uppercase; letter-spacing: 0.7px; margin-top: 7px; }
.kb { height: 3px; background: #E8F0E8; border-radius: 2px; margin-top: 10px; overflow: hidden; }
.kf { height: 100%; border-radius: 2px; transition: width 1.2s ease; }
.ks { font-size: 0.68rem; color: #AABBAA; margin-top: 5px; }

/* ── Section heading ─────────────────────────────────────── */
.sh { font-family: 'Playfair Display', serif; font-size: 1.05rem; color: #0B2818; border-bottom: 2px solid #C8DEC8; padding-bottom: 6px; margin: 18px 0 13px; font-weight: 600; }

/* ── Rec cards ───────────────────────────────────────────── */
.rc { background: #FFF; border: 1px solid #E0EAE0; border-left: 4px solid #CCC; border-radius: 12px; padding: 13px 17px; margin-bottom: 10px; transition: transform .15s, box-shadow .15s; }
.rc:hover { transform: translateX(4px); box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.rC { border-left-color: #C0392B; background: #FFFAFA; }
.rH { border-left-color: #E67E22; background: #FFFDF8; }
.rM { border-left-color: #D4AC0D; background: #FFFFFB; }
.rI { border-left-color: #27AE60; background: #F5FDF7; }
.rb { display: inline-block; font-size: 0.6rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; padding: 2px 8px; border-radius: 4px; margin-right: 6px; letter-spacing: 0.5px; }
.bC { background: #FDEDEC; color: #7B1D1D; }
.bH { background: #FDF2E9; color: #7D3C01; }
.bM { background: #FEF9E7; color: #6D5401; }
.bI { background: #EAFAF1; color: #1A5C35; }
.rt { font-size: 0.88rem; font-weight: 600; color: #0B2818; display: inline; }
.rcat { font-size: 0.68rem; color: #AABBAA; margin-left: 8px; }
.rd { font-size: 0.77rem; color: #445544; margin-top: 5px; line-height: 1.6; }
.ra { font-size: 0.78rem; color: #1A5C33; font-weight: 500; margin-top: 5px; }
.rm { font-size: 0.69rem; color: #AABBAA; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

/* ── Progress bar ────────────────────────────────────────── */
.prog { background: #E8F0E8; border-radius: 6px; height: 10px; overflow: hidden; margin: 6px 0; }
.pf { height: 100%; border-radius: 6px; transition: width 1.4s ease; }

/* ── Risk badge ──────────────────────────────────────────── */
.rbadge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 16px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }
.rLOW      { background: #D5F5E3; color: #145A32; }
.rMODERATE { background: #FEF9E7; color: #6E4F00; }
.rHIGH     { background: #FDF2E9; color: #7D3C01; }
.rCRITICAL { background: #FDEDEC; color: #7B1D1D; }

/* ── Data rows ───────────────────────────────────────────── */
.dr { display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 1px solid #EDF3ED; font-size: 0.78rem; }
.dr:last-child { border-bottom: none; }
.dk { color: #556655; font-weight: 500; }
.dv { font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #0B2818; background: #F0F5F0; padding: 2px 9px; border-radius: 6px; }

/* ── How-it-works steps ──────────────────────────────────── */
.sc { display: flex; align-items: flex-start; gap: 12px; background: #FFF; border: 1px solid #DDE8D8; border-radius: 12px; padding: 12px 15px; margin-bottom: 9px; }
.sn { background: #0B2818; color: #DDF0D8; min-width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 700; flex-shrink: 0; margin-top: 1px; }
.st { font-size: 0.84rem; font-weight: 600; color: #0B2818; }
.sd { font-size: 0.75rem; color: #667766; margin-top: 2px; line-height: 1.45; }

/* ── Empty / pending state ───────────────────────────────── */
.es { text-align: center; padding: 56px 20px; color: #889988; }
.ei { font-size: 4rem; margin-bottom: 12px; }
.etitle { font-family: 'Playfair Display', serif; font-size: 1.4rem; color: #0B2818; margin-bottom: 8px; }
.ebody { font-size: 0.84rem; line-height: 1.6; }

/* API info box */
.ai { background: rgba(30,107,58,0.07); border: 1px solid rgba(30,107,58,0.18); border-radius: 10px; padding: 12px 15px; font-size: 0.75rem; color: #2D5A3D; line-height: 1.7; }

/* ── Focus-visible global ────────────────────────────────── */
:focus-visible { outline: 3px solid #52B788 !important; outline-offset: 2px !important; }

/* ── High-contrast slider tick text ─────────────────────── */
.stSlider [data-testid="stThumbValue"],
.stSlider [data-testid="stSliderTickBarMin"],
.stSlider [data-testid="stSliderTickBarMax"] {
  color: #0D2B1A !important; font-size: 0.72rem !important; font-weight: 500 !important;
}

details summary { font-size: 0.83rem !important; font-weight: 500 !important; color: #2D5A3D !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _rc(risk: str) -> str:
    return {"LOW":"rLOW","MODERATE":"rMODERATE","HIGH":"rHIGH","CRITICAL":"rCRITICAL"}.get(risk,"rMODERATE")

def _pc(p: float) -> str:
    if p >= 70: return "linear-gradient(90deg,#1E6B3A,#52B788)"
    if p >= 45: return "linear-gradient(90deg,#D4AC0D,#F0C040)"
    return "linear-gradient(90deg,#C0392B,#E74C3C)"

def _kpi(val, unit, lbl, bar=0, sub="", ac=False):
    b = f"<div class='kb'><div class='kf' style='width:{bar:.0f}%;background:{_pc(bar)}'></div></div>" if bar else ""
    return (f"<div class='kc{' ac' if ac else ''}'>"
            f"<div class='kv'>{val}<span class='ku'> {unit}</span></div>"
            f"<div class='kl'>{lbl}</div>{b}"
            f"{'<div class=ks>' + sub + '</div>' if sub else ''}</div>")

def _dr(k, v):
    return f"<div class='dr'><span class='dk'>{k}</span><span class='dv'>{v}</span></div>"

def _rec_html(r):
    m  = f"<div class='rm'>📐 {r.metric}</div>" if r.metric else ""
    bc = {"CRITICAL":"bC","HIGH":"bH","MEDIUM":"bM","INFO":"bI"}[r.priority]
    rc = {"CRITICAL":"rC","HIGH":"rH","MEDIUM":"rM","INFO":"rI"}[r.priority]
    return (f"<div class='rc {rc}'>"
            f"<div><span class='rb {bc}'>{r.priority}</span>"
            f"<span class='rt'>{r.icon} {r.title}</span>"
            f"<span class='rcat'>{r.category}</span></div>"
            f"<div class='rd'>{r.detail}</div>"
            f"<div class='ra'>💡 {r.action}</div>{m}</div>")

def _slider_row(icon: str, label: str, value, unit: str,
                lo: float, hi: float, optimal: str = "") -> str:
    """Render a labelled slider row with live value badge and range hints."""
    hint_html = (
        f"<div class='slider-hint'>"
        f"<span>{lo} {unit}</span>"
        f"{'<span class=optimal>✓ optimal: ' + optimal + '</span>' if optimal else '<span></span>'}"
        f"<span>{hi} {unit}</span>"
        f"</div>"
    )
    return (
        f"<div class='slider-label-row'>"
        f"<span class='slider-label'><span class='licon'>{icon}</span>{label}</span>"
        f"<span class='slider-value-badge'>{value} {unit}</span>"
        f"</div>"
        f"{hint_html}"
    )


# ─────────────────────────────────────────────────────────────
#  CACHED ML MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _build():
    import torch
    from src.data_generator    import generate_dataset
    from src.preprocessing     import enrich_weather, enrich_soil, Scaler, make_loaders
    from src.models.fusion_model import FusionModel
    from src.trainer           import train, evaluate

    Path("outputs").mkdir(exist_ok=True)
    ds       = generate_dataset(n=1200)
    wx_eng   = enrich_weather(ds["wx"])
    soil_eng = enrich_soil(ds["soil"]).values.astype(np.float32)
    sc       = Scaler()
    soil_s, wx_s, sat_s, y_s = sc.fit_transform(soil_eng, wx_eng, ds["sat"], ds["yields"])
    tr, va, te = make_loaders(soil_s, wx_s, sat_s, y_s, batch=32)
    model  = FusionModel(sat_ch=ds["sat"].shape[-1], wx_f=wx_eng.shape[-1], soil_f=soil_eng.shape[-1])
    device = torch.device("cpu")
    train(model, tr, va, device, sc, epochs=25, lr=8e-4, patience=10, save_dir=Path("outputs"))
    m = evaluate(model, te, device, sc)
    return model, sc, m


with st.spinner("⚙️ Initialising CropSense AI — one-time training (~30 s)…"):
    try:
        ML_MODEL, ML_SCALER, ML_M = _build()
        _ok = True
    except Exception as _e:
        st.error(f"Model init failed: {_e}"); _ok = False


# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────

for _k, _v in dict(weather=None, result=None, fetch_err="").items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

CROPS   = ["Wheat", "Rice", "Maize", "Soybean", "Cotton"]
C_EMOJI = {"Wheat":"🌾","Rice":"🌾","Maize":"🌽","Soybean":"🫘","Cotton":"☁️"}


# ─────────────────────────────────────────────────────────────
#  SIDEBAR  ← the completely redesigned input system
# ─────────────────────────────────────────────────────────────

wx_loaded   = st.session_state["weather"] is not None
result_done = st.session_state["result"]  is not None

with st.sidebar:

    # ── App brand ─────────────────────────────────────────────
    st.markdown(
        "<div style='padding:14px 4px 10px;'>"
        "<div style='font-family:Playfair Display,serif;font-size:1.3rem;"
        "color:#DDF0D8;font-weight:700;'>🌾 CropSense AI</div>"
        "<div style='font-size:0.7rem;color:#5DBF72;margin-top:2px;letter-spacing:0.5px;'>"
        "Smart Farm Yield Advisor</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════
    # CARD 1 — LOCATION
    # ═══════════════════════════════════════════
    done1 = wx_loaded
    card1_cls = "input-card done" if done1 else "input-card active"
    done_badge = "<span class='step-done-badge'>✓ Done</span>" if done1 else ""

    st.markdown(
        f"<div class='{card1_cls}'>"
        f"<div class='card-hd'>"
        f"<div class='step-circle{'  done' if done1 else ''}'>1</div>"
        f"<div><div class='card-title'>Your Location</div>"
        f"<div class='card-sub'>City, district or village</div></div>"
        f"{done_badge}"
        f"</div>",
        unsafe_allow_html=True,
    )
    city = st.text_input(
        "Location",
        placeholder="e.g.  Jaipur, India",
        label_visibility="collapsed",
        help="Type any city worldwide. Press Fetch to load real weather data.",
    )

    if wx_loaded and st.session_state["weather"]:
        wd_loc = st.session_state["weather"].location
        st.markdown(
            f"<div style='background:#F0FBF4;border:1px solid #B2DFC0;border-radius:8px;"
            f"padding:7px 11px;font-size:0.78rem;color:#1A5C33;margin-top:8px;'>"
            f"✅ <b>{wd_loc.name}, {wd_loc.state}, {wd_loc.country}</b><br>"
            f"<span style='font-size:0.7rem;color:#4A8A60;'>"
            f"Weather loaded · {wd_loc.lat:.2f}°N, {wd_loc.lon:.2f}°E</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)  # close card

    fetch_btn = st.button(
        "🌐  Fetch Live Weather",
        type="primary",
        use_container_width=True,
        disabled=not city.strip(),
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # CARD 2 — CROP
    # ═══════════════════════════════════════════
    done2 = wx_loaded   # crop is always set; mark done once weather is loaded
    card2_cls = "input-card done" if done2 else ("input-card active" if wx_loaded else "input-card")

    st.markdown(
        f"<div class='{card2_cls}'>"
        f"<div class='card-hd'>"
        f"<div class='step-circle'>2</div>"
        f"<div><div class='card-title'>Crop & Farm</div>"
        f"<div class='card-sub'>What are you growing?</div></div>"
        f"</div>"
        f"<div class='crop-label'>Select crop</div>",
        unsafe_allow_html=True,
    )
    crop    = st.selectbox("Crop", CROPS, label_visibility="collapsed")
    st.markdown(
        f"<div style='font-size:0.84rem;font-weight:600;color:#0D2B1A;"
        f"margin:10px 0 5px;'>Farm area</div>",
        unsafe_allow_html=True,
    )
    farm_ha = st.number_input(
        "Farm area (hectares)",
        min_value=0.5, max_value=500.0, value=2.0, step=0.5,
        label_visibility="collapsed",
        help="Total cultivated area in hectares",
    )
    st.markdown(
        f"<div style='font-size:0.7rem;color:#6A8A6A;margin-top:4px;'>"
        f"Hectares (1 ha = 2.47 acres)</div>"
        f"</div>",  # close card
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # CARD 3 — SOIL TEST
    # ═══════════════════════════════════════════
    done3 = result_done
    card3_cls = "input-card done" if done3 else "input-card active"

    st.markdown(
        f"<div class='{card3_cls}'>"
        f"<div class='card-hd'>"
        f"<div class='step-circle'>3</div>"
        f"<div><div class='card-title'>Soil Test Values</div>"
        f"<div class='card-sub'>From your soil health card / lab report</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── pH ────────────────────────────────────
    soil_ph = st.slider("soil_ph_sl", 4.5, 8.5, 6.5, 0.1, label_visibility="collapsed")
    st.markdown(_slider_row("⚗️", "Soil pH", f"{soil_ph:.1f}", "", 4.5, 8.5, "6.0–7.0"), unsafe_allow_html=True)

    # ── Moisture ──────────────────────────────
    soil_mois = st.slider("soil_mois_sl", 5, 65, 30, 1, label_visibility="collapsed")
    st.markdown(_slider_row("💧", "Soil Moisture", soil_mois, "%", 5, 65, "25–45%"), unsafe_allow_html=True)

    # ── Nitrogen ──────────────────────────────
    nitrogen = st.slider("n_sl", 5, 280, 100, 5, label_visibility="collapsed")
    st.markdown(_slider_row("🌱", "Nitrogen (N)", nitrogen, "kg/ha", 5, 280, "80–150"), unsafe_allow_html=True)

    # ── Phosphorus ────────────────────────────
    phosphorus = st.slider("p_sl", 1, 80, 35, 1, label_visibility="collapsed")
    st.markdown(_slider_row("🧪", "Phosphorus (P)", phosphorus, "kg/ha", 1, 80, "40–70"), unsafe_allow_html=True)

    # ── Potassium ─────────────────────────────
    potassium = st.slider("k_sl", 20, 320, 100, 5, label_visibility="collapsed")
    st.markdown(_slider_row("⚡", "Potassium (K)", potassium, "kg/ha", 20, 320, "80–150"), unsafe_allow_html=True)

    # ── Organic Carbon ────────────────────────
    org_c = st.slider("oc_sl", 0.10, 3.50, 0.80, 0.05, label_visibility="collapsed")
    st.markdown(_slider_row("🍂", "Organic Carbon", f"{org_c:.2f}", "%", 0.1, 3.5, ">0.75%"), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close card

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Predict button ────────────────────────
    predict_btn = st.button(
        f"🔍  Predict {C_EMOJI.get(crop,'')} {crop} Yield",
        type="primary",
        use_container_width=True,
        disabled=(not wx_loaded),
    )
    if not wx_loaded:
        st.markdown(
            "<div style='text-align:center;font-size:0.75rem;color:#4A7A54;"
            "margin-top:6px;padding:7px 10px;background:rgba(255,255,255,0.06);"
            "border-radius:8px;'>⬆️  Complete Step 1 first</div>",
            unsafe_allow_html=True,
        )

    # ── Credit ────────────────────────────────
    st.markdown(
        "<div class='sb-credit'>"
        "<b>Weather</b>  open-meteo.com · Free · No API key<br>"
        "<b>Model</b>  CNN + BiLSTM + MLP Fusion<br>"
        "<b>Advisory</b>  9-category rule engine"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
#  WEATHER FETCH
# ─────────────────────────────────────────────────────────────

if fetch_btn and city.strip():
    with st.spinner(f"🌐 Fetching weather for **{city}** via Open-Meteo…"):
        try:
            from src.weather import get_weather
            wd = get_weather(city.strip())
            st.session_state.update(weather=wd, fetch_err="", result=None)
        except RuntimeError as exc:
            st.session_state.update(fetch_err=str(exc), weather=None)
        except Exception as exc:
            st.session_state.update(fetch_err=f"Unexpected error: {exc}", weather=None)


# ─────────────────────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────────────────────

def _predict():
    import torch
    from src.data_generator  import generate_satellite_patches
    from src.preprocessing   import enrich_weather, enrich_soil
    from src.advisor         import advise
    from src.weather         import to_model_inputs

    wd = st.session_state["weather"]
    ph_dev    = abs(soil_ph - 6.5)
    fertility = float(np.clip(
        (nitrogen   / 200) * 0.35 + (org_c      / 3.0) * 0.25 +
        (soil_mois  / 60 ) * 0.20 + (1 - ph_dev / 3  ) * 0.15 +
        (phosphorus / 70 ) * 0.05, 0.05, 1.0))
    health = float(np.clip(fertility * (1 - wd.stress * 0.6), 0.05, 1.0))

    sat      = generate_satellite_patches(1, np.array([health], dtype=np.float32))
    wx_raw   = wd.weekly[np.newaxis, :, :]
    wx_eng   = enrich_weather(wx_raw)
    soil_df  = pd.DataFrame([{
        "soil_ph": soil_ph, "soil_moisture_pct": soil_mois,
        "organic_carbon_pct": org_c, "nitrogen_kg_ha": nitrogen,
        "phosphorus_kg_ha": phosphorus, "potassium_kg_ha": potassium,
        "clay_pct": 25.0, "sand_pct": 40.0, "silt_pct": 35.0,
        "bulk_density_g_cm3": 1.3, "cec_meq_100g": 15.0,
    }])
    soil_eng = enrich_soil(soil_df).values.astype(np.float32)
    soil_s, wx_s, sat_s = ML_SCALER.transform_single(soil_eng, wx_eng, sat)

    sat_t  = torch.from_numpy(sat_s).permute(0, 3, 1, 2)
    wx_t   = torch.from_numpy(wx_s)
    soil_t = torch.from_numpy(soil_s)

    ML_MODEL.eval()
    with torch.no_grad():
        scaled = ML_MODEL(sat_t, wx_t, soil_t).numpy().ravel()
    py  = float(max(0.1, ML_SCALER.inverse_y(scaled.astype(np.float32))[0]))
    adv = advise(
        crop=crop, region=wd.location.state or wd.location.country,
        soil={"soil_ph": soil_ph, "soil_moisture_pct": soil_mois,
              "nitrogen_kg_ha": nitrogen, "phosphorus_kg_ha": phosphorus,
              "potassium_kg_ha": potassium, "organic_carbon_pct": org_c},
        wx=to_model_inputs(wd), pred_yield=py,
        climate_stress=wd.stress, fertility=fertility,
    )
    st.session_state["result"] = dict(
        py=py, adv=adv, crop=crop, farm_ha=farm_ha,
        soil_ph=soil_ph, soil_mois=soil_mois, nitrogen=nitrogen,
        phosphorus=phosphorus, potassium=potassium, org_c=org_c,
    )


if predict_btn and wx_loaded and _ok:
    with st.spinner("🌾 Running multimodal inference with live weather data…"):
        try:
            _predict()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────

st.markdown(
    "<div class='hero'>"
    "<div class='hero-title'>🌾 CropSense AI</div>"
    "<p class='hero-sub'>Climate-Resilient Yield Predictor · Real-Time Weather · Personalised Farm Advisory</p>"
    "<div class='hero-tags'>"
    "<span class='htag'>📍 ANY CITY WORLDWIDE</span>"
    "<span class='htag'>🌐 OPEN-METEO LIVE</span>"
    "<span class='htag'>12-WEEK HISTORY → BiLSTM</span>"
    "<span class='htag'>CNN + BiLSTM + MLP</span>"
    "<span class='htag'>ZERO API KEY</span>"
    "</div></div>",
    unsafe_allow_html=True,
)

# Error banner
if st.session_state["fetch_err"]:
    st.error(
        f"❌ **Weather fetch failed** — {st.session_state['fetch_err']}\n\n"
        "Try a more specific city name, e.g. `Jaipur, India` or `Berlin, Germany`."
    )

# ─────────────────────────────────────────────────────────────
#  EMPTY STATE
# ─────────────────────────────────────────────────────────────

if not wx_loaded and not st.session_state["fetch_err"]:
    col_l, col_r = st.columns([1.1, 1])
    with col_l:
        st.markdown("<div class='sh'>📋 How it works</div>", unsafe_allow_html=True)
        for n, t, d in [
            ("1", "Enter your location", "Type any city — geocoded to GPS automatically."),
            ("2", "Live weather fetched", "Open-Meteo pulls current + 12 weeks of real history."),
            ("3", "Enter soil test values", "N · P · K · pH · moisture · organic carbon — all with range guides."),
            ("4", "Select crop & predict", "One click → CNN + BiLSTM + MLP fusion inference."),
            ("5", "Receive farm report", "Yield forecast + prioritised recommendations + JSON export."),
        ]:
            st.markdown(
                f"<div class='sc'><div class='sn'>{n}</div>"
                f"<div><div class='st'>{t}</div><div class='sd'>{d}</div></div></div>",
                unsafe_allow_html=True,
            )
    with col_r:
        st.markdown("<div class='sh'>🌐 Weather pipeline</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='ai'>"
            "<b>Three API calls — all free, no key:</b><br><br>"
            "① Geocoding → city name to GPS<br>"
            "② Archive → past 84 days of daily weather<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;→ 12 weekly rows → BiLSTM input<br>"
            "③ Forecast → 7-day outlook + current<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;→ stress indices + weather alerts<br><br>"
            "<b>Weather features (9 total):</b><br>"
            "Temp · rain · humidity · solar · temp anomaly<br>"
            "cumulative rain · drought index · heat stress<br><br>"
            "<b>Season auto-detected:</b><br>"
            "Kharif / Rabi / Zaid for India<br>"
            "Spring / Summer / Autumn / Winter elsewhere"
            "</div>",
            unsafe_allow_html=True,
        )
        if ML_M:
            st.markdown("<div class='sh'>📊 Model performance</div>", unsafe_allow_html=True)
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("R²",   f"{ML_M.get('R2',0):.3f}")
            mc2.metric("RMSE", f"{ML_M.get('RMSE',0):.3f}")
            mc3.metric("MAE",  f"{ML_M.get('MAE',0):.3f}")
            mc4.metric("MAPE", f"{ML_M.get('MAPE',0):.1f}%")
            st.caption("Evaluated on held-out synthetic test set.")
    st.stop()


# ─────────────────────────────────────────────────────────────
#  WEATHER CARD
# ─────────────────────────────────────────────────────────────

wd    = st.session_state["weather"]
if wd is None: st.stop()

rlvl  = "LOW" if wd.stress<0.30 else "MODERATE" if wd.stress<0.50 else "HIGH" if wd.stress<0.70 else "CRITICAL"
rico  = {"LOW":"🟢","MODERATE":"🟡","HIGH":"🟠","CRITICAL":"🔴"}[rlvl]
aalrt = "".join(f"<div class='wx-ab'>{a}</div>" for a in wd.alerts)

st.markdown(
    f"<div class='wx-card'>"
    f"<div class='wx-ey'>🌐 Live weather · {wd.fetched_at}</div>"
    f"<div class='wx-city'>📍 {wd.location.name}, {wd.location.state}, {wd.location.country}</div>"
    f"<div class='wx-meta'>🗓️ {wd.season} &nbsp;·&nbsp; 🏔️ {wd.location.elevation:.0f}m"
    f" &nbsp;·&nbsp; {wd.location.lat:.2f}°N, {wd.location.lon:.2f}°E</div>"
    f"<div class='wx-temp-row'>"
    f"<div><div class='wx-temp'>{wd.temp_c}°C</div>"
    f"<div class='wx-cond'>{wd.condition} · feels {wd.feels_c}°C"
    f" · 💨 {wd.wind_kmh} km/h · 💧 {wd.humidity}%</div></div>"
    f"<div style='margin-left:auto'>"
    f"<span class='rbadge {_rc(rlvl)}'>{rico} {rlvl} risk</span>"
    f"</div></div>"
    f"<div class='wx-grid'>"
    f"<div class='wx-stat'><b>{wd.avg_temp}°C</b><span>7-day avg temp</span></div>"
    f"<div class='wx-stat'><b>{wd.max_temp}°C</b><span>7-day max temp</span></div>"
    f"<div class='wx-stat'><b>{wd.min_temp}°C</b><span>7-day min temp</span></div>"
    f"<div class='wx-stat'><b>{wd.rain_7d} mm</b><span>7-day rainfall</span></div>"
    f"<div class='wx-stat'><b>{wd.avg_hum:.0f}%</b><span>avg humidity</span></div>"
    f"<div class='wx-stat'><b>{wd.solar_mjm2} MJ/m²</b><span>solar/day</span></div>"
    f"</div>"
    f"<div class='wx-sr'>"
    f"<span class='stag'>stress {wd.stress:.2f}</span>"
    f"<span class='stag'>drought {wd.drought:.2f}</span>"
    f"<span class='stag'>heat {wd.heat:.2f}</span>"
    f"<span class='stag'>12wk history → BiLSTM ✓</span>"
    f"</div>"
    f"{aalrt}"
    f"<div class='wx-src'>Source: {wd.source}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

with st.expander("📊 12-Week Historical Weather (BiLSTM input)", expanded=False):
    wdf = pd.DataFrame(
        wd.weekly,
        columns=["Avg Temp °C","Max Temp °C","Rain mm","Humidity %","Solar MJ/m²"],
        index=[f"Wk {i+1}" for i in range(12)],
    )
    ca, cb = st.columns(2)
    ca.line_chart(wdf[["Avg Temp °C","Max Temp °C"]], height=165)
    cb.bar_chart(wdf["Rain mm"], height=165)
    st.caption("12 weeks of real daily weather enriched with temp-anomaly, drought-index, heat-stress features and fed into the BiLSTM branch.")


# ─────────────────────────────────────────────────────────────
#  PENDING STATE
# ─────────────────────────────────────────────────────────────

if st.session_state["result"] is None:
    st.markdown(
        "<div class='es'><div class='ei'>🌱</div>"
        "<div class='etitle'>Weather loaded — ready to predict</div>"
        "<div class='ebody'>Adjust the soil values in the sidebar and click<br>"
        "<b>Predict Yield</b> to run the analysis.</div></div>",
        unsafe_allow_html=True,
    )
    st.stop()


# ─────────────────────────────────────────────────────────────
#  RESULTS DASHBOARD
# ─────────────────────────────────────────────────────────────

R   = st.session_state["result"]
adv = R["adv"]
py  = R["py"]
ha  = R["farm_ha"]

# KPI row
st.markdown(
    "<div class='kg'>"
    + _kpi(f"{py:.2f}", "t/ha", f"Predicted yield · {R['crop']}",
           bar=py/adv.max_yield*100, ac=True)
    + _kpi(f"{adv.gap_pct:.0f}", "%", "Yield gap vs potential",
           bar=adv.gap_pct, sub=f"Potential: {adv.max_yield:.1f} t/ha")
    + _kpi(f"{py*ha:.1f}", "t", "Total farm production",
           bar=min(100, py/adv.max_yield*100), sub=f"{ha:.1f} ha · {py:.2f} t/ha")
    + _kpi(f"{len(adv.recs)}", "actions", "Advisory items",
           bar=adv.confidence,
           sub=f"{sum(1 for r in adv.recs if r.priority=='CRITICAL')} critical"
               f" · {adv.confidence}% confidence")
    + "</div>",
    unsafe_allow_html=True,
)

# Two-column results
cl, cr = st.columns([1.45, 1])

with cl:
    st.markdown("<div class='sh'>📋 Personalised Recommendations</div>",
                unsafe_allow_html=True)
    st.caption(
        f"Powered by **live weather at {wd.location.name}** "
        f"({wd.fetched_at}) + your soil values."
    )
    if not adv.recs:
        st.success("✅ Excellent farm conditions — no critical gaps detected.")
    for rec in adv.recs:
        st.markdown(_rec_html(rec), unsafe_allow_html=True)

with cr:
    st.markdown("<div class='sh'>🌾 Yield vs Potential</div>", unsafe_allow_html=True)
    pct = max(5, min(100, int(py/adv.max_yield*100)))
    st.markdown(
        f"<div class='prog'><div class='pf' style='width:{pct}%;background:{_pc(pct)};'></div></div>"
        f"<div style='font-size:0.74rem;color:#556655;margin-top:4px;'>"
        f"{py:.2f} / {adv.max_yield:.1f} t/ha &nbsp;·&nbsp; {pct}% of potential</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='margin:12px 0;'>"
        f"<span class='rbadge {_rc(adv.risk)}'>{adv.risk_icon} {adv.risk}</span>"
        f"&nbsp;&nbsp;<span style='font-size:0.78rem;color:#667766;'>{adv.outlook}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sh'>🌡️ Weather Inputs Used</div>", unsafe_allow_html=True)
    wx_rows = [
        ("Location",      f"{wd.location.name}, {wd.location.country}"),
        ("Season",        wd.season.split("(")[0].strip()),
        ("7-day avg",     f"{wd.avg_temp}°C"),
        ("7-day max",     f"{wd.max_temp}°C"),
        ("7-day rain",    f"{wd.rain_7d} mm"),
        ("Humidity",      f"{wd.avg_hum:.0f}%"),
        ("Climate stress",f"{wd.stress:.2f}"),
        ("Drought idx",   f"{wd.drought:.2f}"),
        ("Heat stress",   f"{wd.heat:.2f}"),
        ("History",       "12 weeks real"),
    ]
    st.markdown(
        "<div style='background:#FFF;border:1px solid #D8E8D8;border-radius:12px;padding:8px 14px;'>"
        + "".join(_dr(k, v) for k, v in wx_rows)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sh'>🧪 Soil Inputs Used</div>", unsafe_allow_html=True)
    soil_rows = [
        ("pH",        f"{R['soil_ph']:.1f}"),
        ("Moisture",  f"{R['soil_mois']}%"),
        ("Nitrogen",  f"{R['nitrogen']} kg/ha"),
        ("Phosphorus",f"{R['phosphorus']} kg/ha"),
        ("Potassium", f"{R['potassium']} kg/ha"),
        ("Organic C", f"{R['org_c']:.2f}%"),
    ]
    st.markdown(
        "<div style='background:#FFF;border:1px solid #D8E8D8;border-radius:12px;padding:8px 14px;'>"
        + "".join(_dr(k, v) for k, v in soil_rows)
        + "</div>",
        unsafe_allow_html=True,
    )

# Alerts
if wd.alerts:
    st.markdown("<div class='sh'>⚠️ Live Weather Alerts</div>", unsafe_allow_html=True)
    for a in wd.alerts:
        st.warning(a)

# Export
st.divider()
export = {
    "generated_at":         datetime.now().isoformat(),
    "location":             {"city": wd.location.name, "state": wd.location.state,
                             "country": wd.location.country,
                             "lat": wd.location.lat, "lon": wd.location.lon},
    "season":               wd.season,
    "crop":                 R["crop"],
    "farm_ha":              ha,
    "predicted_yield_t_ha": round(py, 3),
    "yield_potential_t_ha": adv.max_yield,
    "yield_gap_pct":        round(adv.gap_pct, 1),
    "confidence_pct":       adv.confidence,
    "risk":                 adv.risk,
    "weather": {
        "source": wd.source, "fetched_at": wd.fetched_at,
        "avg_temp": wd.avg_temp, "max_temp": wd.max_temp,
        "rain_7d_mm": wd.rain_7d, "humidity": wd.avg_hum,
        "stress": wd.stress, "drought": wd.drought, "heat": wd.heat,
    },
    "soil": {
        "ph": R["soil_ph"], "moisture": R["soil_mois"],
        "N_kg_ha": R["nitrogen"], "P_kg_ha": R["phosphorus"],
        "K_kg_ha": R["potassium"], "OC_pct": R["org_c"],
    },
    "recommendations": [
        dict(priority=r.priority, category=r.category,
             title=r.title, action=r.action)
        for r in adv.recs
    ],
    "model_metrics": ML_M,
}
fname = (
    f"cropsense_{R['crop'].lower()}_"
    f"{wd.location.name.lower().replace(' ','_')}_"
    f"{datetime.now().strftime('%Y%m%d')}.json"
)
st.download_button(
    "⬇️  Download Farm Report (JSON)",
    data=json.dumps(export, indent=2),
    file_name=fname,
    mime="application/json",
)

st.markdown(
    "<div style='text-align:center;padding:22px 0 8px;font-size:0.68rem;color:#AABBAA;'>"
    "🌾 CropSense AI &nbsp;·&nbsp; CNN + BiLSTM + MLP &nbsp;·&nbsp; "
    "Weather: <a href='https://open-meteo.com' style='color:#7BC88A;'>open-meteo.com</a>"
    " (free) &nbsp;·&nbsp; SDG 2: Zero Hunger"
    "</div>",
    unsafe_allow_html=True,
)
