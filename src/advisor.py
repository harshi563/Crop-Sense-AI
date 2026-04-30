"""
advisor.py
Rule-based advisory engine.
Takes predicted yield + real weather data + soil values
and returns prioritised, actionable recommendations.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


CROP_OPT = {
    "Wheat":   dict(temp=20, rain_wk=15, ph=6.5, N=120, P=60, K=40, max_y=5.5),
    "Rice":    dict(temp=27, rain_wk=40, ph=6.0, N=150, P=80, K=60, max_y=7.0),
    "Maize":   dict(temp=25, rain_wk=30, ph=6.2, N=180, P=90, K=80, max_y=8.0),
    "Soybean": dict(temp=24, rain_wk=20, ph=6.5, N=40,  P=60, K=60, max_y=3.5),
    "Cotton":  dict(temp=30, rain_wk=20, ph=7.0, N=120, P=60, K=60, max_y=2.5),
}


@dataclass
class Rec:
    priority: str          # CRITICAL | HIGH | MEDIUM | INFO
    icon:     str
    category: str
    title:    str
    detail:   str
    action:   str
    metric:   Optional[str] = None


@dataclass
class Advisory:
    crop:       str
    region:     str
    pred_yield: float
    max_yield:  float
    gap_pct:    float
    risk:       str
    risk_icon:  str
    outlook:    str
    confidence: int
    recs:       List[Rec] = field(default_factory=list)


_RISK = [
    (0.30, "LOW",      "🟢", "Favourable growing conditions."),
    (0.50, "MODERATE", "🟡", "Minor stress — monitor closely."),
    (0.70, "HIGH",     "🟠", "Significant stress — adapt practices."),
    (1.01, "CRITICAL", "🔴", "Extreme risk — immediate action needed."),
]


def advise(crop: str, region: str,
           soil: Dict[str, float],
           wx:   Dict[str, float],
           pred_yield: float,
           climate_stress: float,
           fertility: float) -> Advisory:

    opt  = CROP_OPT.get(crop, CROP_OPT["Wheat"])
    recs = []

    def add(pri, icon, cat, title, detail, action, metric=None):
        recs.append(Rec(pri, icon, cat, title, detail, action, metric))

    # ── Irrigation ────────────────────────────────────────────
    moist    = soil.get("soil_moisture_pct", 30)
    rain_wk  = wx.get("avg_weekly_rain_mm", 20)
    need     = opt["rain_wk"]
    deficit  = max(0, need - rain_wk)

    if moist < 20 or rain_wk < need*0.4:
        add("CRITICAL","💧","Irrigation","Severe water deficit",
            f"Soil moisture {moist:.0f}% and weekly rain {rain_wk:.1f} mm "
            f"(crop needs {need} mm/wk).",
            f"Apply {deficit:.0f} mm/week via drip or sprinkler irrigation. "
            "Irrigate early morning to reduce evaporation.",
            f"Deficit: {deficit:.0f} mm/week")
    elif moist < 30 or rain_wk < need*0.7:
        add("HIGH","💧","Irrigation","Moderate water stress",
            f"Rain {rain_wk:.1f} mm/wk below {crop} optimum ({need} mm).",
            "Increase irrigation by 20–30%. Mulch to reduce soil moisture loss.",
            f"Rain deficit: {max(0,need-rain_wk):.0f} mm/wk")
    elif rain_wk > need*2.0:
        add("MEDIUM","🌧️","Irrigation","Excess rainfall — waterlogging risk",
            f"Weekly rainfall {rain_wk:.1f} mm exceeds {crop} requirement.",
            "Clear drainage channels. Suspend all irrigation. Watch for root rot.",
            f"Excess: {rain_wk-need:.0f} mm/wk")

    # ── Nitrogen ──────────────────────────────────────────────
    N_have = soil.get("nitrogen_kg_ha", 80); N_need = opt["N"]
    if N_have < N_need*0.45:
        add("CRITICAL","🌱","Fertilizer","Critical nitrogen deficiency",
            f"Soil N = {N_have:.0f} kg/ha (target {N_need} kg/ha for {crop}).",
            f"Apply {N_need-N_have:.0f} kg/ha urea in 2 split doses "
            "(basal + top-dress at tillering).",
            f"N gap: {N_need-N_have:.0f} kg/ha")
    elif N_have < N_need*0.75:
        add("HIGH","🌱","Fertilizer","Low nitrogen — top-dress needed",
            f"N at {N_have:.0f} kg/ha; {crop} needs {N_need} kg/ha.",
            f"Top-dress with {(N_need-N_have)*0.6:.0f} kg/ha urea now.",
            f"N: {N_have:.0f}/{N_need} kg/ha")

    # ── Phosphorus ────────────────────────────────────────────
    P_have = soil.get("phosphorus_kg_ha", 30); P_need = opt["P"]
    if P_have < P_need*0.4:
        add("HIGH","🧪","Fertilizer","Phosphorus deficiency",
            f"Soil P = {P_have:.0f} kg/ha (target {P_need} kg/ha).",
            f"Apply SSP: {(P_need-P_have)*2.2:.0f} kg/ha as basal dose before sowing.",
            f"P gap: {P_need-P_have:.0f} kg/ha")

    # ── pH ────────────────────────────────────────────────────
    ph = soil.get("soil_ph", 6.5); ph_opt = opt["ph"]
    if ph < ph_opt - 0.9:
        add("HIGH","⚗️","Soil Health","Soil too acidic",
            f"pH {ph:.1f} — {crop} optimum is {ph_opt}±0.5.",
            f"Apply lime: {(ph_opt-ph)*1200:.0f} kg/ha. Wait 3–4 weeks before sowing.",
            f"pH deviation: {ph_opt-ph:.1f}")
    elif ph > ph_opt + 0.9:
        add("MEDIUM","⚗️","Soil Health","Soil too alkaline",
            f"pH {ph:.1f} above {crop} optimum.",
            "Apply gypsum 150–200 kg/ha or sulphur. Use ammonium sulphate fertiliser.",
            f"pH: {ph:.1f} (target {ph_opt})")

    # ── Heat stress ───────────────────────────────────────────
    max_t = wx.get("max_temp", 30); opt_t = opt["temp"]
    if max_t > opt_t + 10:
        add("CRITICAL","🌡️","Climate","Extreme heat stress",
            f"Forecast max {max_t:.0f}°C — {crop} optimal is {opt_t}°C.",
            "Increase irrigation frequency. Apply kaolin spray to leaves. "
            "Consider shade nets during grain-filling.",
            f"Excess heat: +{max_t-opt_t:.0f}°C above optimum")
    elif max_t > opt_t + 5:
        add("MEDIUM","☀️","Climate","Elevated temperature",
            f"Max {max_t:.0f}°C slightly above {crop} comfort zone.",
            "Maintain adequate soil moisture. Plan for heat-tolerant variety next season.",
            f"Max temp: {max_t:.0f}°C vs {opt_t}°C optimum")

    # ── Climate stress ────────────────────────────────────────
    if climate_stress > 0.6:
        add("HIGH","🌀","Climate","High seasonal climate stress",
            f"Climate stress index {climate_stress:.2f} (high variability detected).",
            "Enrol in crop insurance. Stagger sowing dates by 1–2 weeks. "
            "Plant stress-tolerant varieties.",
            f"Stress index: {climate_stress:.2f}/1.0")

    # ── Organic carbon ────────────────────────────────────────
    oc = soil.get("organic_carbon_pct", 0.8)
    if oc < 0.5:
        add("MEDIUM","🍂","Soil Health","Low organic carbon / SOM",
            f"Soil OC {oc:.2f}% — healthy soils need >0.75%.",
            "Incorporate crop residues (don't burn). Apply 2–3 t/ha FYM. "
            "Green-manure with legumes in next cycle.",
            f"OC: {oc:.2f}% (target >0.75%)")

    # ── Yield gap ─────────────────────────────────────────────
    max_y = opt["max_y"]
    gap   = max(0, (max_y - pred_yield) / max_y * 100)
    if gap > 35:
        add("INFO","📈","Yield","Significant yield gap",
            f"Predicted {pred_yield:.2f} t/ha vs potential {max_y} t/ha.",
            "Addressing the top 2–3 nutrient/water gaps above could recover "
            f"{max_y*0.3:.1f}–{max_y*0.5:.1f} t/ha. Contact your KVK for a farm visit.",
            f"Gap: {max_y-pred_yield:.2f} t/ha ({gap:.0f}%)")

    # Sort by priority
    order = {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"INFO":3}
    recs.sort(key=lambda r: order.get(r.priority, 4))

    # Risk level
    risk, r_icon, outlook = "MODERATE","🟡","Monitor closely."
    for thr, lbl, ic, msg in _RISK:
        if climate_stress < thr:
            risk, r_icon, outlook = lbl, ic, msg; break

    confidence = int(np.clip(85 - climate_stress*22 + fertility*12, 60, 96))

    return Advisory(
        crop=crop, region=region,
        pred_yield=pred_yield, max_yield=max_y,
        gap_pct=round(gap,1), risk=risk, risk_icon=r_icon,
        outlook=outlook, confidence=confidence, recs=recs,
    )


import numpy as np   # needed for np.clip in confidence calc
