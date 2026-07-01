import sys
from pathlib import Path

# Añade la raíz del proyecto al sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
                
from src.features.indicators import (
    MACD, PriceIntensity,
    PriceChangeOscillator, CMMA, MADifference,
    PriceVarianceRatio, ChangeVarianceRatio
)
from src.eda.eda import prepare_data
import pandas as pd

path = "data/raw/1d/BTCUSDT_1d_01-01-2016_18-01-2026.csv"
df   = pd.read_csv(path, sep=",")
df   = prepare_data(df)

# ------------------------------------------------------------------
# Indicadores donde el parametro normalize tiene efecto real.
# ADX, Aroon, AroonOscillator y ATR quedan fuera porque normalize
# no afecta a su calculo.
# ------------------------------------------------------------------
NORMALIZABLE = {
    #"MACD":                  (MACD,                  dict(short_length=12, long_length=26)),
    #"PriceIntensity":        (PriceIntensity,        dict(smooth_window=20)),
    #"PriceChangeOscillator": (PriceChangeOscillator, dict(short_length=10, mult=5)),
    #"CMMA":                  (CMMA,                  dict(window=10, atr_window=14)),
    "MADifference":          (MADifference,          dict(short_length=10, long_length=50)),
    "PriceVarianceRatio":    (PriceVarianceRatio,    dict(short_length=10, mult=4)),
    "ChangeVarianceRatio":   (ChangeVarianceRatio,   dict(short_length=10, mult=4)),
}

for name, (cls, kwargs) in NORMALIZABLE.items():

    print(f"\n{'='*60}")
    print(f"{name} — version CRUDA (normalize=False)")
    print(f"{'='*60}")
    ind_raw = cls(df, **kwargs, normalize=False)
    ind_raw.compute()
    ind_raw.calculate_entropy()
    ind_raw.calculate_stats()

    print(f"\n{'='*60}")
    print(f"{name} — version NORMALIZADA (normalize=True)")
    print(f"{'='*60}")
    ind_norm = cls(df, **kwargs, normalize=True)
    ind_norm.compute()
    ind_norm.calculate_entropy()
    ind_norm.calculate_stats()