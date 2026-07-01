import sys
from pathlib import Path

# Añade la raíz del proyecto al sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.evaluation.thresholds import ThresholdEvaluator
from src.evaluation.mcpt import MonteCarloPT
from src.features.indicators import (
    RSI, Stochastic, StochasticRSI, MACD, PriceIntensity,
    ADX, Aroon, AroonOscillator, ATR, PriceChangeOscillator, CMMA, MADifference,
    PriceVarianceRatio, ChangeVarianceRatio
)
from src.features.targets import NormalizedFutureReturn
from src.eda.eda import prepare_data
import pandas as pd
import numpy as np

path = "data/raw/1d/BTCUSDT_1d_01-01-2016_18-01-2026.csv"
df   = pd.read_csv(path, sep=",")
df   = prepare_data(df)
returns = np.log(df["close"].shift(-1) / df["close"])
te = ThresholdEvaluator(returns)

indicators = [
    RSI(df, window=7),
    RSI(df, window=14),
    Stochastic(df, window=14),
    StochasticRSI(df, rsi_window=14),
    MACD(df, short_length=12, long_length=26),
    PriceIntensity(df, smooth_window=20),
    ADX(df, window=14),
    Aroon(df, window=14),
    AroonOscillator(df, window=14),
    PriceChangeOscillator(df, short_length=10, mult=5),
    CMMA(df, window=10, atr_window=14),
    MADifference(df, short_length=10, long_length=50),
    PriceVarianceRatio(df, short_length=10, mult=4),
    ChangeVarianceRatio(df, short_length=10, mult=4),
    NormalizedFutureReturn(df, window=14)
]

for ind in indicators:
    ind.compute()
    print(f"{ind.name:<25} OK — columnas: {list(ind.result.columns)}")
    df = pd.concat([df, ind.result], axis=1)
    te.evaluate_all_thresholds(ind.result.iloc[:, 0])



#df.to_csv("data/processed/BTCUSDT_1d_01-01-2016_18-01-2026.csv")
#print("\nCSV exportado correctamente.")