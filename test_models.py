import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
 
from modelos.models import SklearnModel
from evaluacion.walkforward import WalkForwardEvaluator
 
# ------------------------------------------------------------------
# 1. Carga y limpieza
# ------------------------------------------------------------------
df = pd.read_csv("datos/procesados/BTCUSDT_1d_01-01-2016_18-01-2026.csv")
df = df.dropna()
 
features = ["rsi_7", "%K_7", "%D_7"]
target   = "nfr_1_atr_14"
 
# ------------------------------------------------------------------
# 2. Walk-forward
# ------------------------------------------------------------------
wf = WalkForwardEvaluator(
    df,
    features=features,
    target=target,
    train_window=500,
    test_window=100,
    expanding=False
)
 
sklearn_pipe = Pipeline([
    ("regresion", LinearRegression())
])
model = SklearnModel(sklearn_pipe, name="LinearRegression")
 
fold_results, predictions_df = wf.run(model)
wf.summary()