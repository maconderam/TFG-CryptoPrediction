import sys
from pathlib import Path

# Añade la raíz del proyecto al sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.indicator_selector import IndicatorSelector, plot_period_correlation
from src.eda.eda import prepare_data
import pandas as pd
import matplotlib.pyplot as plt

path = "data/raw/4h/BTCUSDT_4h_01-01-2016_18-01-2026.csv"

df = pd.read_csv(path, sep=",")
df = prepare_data(df)

df = df.dropna()
"""
sel = IndicatorSelector(df, 
                        train_window=1000, 
                        test_window=500,
                        min_kept=400,
                        n_mcpt=50, 
                        p_threshold=0.05)
df_results = sel.run()                        # grid search completo
print(sel.top_n(10))                  # top 10 significativos

fig1 = sel.plot_summary(top_n=20)     # dashboard comparativo
fig1.show()

fig2 = sel.plot_mc_distributions(top_n=6)
fig2.show()

fig3 = sel.plot_significant()  # solo señales significativas
fig3.show()
"""

ultima_fecha = df["timestamp"].max()
fecha_corte = ultima_fecha - pd.DateOffset(years=3)

df_train = df[df["timestamp"] < fecha_corte].copy()
df_test = df[df["timestamp"] >= fecha_corte].copy()

sel_2017_2023 = IndicatorSelector(df_train, 
                        train_window=6000, 
                        test_window=3000,
                        min_kept=1500,
                        n_mcpt=100, 
                        p_threshold=0.05)
df_results = sel_2017_2023.run()                        # grid search completo
print(sel_2017_2023.top_n(10))                  # top 10 significativos

fig1 = sel_2017_2023.plot_summary(top_n=20)     # dashboard comparativo
fig1.show()

fig2 = sel_2017_2023.plot_mc_distributions(top_n=6)
fig2.show()

fig3 = sel_2017_2023.plot_significant()  # solo señales significativas
fig3.show()

sel_2023_2026 = IndicatorSelector(df_test, 
                        train_window=4000, 
                        test_window=2000,
                        min_kept=1000,
                        n_mcpt=100, 
                        p_threshold=0.05)
df_results = sel_2023_2026.run()                        # grid search completo
print(sel_2023_2026.top_n(10))                  # top 10 significativos

fig1 = sel_2023_2026.plot_summary(top_n=20)     # dashboard comparativo
fig1.show()

fig2 = sel_2023_2026.plot_mc_distributions(top_n=6)
fig2.show()

fig3 = sel_2023_2026.plot_significant()  # solo señales significativas
fig3.show()

fig4 = plot_period_correlation(sel_2017_2023, sel_2023_2026)
fig4.show()