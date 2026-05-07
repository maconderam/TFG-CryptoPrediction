from evaluacion.visualizador import Visualizer
from evaluacion.thresholds import ThresholdEvaluator
from evaluacion.mcpt import MonteCarloPT
from indicadores.indicators import RSI, Stochastic, StochasticRSI, MACD, PriceIntensity
import indicadores.targets as targ
from eda.eda import prepare_data
from pathlib import Path
import pandas as pd

path = "datos/crudo/1d/BTCUSDT_1d_01-01-2016_18-01-2026.csv"

df = pd.read_csv(path, sep=",")
df = prepare_data(df)
df_train, df_test = df[(df.index >= "2025-01-01") & (df.index < "2026-01-01")], df[df.index >= "2026-01-01"]
te_tr = ThresholdEvaluator(df_train)
te_ts = ThresholdEvaluator(df_test)
mcpt = MonteCarloPT(df)

rsi_7 = RSI(df, window=7)
rsi_7.compute()
te_tr.evaluate_all_thresholds(rsi_7.get_result().iloc[:, 0])

result_tr = te_tr.prepare(rsi_7.get_result().iloc[:, 0], flip_sign=True).find_optimized_threshold()
result_ts_ht = te_ts.evaluate_threshold(rsi_7.get_result().iloc[:, 0], result_tr["high_thresh"])
result_ts_lt = te_ts.evaluate_threshold(rsi_7.get_result().iloc[:, 0], result_tr["low_thresh"])

print(f"RSI 7 - Threshold: {result_tr['high_thresh']:.4f}")
print(f"RSI 7 - Train: {result_tr['pf_high']:.3f}, Test: {result_ts_ht['pf_long_above']:.3f} | {result_ts_ht['pf_long_below']:.3f}")
print(f"RSI 7 - Train: {result_tr['pf_low']:.3f}, Test: {result_ts_lt['pf_short_above']:.3f} | {result_ts_lt['pf_short_below']:.3f}")
print()
# mcpt.mcpt_threshold(macd_12_26.get_result().iloc[:, 0], n_test=1000)
# mcpt.mcpt_threshold(pi_10.get_result().iloc[:, 0], n_test=1000)

# df.to_csv("datos/procesados/BTCUSDT_1d_01-01-2016_18-01-2026.csv")

#visualizer.update_data(df)
#visualizer.plot_with_indicators(panels=["rsi_7"])