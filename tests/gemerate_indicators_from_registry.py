import sys
import itertools
from pathlib import Path

# Añade la raíz del proyecto al sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.features.targets import NormalizedFutureReturn
from src.eda.eda import prepare_data

from src.evaluation.indicator_registry import INDICATOR_REGISTRY

path = "data/raw/1d/BTCUSDT_1d_01-01-2016_18-01-2026.csv"
df   = pd.read_csv(path, sep=",")
df   = prepare_data(df)

# ---------------------------------------------------------------------------
# Genera TODAS las combinaciones de parámetros de cada familia del registro
# ---------------------------------------------------------------------------
def param_combinations(params_grid: dict) -> list[dict]:
    """Devuelve todas las combinaciones de un grid de parámetros como lista de dicts."""
    keys = list(params_grid.keys())
    values = list(params_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


total_combos = sum(
    len(param_combinations(spec["params"]))
    for spec in INDICATOR_REGISTRY.values()
)
done = 0
n_ok = 0
n_failed = 0

print(f"Generando {total_combos} combinaciones a partir del registro de OptunaWalkForwardFixed...\n")

for family_name, spec in INDICATOR_REGISTRY.items():
    cls = spec["cls"]
    combos = param_combinations(spec["params"])

    for params in combos:
        done += 1
        try:
            indicator = cls(df, **params)
            indicator.compute()
            df = pd.concat([df, indicator.result], axis=1)
            n_ok += 1
        except Exception as e:
            n_failed += 1
            print(f"[{done}/{total_combos}] {family_name} FAILED con {params}: {e}")
            continue

        print(f"[{done}/{total_combos}] {family_name:<22} {params} OK — columnas: {list(indicator.result.columns)}")

# ---------------------------------------------------------------------------
# Target: retorno futuro normalizado
# ---------------------------------------------------------------------------
target = NormalizedFutureReturn(df, window=14)
target.compute()
df = pd.concat([df, target.result], axis=1)
print(f"\nTarget {target.name} OK — columnas: {list(target.result.columns)}")

print(f"\nResumen: {n_ok} indicadores generados correctamente, {n_failed} fallidos.")
print(f"Shape final del DataFrame: {df.shape}")

# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------
df.to_csv("data/processed/BTCUSDT_1d_01-01-2016_18-01-2026.csv")
print("\nCSV exportado correctamente.")