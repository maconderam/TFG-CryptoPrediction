# TFG — Análisis e implementación de indicadores y modelos de aprendizaje automático para la predicción de series temporales financieras

Framework para evaluar si los indicadores técnicos clásicos, combinados con modelos de aprendizaje automático, tienen capacidad predictiva sobre criptomonedas (Bitcoin y Ethereum), bajo un protocolo riguroso de validación temporal (walk-forward), optimización de hiperparámetros (Optuna) y contraste de significancia estadística (tests de permutación Monte Carlo).

## Estructura del proyecto

```
Proyecto/
├── requirements.txt
├── data/
│   ├── raw/                          # CSVs OHLCV originales de Binance (por resolución: 1d, 4h, 1h)
│   └── processed/                    # CSVs con indicadores/target ya calculados (opcional, ver más abajo)
├── src/
│   ├── eda/
│   │   └── eda.py                        # prepare_data(): limpieza inicial del OHLCV
│   ├── features/
│   │   ├── indicators.py                 # 17 clases de indicadores (RSI, MACD, ADX, ATR, ...)
│   │   ├── targets.py                    # NormalizedFutureReturn (variable objetivo)
│   │   └── indicator_selector.py         # IndicatorSelector + plot_period_correlation (Experimento 1)
│   ├── models/
│   │   ├── models.py                     # Model (ABC) y SklearnModel (wrapper para Lasso/RF/GB)
│   │   └── model_optimizer.py            # ModelOptimizer: versión previa, un único split fijo (ver nota)
│   └── evaluation/
│       ├── indicator_registry.py         # Registro único de familias de indicadores (fuente de verdad)
│       ├── thresholds.py                 # ThresholdEvaluator: cálculo de umbrales óptimos y Profit Factor
│       ├── mcpt.py                       # MonteCarloPT: test de permutación de Monte Carlo
│       ├── walkforward.py                # WalkForwardEvaluator: splits y evaluación por fold
│       ├── optuna_walkforward.py         # OptunaWalkForward (versión legacy, selección por columna)
│       ├── optuna_walkforward_fixed.py   # OptunaWalkForwardFixed (versión actual, selección jerárquica)
│       ├── feature_analyzer.py           # FeatureAnalyzer: correlación de features train vs test
│       ├── optuna_plots.py               # Gráficas de diagnóstico de Optuna (para ModelOptimizer)
│       └── visualizer_walkforward.py     # VisualizerWalkForward: gráficas de resultados (Plotly)
└── tests/                             # Scripts de ejecución (punto de entrada de cada experimento)
    ├── generate_indicators_from_registry.py
    ├── test_indicator_selector.py
    ├── test_normalize.py
    ├── test_walkforward_indicators
    ├── test_walkforward_models.py
    ├── test_model_optimizer
    ├── test_optuna_walkforward
    └── test_optuna_fixed
```

> **Nota sobre `OptunaWalkForward` vs `OptunaWalkForwardFixed`:** el proyecto conserva ambas clases. `OptunaWalkForward` es la versión inicial, donde Optuna elegía columnas de indicadores precalculadas por índice (cada parametrización se trataba como una variable independiente). `OptunaWalkForwardFixed` es la versión corregida y la que se usa en los resultados finales del TFG: la búsqueda es jerárquica (familia → parámetros), corrige un bug de propagación de `family_params` en trials fallidos, y filtra filas con `NaN` antes de entrenar. **Usa siempre `OptunaWalkForwardFixed` para trabajo nuevo.**

> **Nota sobre `ModelOptimizer` vs `OptunaWalkForwardFixed`:** `ModelOptimizer` busca una única combinación de features/hiperparámetros fija para todo el histórico (evaluada con `WalkForwardEvaluator` de forma agregada). `OptunaWalkForwardFixed` reoptimiza en cada fold de forma independiente. Este segundo enfoque es el empleado en los Experimentos 2 y 3 de la memoria.

**Importante:** no hay archivos `__init__.py` en `src/` — los imports tipo `from src.evaluation... import ...` funcionan como *namespace packages* de Python 3 siempre que ejecutes los scripts **desde la raíz del proyecto** (los scripts de `tests/` añaden la raíz a `sys.path` automáticamente vía `sys.path.append(str(Path(__file__).resolve().parent.parent))`, pero la ruta de los CSV en `pd.read_csv("data/...")` es relativa al directorio de trabajo actual, no al script).

## Instalación

```bash
pip install -r requirements.txt
```

**TA-Lib requiere una librería C instalada aparte del paquete de Python.** Si `pip install TA-Lib` falla, instala primero la librería nativa:

- **Windows:** descarga el wheel precompilado correspondiente a tu versión de Python desde un repositorio de wheels no oficiales e instálalo con `pip install <wheel_descargado>.whl`.
- **Linux (Ubuntu/Debian):** `sudo apt-get install ta-lib` o compila desde el código fuente oficial antes de `pip install TA-Lib`.
- **macOS:** `brew install ta-lib` antes de `pip install TA-Lib`.

Ver la sección [Revisión de requirements.txt](#revisión-de-requirementstxt) más abajo para más detalles.

## 1. Preparar los datos

Todos los experimentos parten de un CSV OHLCV crudo (`timestamp, open, high, low, close, volume`) descargado de Binance, guardado en `data/raw/<resolución>/`. No hace falta precalcular ningún indicador a mano — con `OptunaWalkForwardFixed` se generan dinámicamente.

```python
import pandas as pd
from src.eda.eda import prepare_data
from src.features.targets import NormalizedFutureReturn

df = pd.read_csv("data/raw/1d/BTCUSDT_1d_01-01-2016_18-01-2026.csv", sep=",")
df = prepare_data(df)

target = NormalizedFutureReturn(df, window=14)
target.compute()
df = pd.concat([df, target.result], axis=1)
df = df.dropna(subset=["open", "high", "low", "close", "volume", "nfr_1_atr_14"])
```

Si en algún momento necesitas un CSV con **todas** las combinaciones de indicadores precalculadas (por ejemplo, para inspección manual o para scripts antiguos como `test_optuna_walkforward` que sí esperan columnas ya existentes), usa:

```bash
python tests/generate_indicators_from_registry.py
```

Este script recorre `INDICATOR_REGISTRY` y genera una columna por cada combinación de parámetros de cada familia, garantizando que los nombres coincidan exactamente con los que generaría `OptunaWalkForwardFixed` en tiempo real.

## 2. Experimento 1 — Grid search de indicadores individuales

`tests/test_indicator_selector.py`:

```python
from src.features.indicator_selector import IndicatorSelector
from src.eda.eda import prepare_data
import pandas as pd

df = pd.read_csv("data/raw/4h/ETHUSDT_4h_01-01-2016_18-01-2026.csv", sep=",")
df = prepare_data(df).dropna()

sel = IndicatorSelector(
    df,
    train_window=4380,
    test_window=1080,
    min_kept=438,     # mínimo de operaciones exigidas en train
    n_mcpt=100,       # nº de permutaciones del test MCPT
    p_threshold=0.05,
)

df_results = sel.run()
print(sel.top_n(10))

sel.plot_summary(top_n=20).show()
sel.plot_significant().show()
```

Para comparar la estabilidad temporal de los indicadores entre dos rangos de fechas (por ejemplo, 2017-2023 vs 2023-2026), usa `plot_period_correlation` del mismo módulo, pasándole dos instancias de `IndicatorSelector` ya ejecutadas (`run()` llamado sobre cada rango por separado).

## 3. Experimentos 2 y 3 — Modelos + Optuna + walk-forward jerárquico

`tests/test_optuna_fixed`:

```python
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.models.models import SklearnModel
from src.evaluation.optuna_walkforward_fixed import OptunaWalkForwardFixed
from src.evaluation.visualizer_walkforward import VisualizerWalkForward
from src.evaluation.indicator_registry import INDICATOR_REGISTRY

def build_lasso(params):
    return SklearnModel(Lasso(**params, random_state=42, max_iter=5000), name="Lasso")

def build_random_forest(params):
    return SklearnModel(RandomForestRegressor(**params, random_state=42, n_jobs=-1), name="Random Forest")

def build_gradient_boosting(params):
    return SklearnModel(GradientBoostingRegressor(**params, random_state=42), name="Gradient Boosting")

param_space_lasso = {"alpha": ("loguniform", 1e-4, 1.0)}

df = pd.read_csv("data/processed/BTCUSDT_1d_01-01-2016_18-01-2026.csv", index_col=0)
target = "nfr_1_atr_14"
df = df.dropna(subset=["open", "high", "low", "close", "volume", target])

owf = OptunaWalkForwardFixed(
    data=df,
    model_builder=build_lasso,       # cambia aquí para probar RF o GB
    target=target,
    param_space=param_space_lasso,
    train_window=730,
    test_window=180,
    min_families=3,
    max_families=5,
    min_kept=73,                     # ~10% del train_window
    n_trials_per_fold=30,
    mcpt=True,
    n_mcpt=200,
)

fold_results = owf.run()
owf.summary()

viz = VisualizerWalkForward(df, fold_results, indicator_registry=INDICATOR_REGISTRY)
viz.plot().show()                     # precio + señales + equity curve
viz.plot_walkforward_metrics().show() # thresholds y PF por fold
viz.plot_pf_evolution().show()        # evolución del PF y p-value por fold
viz.plot_error_metrics().show()       # MSE/RMSE train vs test, con baseline
```

## 4. (Legacy) `ModelOptimizer` — una única combinación para todo el histórico

`tests/test_model_optimizer` muestra el enfoque anterior a `OptunaWalkForwardFixed`: busca **una sola** combinación de features/hiperparámetros óptima para todo el periodo, en vez de reoptimizar por fold. Útil como comparación, pero no es el método principal del TFG.

```python
from src.models.model_optimizer import ModelOptimizer
from src.evaluation.optuna_plots import show_all
```

`optuna_plots.py` da gráficas de diagnóstico de la propia búsqueda de Optuna (historial de optimización, importancia de hiperparámetros, coordenadas paralelas) — pensadas para este optimizador de una sola pasada, no para el walk-forward jerárquico.

## 5. Cómo interpretar los resultados

| Métrica | Qué mirar |
|---|---|
| `MSE test` vs `MSE baseline test` | Si `MSE test ≥ MSE baseline test`, el modelo no predice mejor que "siempre la media" |
| `p_value_high` / `p_value_low` (MCPT) | Si `p > 0.05`, el Profit Factor observado no se distingue del azar |
| `Folds PF > 1` | Muchos folds positivos sin p-value bajo es compatible con sesgo direccional del mercado, no con edge genuino |
| Frecuencia de familias por fold (`summary()`) | Si los parámetros concretos varían mucho entre folds aunque la familia se repita, es indicio de ruido/inestabilidad |
