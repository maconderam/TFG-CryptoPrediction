"""Registro único de familias de indicadores técnicos.

Este módulo es la ÚNICA fuente de verdad sobre qué indicadores existen,
qué clase los implementa y qué valores de parámetros son candidatos
válidos para cada uno. Se importa desde:

  - IndicatorSelector (grid search clásico de indicadores individuales)
  - OptunaWalkForwardFixed (selección jerárquica familia -> parámetros)
  - VisualizerWalkForward (reconstrucción de features por fold)
  - Scripts de generación de CSVs con indicadores precalculados

Mantener el registro en un único sitio evita que estos componentes se
desincronicen entre sí (por ejemplo, que Optuna pueda elegir una
combinación de parámetros que el script de generación de CSV no haya
precalculado, o que el visualizador no sepa reconstruir).
"""

from src.features.indicators import (
    RSI, Stochastic, StochasticRSI, MACD, PriceIntensity,
    ADX, Aroon, AroonOscillator, ATR, PriceChangeOscillator, CMMA, MADifference,
    PriceVarianceRatio, ChangeVarianceRatio, IntradayIntensity, ChaikinMoneyFlow, OBV,
)

# ---------------------------------------------------------------------------
# Registro de familias de indicadores: familia -> {"cls": clase, "params": grid}
#
# "params" es un dict {nombre_parametro: [valores candidatos]}. El grid
# completo (todas las combinaciones vía itertools.product) es lo que usa
# IndicatorSelector y los scripts de generación de CSV. OptunaWalkForwardFixed
# usa este mismo grid como espacio de búsqueda categórico (suggest_categorical
# sobre cada lista de valores), en vez de recorrerlo exhaustivamente.
# ---------------------------------------------------------------------------
INDICATOR_REGISTRY = {
    "rsi": {
        "cls": RSI,
        "params": {
            "window":        [7, 10, 14, 20, 30, 50],
            "smooth_window": [2, 3, 5],
        },
    },
    "stochastic": {
        "cls": Stochastic,
        "params": {
            "window":        [7, 10, 14, 20, 30, 50],
            "smooth_window": [2, 3, 5],
        },
    },
    "stochastic_rsi": {
        "cls": StochasticRSI,
        "params": {
            "rsi_window":    [10, 14, 20, 30],
            "stoch_window":  [3, 5, 10],
            "smooth_window": [2, 3],
        },
    },
    "macd": {
        "cls": MACD,
        "params": {
            "short_length":  [8, 12, 16],
            "long_length":   [20, 26, 35, 50],
            "smooth_window": [9],
        },
    },
    "price_intensity": {
        "cls": PriceIntensity,
        "params": {
            "smooth_window": [5, 10, 14, 20, 30, 50],
        },
    },
    "adx": {
        "cls": ADX,
        "params": {
            "window": [7, 10, 14, 20, 30, 50],
        },
    },
    "aroon": {
        "cls": Aroon,
        "params": {
            "window": [10, 14, 20, 30, 50, 100],
        },
    },
    "aroon_osc": {
        "cls": AroonOscillator,
        "params": {
            "window": [10, 14, 20, 30, 50, 100],
        },
    },
    "atr": {
        "cls": ATR,
        "params": {
            "window": [7, 14, 21, 30, 50],
        },
    },
    "price_change_osc": {
        "cls": PriceChangeOscillator,
        "params": {
            "short_length": [5, 10, 20],
            "mult":         [2, 3, 5],
        },
    },
    "price_variance_ratio": {
        "cls": PriceVarianceRatio,
        "params": {
            "short_length": [5, 10, 20],
            "mult":         [2, 4, 6],
        },
    },
    "change_variance_ratio": {
        "cls": ChangeVarianceRatio,
        "params": {
            "short_length": [5, 10, 20],
            "mult":         [2, 4, 6],
        },
    },
    "cmma": {
        "cls": CMMA,
        "params": {
            "window":     [5, 10, 20, 50],
            "atr_window": [14, 60, 252],
            "c":          [1.0],
        },
    },
    "madifference": {
        "cls": MADifference,
        "params": {
            "short_length": [5, 10, 20],
            "long_length":  [50, 100, 150],
            "lag":          [0],
        },
    },
    "intraday_intensity": {
        "cls": IntradayIntensity,
        "params": {
            "window":        [7, 14, 21, 30],
            "smooth_window": [1, 5, 10],
        },
    },
    "chaikin_money_flow": {
        "cls": ChaikinMoneyFlow,
        "params": {
            "window": [7, 10, 14, 21, 30, 50],
        },
    },
    "obv": {
        "cls": OBV,
        "params": {
            "window":     [5, 10, 14, 20, 30],
            "atr_window": [14, 30],
        },
    },
}


def get_indicator_classes() -> dict:
    """Devuelve {familia: clase}, útil donde solo se necesite la clase
    (p.ej. compatibilidad con código antiguo que usaba _INDICATOR_CLASSES)."""
    return {name: spec["cls"] for name, spec in INDICATOR_REGISTRY.items()}


def get_default_grids() -> dict:
    """Devuelve {familia: grid_de_parametros}, útil donde solo se necesite
    el grid (p.ej. compatibilidad con código antiguo que usaba DEFAULT_GRIDS)."""
    return {name: dict(spec["params"]) for name, spec in INDICATOR_REGISTRY.items()}