import itertools
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

from .indicators import (
    RSI, Stochastic, StochasticRSI, MACD, PriceIntensity,
    ADX, Aroon, AroonOscillator, ATR, PriceChangeOscillator,
    PriceVarianceRatio, ChangeVarianceRatio, CMMA, MADifference,
    IntradayIntensity, ChaikinMoneyFlow, OBV,
)
from src.evaluacion.mcpt import MonteCarloPT

# ---------------------------------------------------------------------------
# Default parameter grids for each indicator class
#
# Cada grid se mantiene entre 8 y 16 combinaciones aproximadamente para no
# disparar el tiempo total de ejecución del grid search (cada combinación
# implica un MCPT completo con n_mcpt permutaciones).
# ---------------------------------------------------------------------------
DEFAULT_GRIDS = {
    "RSI": {
        "window":        [7, 10, 14, 20, 30, 50],
        "smooth_window": [2, 3, 5],
    },
    "Stochastic": {
        "window":        [7, 10, 14, 20, 30, 50],
        "smooth_window": [2, 3, 5],
    },
    "StochasticRSI": {
        "rsi_window":    [10, 14, 20, 30],
        "stoch_window":  [3, 5, 10],
        "smooth_window": [2, 3],
    },
    "MACD": {
        "short_length":  [8, 12, 16],
        "long_length":   [20, 26, 35, 50],
        "smooth_window": [9],
    },
    "PriceIntensity": {
        "smooth_window": [5, 10, 14, 20, 30, 50],
    },
    "ADX": {
        "window": [7, 10, 14, 20, 30, 50],
    },
    "Aroon": {
        "window": [10, 14, 20, 30, 50, 100],
    },
    "AroonOscillator": {
        "window": [10, 14, 20, 30, 50, 100],
    },
    "PriceChangeOscillator": {
        "short_length": [5, 10, 20],
        "mult":         [2, 3, 5],
    },
    "PriceVarianceRatio": {
        "short_length": [5, 10, 20],
        "mult":         [2, 4, 6],
    },
    "ChangeVarianceRatio": {
        "short_length": [5, 10, 20],
        "mult":         [2, 4, 6],
    },
    "CMMA": {
        "window":     [5, 10, 20, 50],
        "atr_window": [14, 60, 252],
        "c":          [1.0],
    },
    "MADifference": {
        "short_length": [5, 10, 20],
        "long_length":  [50, 100, 150],
        "lag":          [0],
    },
    "IntradayIntensity": {
        "window":        [7, 14, 21, 30],
        "smooth_window": [1, 5, 10],
    },
    "ChaikinMoneyFlow": {
        "window": [7, 10, 14, 21, 30, 50],
    },
    "OBV": {
        "window":     [5, 10, 14, 20, 30],
        "atr_window": [14, 30],
    },
}

# Map name → class
_INDICATOR_CLASSES = {
    "RSI": RSI, "Stochastic": Stochastic, "StochasticRSI": StochasticRSI,
    "MACD": MACD, "PriceIntensity": PriceIntensity, "ADX": ADX,
    "Aroon": Aroon, "AroonOscillator": AroonOscillator,
    "PriceChangeOscillator": PriceChangeOscillator, "PriceVarianceRatio": PriceVarianceRatio,
    "ChangeVarianceRatio": ChangeVarianceRatio, "CMMA": CMMA,
    "MADifference": MADifference, "IntradayIntensity": IntradayIntensity,
    "ChaikinMoneyFlow": ChaikinMoneyFlow, "OBV": OBV,
}


# ---------------------------------------------------------------------------
# IndicatorSelector
# ---------------------------------------------------------------------------
class IndicatorSelector:
    def __init__(
        self,
        data: pd.DataFrame,
        target=None,
        min_kepts: int = 300,
        n_mcpt: int = 200,
        p_threshold: float = 0.10,
        custom_grids: Optional[dict] = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.data = data.copy()
        self.target = target
        self.min_kepts = min_kepts
        self.n_mcpt = n_mcpt
        self.p_threshold = p_threshold
        self.seed = seed
        self.verbose = verbose

        # Merge default grids with any user overrides
        self.grids = {k: dict(v) for k, v in DEFAULT_GRIDS.items()}
        if custom_grids:
            for name, params in custom_grids.items():
                if name in self.grids:
                    self.grids[name].update(params)
                else:
                    self.grids[name] = params

        self.results: list[dict] = []
        self.summary_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _param_combinations(grid: dict) -> list[dict]:
        """Return all combinations of a parameter grid as a list of dicts."""
        keys = list(grid.keys())
        values = list(grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _build_indicator(self, name: str, params: dict):
        """Instantiate an indicator by name with given params."""
        cls = _INDICATOR_CLASSES[name]
        return cls(self.data, **params)

    def _evaluate_signal(self, signal: pd.Series, name: str) -> dict:
        """Run MCPT on a single signal column and return metrics dict."""
        mcpt = MonteCarloPT(self.data, seed=self.seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Suppress internal print output unless verbose
            if not self.verbose:
                import io, sys
                _stdout = sys.stdout
                sys.stdout = io.StringIO()

            res = mcpt.mcpt_threshold(signal, n_test=self.n_mcpt, min_kept=self.min_kepts)

            if not self.verbose:
                sys.stdout = _stdout

        return {
            "real_score":        res["real_score"],
            "pf_high":           res["real"]["pf_high"],
            "pf_low":            res["real"]["pf_low"],
            "high_thresh":       res["real"]["high_thresh"],
            "low_thresh":        res["real"]["low_thresh"],
            "p_value":           res["p_value"],
            "mc_mean_score":     res["mc_mean_score"],
            "mc_std_score":      res["mc_std_score"],
            "mc_distribution":   res["mc_distribution"],
            "signal_name":       name,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, indicators: Optional[list[str]] = None) -> pd.DataFrame:
        if indicators is None:
            indicators = list(self.grids.keys())

        self.results = []
        total = sum(
            len(self._param_combinations(self.grids[ind]))
            for ind in indicators
            if ind in self.grids
        )
        done = 0

        for ind_name in indicators:
            if ind_name not in _INDICATOR_CLASSES:
                warnings.warn(f"Unknown indicator '{ind_name}', skipping.")
                continue

            grid = self.grids.get(ind_name, {})
            combos = self._param_combinations(grid)

            for params in combos:
                done += 1
                try:
                    indicator = self._build_indicator(ind_name, params)
                    result_df = indicator.compute()
                except Exception as e:
                    warnings.warn(f"[{ind_name}] compute failed with {params}: {e}")
                    continue

                for col in result_df.columns:
                    signal = result_df[col].dropna()
                    if len(signal) < 100:
                        warnings.warn(
                            f"[{col}] too few observations ({len(signal)}), skipping."
                        )
                        continue

                    print(
                        f"[{done}/{total}] Evaluating {col} ...",
                        end="\r",
                        flush=True,
                    )

                    try:
                        metrics = self._evaluate_signal(
                            result_df[col].rename(col), col
                        )
                    except Exception as e:
                        warnings.warn(f"[{col}] MCPT failed: {e}")
                        continue

                    row = {
                        "indicator":    ind_name,
                        "signal":       col,
                        **{f"param_{k}": v for k, v in params.items()},
                        **{k: v for k, v in metrics.items()
                           if k != "mc_distribution"},
                        "_mc_dist":     metrics["mc_distribution"],
                    }
                    self.results.append(row)

        print()  # newline after \r progress

        if not self.results:
            print("No results collected.")
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # --- Composite ranking score -------------------------------------------
        # Normalise profit factor (higher = better) and p-value (lower = better)
        pf_min, pf_max = df["real_score"].min(), df["real_score"].max()
        pf_range = pf_max - pf_min if pf_max != pf_min else 1.0
        df["pf_norm"] = (df["real_score"] - pf_min) / pf_range

        pv_min, pv_max = df["p_value"].min(), df["p_value"].max()
        pv_range = pv_max - pv_min if pv_max != pv_min else 1.0
        df["pv_norm"] = 1 - (df["p_value"] - pv_min) / pv_range  # inverted

        df["composite_score"] = 0.6 * df["pf_norm"] + 0.4 * df["pv_norm"]
        df["significant"] = df["p_value"] <= self.p_threshold

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        self.summary_df = df
        return df

    def get_summary(self, only_significant: bool = False) -> pd.DataFrame:
        if self.summary_df is None:
            raise RuntimeError("Call run() first.")

        df = self.summary_df
        if only_significant:
            df = df[df["significant"]]

        display_cols = [
            "rank", "indicator", "signal",
            "real_score", "pf_high", "pf_low",
            "high_thresh", "low_thresh",
            "p_value", "mc_mean_score", "mc_std_score",
            "composite_score", "significant",
        ]
        param_cols = [c for c in df.columns if c.startswith("param_")]
        return df[display_cols + param_cols].copy()

    def top_n(self, n: int = 10, only_significant: bool = True) -> pd.DataFrame:
        return self.get_summary(only_significant=only_significant).head(n)


    # ------------------------------------------------------------------
    # Plotting (Plotly)
    # ------------------------------------------------------------------

    def plot_summary(
        self,
        top_n: int = 20,
        only_significant: bool = False,
    ) -> go.Figure:
        """Genera un dashboard de 4 paneles con los resultados del grid search.

        Args:
            top_n: Número de señales a mostrar en los paneles de barras.
            only_significant: Si True, filtra solo señales con p_value <= p_threshold.

        Returns:
            Figura de Plotly con 4 subplots: Profit Factor, P-value,
            scatter PF vs P-value, y Composite Score.
        """
        if self.summary_df is None:
            raise RuntimeError("Call run() first.")

        df = self.get_summary(only_significant=only_significant).head(top_n)

        if df.empty:
            print("No data to plot.")
            return None

        param_cols = [c for c in df.columns if c.startswith("param_")]

        def hover_text(row):
            params_str = "<br>".join(f"{c.replace('param_', '')}: {row[c]}" for c in param_cols)
            return (
                f"<b>{row['signal']}</b><br>"
                f"Indicator: {row['indicator']}<br>"
                f"{params_str}<br>"
                f"PF: {row['real_score']:.3f}<br>"
                f"p-value: {row['p_value']:.3f}<br>"
                f"Composite: {row['composite_score']:.3f}"
            )

        df = df.copy()
        df["hover"] = df.apply(hover_text, axis=1)

        labels = df["signal"].tolist()

        # Colour map per indicator family
        families = df["indicator"].unique()
        palette = [
            "#5B8CFF", "#00C896", "#FF4C6A", "#FFD166", "#B07FFF",
            "#FF8C42", "#4ECDC4", "#F7B801", "#A8DADC", "#E76F51",
            "#06D6A0", "#EF476F", "#118AB2", "#073B4C",
        ]
        colour_map = {fam: palette[i % len(palette)] for i, fam in enumerate(families)}
        bar_colours = df["indicator"].map(colour_map)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Profit Factor (real_score)",
                "MCPT P-value",
                "PF vs P-value (todas las señales)",
                "Composite Score",
            ),
            vertical_spacing=0.14,
            horizontal_spacing=0.10,
        )

        # --- Panel 1: Profit Factor ---
        fig.add_trace(go.Bar(
            x=labels, y=df["real_score"],
            marker_color=bar_colours,
            hovertext=df["hover"], hoverinfo="text",
            showlegend=False,
        ), row=1, col=1)
        fig.add_hline(y=1.0, line=dict(color="#333333", width=1, dash="dash"),
                      annotation_text="PF = 1", row=1, col=1)

        # --- Panel 2: P-value ---
        fig.add_trace(go.Bar(
            x=labels, y=df["p_value"],
            marker_color=bar_colours,
            hovertext=df["hover"], hoverinfo="text",
            showlegend=False,
        ), row=1, col=2)
        fig.add_hline(y=self.p_threshold, line=dict(color="#333333", width=1, dash="dash"),
                      annotation_text=f"p = {self.p_threshold}", row=1, col=2)
        fig.update_yaxes(range=[0, 1], row=1, col=2)

        # --- Panel 3: Scatter PF vs P-value (todas las señales, no solo top_n) ---
        full_df = self.get_summary(only_significant=False).copy()
        full_df["hover"] = full_df.apply(hover_text, axis=1)

        for fam in families:
            sub  = full_df[full_df["indicator"] == fam]
            sig  = sub[sub["significant"]]
            nsig = sub[~sub["significant"]]

            fig.add_trace(go.Scatter(
                x=nsig["p_value"], y=nsig["real_score"],
                mode="markers", name=f"{fam} (n.s.)",
                marker=dict(color=colour_map[fam], size=7, opacity=0.4),
                hovertext=nsig["hover"], hoverinfo="text",
                legendgroup=fam,
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=sig["p_value"], y=sig["real_score"],
                mode="markers", name=f"{fam} (sig.)",
                marker=dict(color=colour_map[fam], size=11, symbol="star",
                            line=dict(width=1, color="#333333")),
                hovertext=sig["hover"], hoverinfo="text",
                legendgroup=fam,
            ), row=2, col=1)

        fig.add_vline(x=self.p_threshold, line=dict(color="#333333", width=1, dash="dash"), row=2, col=1)
        fig.update_xaxes(title_text="P-value", row=2, col=1)
        fig.update_yaxes(title_text="Profit Factor", row=2, col=1)

        # --- Panel 4: Composite score ---
        fig.add_trace(go.Bar(
            x=labels, y=df["composite_score"],
            marker_color=bar_colours,
            hovertext=df["hover"], hoverinfo="text",
            showlegend=False,
        ), row=2, col=2)

        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=2)

        title_suffix = " (solo significativos)" if only_significant else ""
        fig.update_layout(
            title=f"Indicator Grid Search — Top {top_n}{title_suffix}",
            template="plotly_white",
            height=850,
            width=1300,
            showlegend=True,
            legend=dict(orientation="h", y=-0.25, font=dict(size=9)),
        )

        return fig

    def plot_mc_distributions(
        self,
        top_n: int = 6,
        only_significant: bool = False,
    ) -> go.Figure:
        """Genera histogramas de la distribución Monte Carlo para las mejores señales.

        Cada subplot muestra la distribución de scores obtenidos por
        permutación junto con el score real observado, para inspeccionar
        visualmente qué tan extremo es el resultado real frente al azar.

        Args:
            top_n: Número de señales a mostrar (según ranking de composite_score).
            only_significant: Si True, filtra solo señales con p_value <= p_threshold.

        Returns:
            Figura de Plotly con un histograma por señal.
        """
        if self.summary_df is None:
            raise RuntimeError("Call run() first.")

        df_top = self.summary_df.head(top_n)
        if only_significant:
            df_top = self.summary_df[self.summary_df["significant"]].head(top_n)

        n = len(df_top)
        if n == 0:
            print("No data to plot.")
            return None

        param_cols = [c for c in df_top.columns if c.startswith("param_")]
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=df_top["signal"].tolist(),
            vertical_spacing=0.15,
            horizontal_spacing=0.08,
        )

        for i, (_, row) in enumerate(df_top.iterrows()):
            r = i // ncols + 1
            c = i % ncols + 1

            dist = row["_mc_dist"]
            real = row["real_score"]
            pv   = row["p_value"]

            params_str = ", ".join(f"{p.replace('param_', '')}={row[p]}" for p in param_cols)
            sig_tag = "✓ significativo" if row["significant"] else "✗ no significativo"

            fig.add_trace(go.Histogram(
                x=dist, nbinsx=30,
                marker_color="#5B8CFF", opacity=0.75,
                name="MC scores",
                showlegend=(i == 0),
                hovertemplate="score: %{x}<br>count: %{y}<extra></extra>",
            ), row=r, col=c)

            fig.add_vline(
                x=real, line=dict(color="#FF4C6A", width=2.5),
                annotation_text=f"real: {real:.3f}",
                annotation_font=dict(size=9, color="#FF4C6A"),
                row=r, col=c,
            )

            fig.update_xaxes(
                title_text=f"p={pv:.3f} · {sig_tag}",
                title_font=dict(size=9),
                row=r, col=c,
            )

        title_suffix = " (solo significativos)" if only_significant else ""
        fig.update_layout(
            title=f"Distribuciones Monte Carlo{title_suffix}",
            template="plotly_white",
            height=320 * nrows,
            width=420 * ncols,
            showlegend=True,
            legend=dict(orientation="h", y=-0.05),
        )

        return fig