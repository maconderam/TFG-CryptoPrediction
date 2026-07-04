import itertools
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

from src.evaluation.mcpt import MonteCarloPT
from src.evaluation.walkforward import WalkForwardEvaluator
from src.evaluation.indicator_registry import INDICATOR_REGISTRY, get_default_grids


# ---------------------------------------------------------------------------
# IndicatorSelector
#
# Los grids de parámetros y las clases de indicador ya NO se definen aquí:
# se importan de src.evaluation.indicator_registry, que es la única fuente
# de verdad compartida también por OptunaWalkForwardFixed y
# VisualizerWalkForward. Así evitamos que este selector, la búsqueda de
# Optuna y los scripts de generación de CSV se desincronicen entre sí.
# ---------------------------------------------------------------------------
class IndicatorSelector:
    def __init__(
        self,
        data: pd.DataFrame,
        train_window: int = 500,
        test_window: int = 100,
        step: int = None,
        expanding: bool = False,
        min_kept: int = 300,
        n_mcpt: int = 200,
        p_threshold: float = 0.10,
        custom_grids: Optional[dict] = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.data = data.copy()
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.expanding = expanding
        self.min_kept = min_kept
        self.n_mcpt = n_mcpt
        self.p_threshold = p_threshold
        self.seed = seed
        self.verbose = verbose
        self.wf = WalkForwardEvaluator(self.data, 
                                  train_window=self.train_window, 
                                  test_window=self.test_window, 
                                  step=self.step, 
                                  expanding=self.expanding)

        # Merge del registro compartido con cualquier override del usuario
        self.grids = {k: dict(v) for k, v in get_default_grids().items()}
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
        cls = INDICATOR_REGISTRY[name]["cls"]
        return cls(self.data, **params)

    def _evaluate_signal(self, signal: pd.Series, name: str) -> dict:
            """Ejecuta el esquema Walk-Forward (con MCPT interno por fold) sobre 
            una columna de señal y devuelve las métricas consolidadas (medias).
            """
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Suprimir salida por consola si no estamos en modo verbose
                if not self.verbose:
                    import io, sys
                    _stdout = sys.stdout
                    sys.stdout = io.StringIO()

                # Delegamos la ejecución completa al WalkForwardEvaluator integrado
                folds_res = self.wf.run_indicator(
                    signal=signal,
                    mcpt=True,           # Activamos el análisis Monte Carlo por fold
                    n_mcpt=self.n_mcpt,
                    min_kept=self.min_kept,
                    seed=self.seed
                )

                if not self.verbose:
                    sys.stdout = _stdout

            # Si el validador no arrojó resultados, devolvemos un dict vacío con NaNs
            if not folds_res:
                return {
                    "best_pf": np.nan, "pf_high": np.nan, "pf_low": np.nan,
                    "high_thresh": np.nan, "low_thresh": np.nan, "p_value": np.nan,
                    "mc_mean_score": np.nan, "mc_std_score": np.nan,
                    "mc_distribution": [], "signal_name": name, "folds_history": []
                }

            # Convertimos la lista de folds a DataFrame para calcular los promedios ponderados/medios
            df_folds = pd.DataFrame(folds_res)

            # Extraemos los p-values si el MCPT estuvo activo en la simulación
            p_values_hi = df_folds["p_value_high"].dropna() if "p_value_high" in df_folds.columns else pd.Series([np.nan])
            
            # --- Mapeo de Métricas Agregadas ---
            # Usamos la media de los folds de TEST como el rendimiento real del sistema
            mean_test_long_above = df_folds["pf_test_long_above"].mean()
            mean_test_short_above = df_folds["pf_test_short_above"].mean() 
            mean_test_long_below = df_folds["pf_test_long_below"].mean() 
            mean_test_short_below = df_folds["pf_test_short_below"].mean() 
            mean_pf_high    = df_folds["pf_train_high"].mean()
            mean_pf_low     = df_folds["pf_train_low"].mean()
            mean_high_th    = df_folds["high_thresh"].mean()
            mean_low_th     = df_folds["low_thresh"].mean()
            mean_p_value    = p_values_hi.mean()

            # Determinamos el mejor rendimiento obtenido fuera de muestra (Test)
            best_test_pf = max(mean_test_long_above, mean_test_short_above, mean_test_long_below, mean_test_short_below)

            return {
                "best_pf":               best_test_pf,  # Mejor Profit Factor fuera de muestra
                "mean_test_long_above":  mean_test_long_above,
                "mean_test_short_above": mean_test_short_above,
                "mean_test_long_below":  mean_test_long_below,
                "mean_test_short_below": mean_test_short_below,
                "pf_high":               mean_pf_high,
                "pf_low":                mean_pf_low,
                "high_thresh":           mean_high_th,
                "low_thresh":            mean_low_th,
                "p_value":               mean_p_value,
                "mc_mean_score":         np.nan,
                "mc_std_score":          np.nan,
                "mc_distribution":       p_values_hi.tolist(),
                "signal_name":           name,
                "folds_history":         folds_res,
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
            if ind_name not in INDICATOR_REGISTRY:
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

        df["significant"] = df["p_value"] <= self.p_threshold

        df = df.sort_values("best_pf", ascending=False).reset_index(drop=True)
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
            "best_pf", "pf_high", "pf_low",
            "high_thresh", "low_thresh",
            "p_value", "mc_mean_score", "mc_std_score",
            "significant",
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
        """Genera un dashboard de 3 paneles con los resultados del grid search.

        Args:
            top_n: Número de señales a mostrar en los paneles de barras.
            only_significant: Si True, filtra solo señales con p_value <= p_threshold.

        Returns:
            Figura de Plotly con 3 subplots: Profit Factor, P-value,
            scatter PF vs P-value.
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
                f"PF: {row['best_pf']:.3f}<br>"
                f"p-value: {row['p_value']:.3f}"
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
            rows=1, cols=3,
            subplot_titles=(
                "Profit Factor (mejor valor)",
                "MCPT P-value",
                "PF vs P-value (todas las señales)",
            ),
            horizontal_spacing=0.12,
        )

        # --- Panel 1: Profit Factor ---
        fig.add_trace(go.Bar(
            x=labels, y=df["best_pf"],
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
                x=nsig["p_value"], y=nsig["best_pf"],
                mode="markers", name=f"{fam} (n.s.)",
                marker=dict(color=colour_map[fam], size=7, opacity=0.4),
                hovertext=nsig["hover"], hoverinfo="text",
                legendgroup=fam,
            ), row=1, col=3)

            fig.add_trace(go.Scatter(
                x=sig["p_value"], y=sig["best_pf"],
                mode="markers", name=f"{fam} (sig.)",
                marker=dict(color=colour_map[fam], size=11, symbol="star",
                            line=dict(width=1, color="#333333")),
                hovertext=sig["hover"], hoverinfo="text",
                legendgroup=fam,
            ), row=1, col=3)

        fig.add_vline(x=self.p_threshold, line=dict(color="#333333", width=1, dash="dash"), row=1, col=3)
        fig.update_xaxes(title_text="P-value", row=1, col=3)
        fig.update_yaxes(title_text="Profit Factor", row=1, col=3)
        fig.update_yaxes(title_text="P-value", row=1, col=2)
        fig.update_yaxes(title_text="Profit Factor", row=1, col=1)

        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)

        title_suffix = " (solo significativos)" if only_significant else ""
        fig.update_layout(
            title=f"Indicator Grid Search — Top {top_n}{title_suffix}",
            template="plotly_white",
            height=550,
            width=1500,
            showlegend=True,
            legend=dict(orientation="h", y=-0.3, font=dict(size=9)),
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
            top_n: Número de señales a mostrar (según ranking de best_pf).
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
            real = row["best_pf"]
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

    def plot_significant(self, only_significant: bool = True) -> go.Figure:
        """Genera un gráfico de barras con los indicadores que superan el
        umbral de significancia (p_value <= p_threshold), mostrando su
        Profit Factor asociado.

        Args:
            only_significant: Si True (por defecto), filtra solo señales con
                p_value <= p_threshold. Si False, muestra todas las señales
                pero mantiene el mismo formato visual.

        Returns:
            Figura de Plotly con un gráfico de barras de Profit Factor,
            coloreado por familia de indicador.
        """
        if self.summary_df is None:
            raise RuntimeError("Call run() first.")

        df = self.summary_df.copy()
        if only_significant:
            df = df[df["significant"]]
        df = df.sort_values("best_pf", ascending=False).reset_index(drop=True)

        if df.empty:
            print("No hay señales significativas para graficar.")
            return None

        param_cols = [c for c in df.columns if c.startswith("param_")]

        def hover_text(row):
            params_str = "<br>".join(f"{c.replace('param_', '')}: {row[c]}" for c in param_cols)
            return (
                f"<b>{row['signal']}</b><br>"
                f"Indicator: {row['indicator']}<br>"
                f"{params_str}<br>"
                f"PF: {row['best_pf']:.3f}<br>"
                f"p-value: {row['p_value']:.3f}"
            )

        df["hover"] = df.apply(hover_text, axis=1)
        labels = df["signal"].tolist()

        # Colour map per indicator family (igual que en plot_summary)
        families = df["indicator"].unique()
        palette = [
            "#5B8CFF", "#00C896", "#FF4C6A", "#FFD166", "#B07FFF",
            "#FF8C42", "#4ECDC4", "#F7B801", "#A8DADC", "#E76F51",
            "#06D6A0", "#EF476F", "#118AB2", "#073B4C",
        ]
        colour_map = {fam: palette[i % len(palette)] for i, fam in enumerate(families)}
        bar_colours = df["indicator"].map(colour_map)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=labels, y=df["best_pf"],
            marker_color=bar_colours,
            hovertext=df["hover"], hoverinfo="text",
            showlegend=False,
        ))

        fig.add_hline(
            y=1.0, line=dict(color="#333333", width=1, dash="dash"),
            annotation_text="PF = 1",
        )

        # Trazas "fantasma" para poder mostrar leyenda por familia de indicador
        for fam in families:
            fig.add_trace(go.Bar(
                x=[None], y=[None],
                marker_color=colour_map[fam],
                name=fam,
                showlegend=True,
            ))

        title_suffix = f" (p-value ≤ {self.p_threshold})" if only_significant else " (todas las señales)"
        fig.update_layout(
            title=f"Indicadores Significativos — Profit Factor{title_suffix}",
            template="plotly_white",
            xaxis_title="Señal",
            yaxis_title="Profit Factor",
            height=550,
            width=max(900, 40 * len(labels)),
            legend=dict(orientation="h", y=-0.3, font=dict(size=9)),
        )
        fig.update_xaxes(tickangle=45)

        return fig

def plot_period_correlation(
    sel_2017_2023: "IndicatorSelector",
    sel_2023_2026: "IndicatorSelector",
    only_significant_both: bool = False,
) -> go.Figure:
    """Compara el Profit Factor (best_pf) de cada señal entre dos periodos
    temporales distintos (2017-2023 y 2023-2026), para detectar si el
    rendimiento de los indicadores es estable frente a cambios de régimen
    de mercado o si depende del intervalo concreto analizado.
 
    Importante: pese al nombre habitual de "train/test" en otros análisis
    de este proyecto, aquí NO se trata de un esquema de entrenamiento y
    evaluación de un modelo, sino de dos particiones cronológicas
    independientes del mismo walk-forward (cada una evaluada con su propio
    IndicatorSelector). El objetivo es comprobar la persistencia temporal
    del rendimiento de cada indicador entre regímenes de mercado distintos,
    no medir generalización de un modelo entrenado.
 
    Args:
        sel_2017_2023: IndicatorSelector ya ejecutado (run()) sobre el
            periodo 2017-2023.
        sel_2023_2026: IndicatorSelector ya ejecutado (run()) sobre el
            periodo 2023-2026.
        only_significant_both: Si True, solo incluye señales que fueron
            significativas (p_value <= p_threshold) en ambos periodos.
 
    Returns:
        Figura de Plotly con scatter PF 2017-2023 vs PF 2023-2026 y línea
        diagonal roja de referencia (PF idéntico en ambos periodos).
    """
    if sel_2017_2023.summary_df is None or sel_2023_2026.summary_df is None:
        raise RuntimeError("Ambos selectores deben haber ejecutado run() primero.")
 
    df_p1 = sel_2017_2023.summary_df[[
        "indicator", "signal", "best_pf", "p_value", "significant"
    ]].rename(columns={
        "best_pf": "pf_2017_2023", "p_value": "p_value_2017_2023", "significant": "sig_2017_2023"
    })
 
    df_p2 = sel_2023_2026.summary_df[[
        "signal", "best_pf", "p_value", "significant"
    ]].rename(columns={
        "best_pf": "pf_2023_2026", "p_value": "p_value_2023_2026", "significant": "sig_2023_2026"
    })
 
    # Merge por nombre de señal (indicador + parámetros codificados en el nombre de columna)
    df = df_p1.merge(df_p2, on="signal", how="inner")
 
    if df.empty:
        print("No hay señales comunes entre ambos periodos para comparar.")
        return None
 
    if only_significant_both:
        df = df[df["sig_2017_2023"] & df["sig_2023_2026"]]
        if df.empty:
            print("No hay señales significativas en ambos periodos.")
            return None
 
    def hover_text(row):
        return (
            f"<b>{row['signal']}</b><br>"
            f"Indicator: {row['indicator']}<br>"
            f"PF 2017-2023: {row['pf_2017_2023']:.3f} (p={row['p_value_2017_2023']:.3f})<br>"
            f"PF 2023-2026: {row['pf_2023_2026']:.3f} (p={row['p_value_2023_2026']:.3f})"
        )
 
    df = df.copy()
    df["hover"] = df.apply(hover_text, axis=1)
 
    families = df["indicator"].unique()
    palette = [
        "#5B8CFF", "#00C896", "#FF4C6A", "#FFD166", "#B07FFF",
        "#FF8C42", "#4ECDC4", "#F7B801", "#A8DADC", "#E76F51",
        "#06D6A0", "#EF476F", "#118AB2", "#073B4C",
    ]
    colour_map = {fam: palette[i % len(palette)] for i, fam in enumerate(families)}
 
    fig = go.Figure()
 
    for fam in families:
        sub = df[df["indicator"] == fam]
        both_sig = sub["sig_2017_2023"] & sub["sig_2023_2026"]
 
        # No significativas en ambos -> puntos pequeños y opacos
        sub_ns = sub[~both_sig]
        fig.add_trace(go.Scatter(
            x=sub_ns["pf_2017_2023"], y=sub_ns["pf_2023_2026"],
            mode="markers", name=f"{fam}",
            marker=dict(color=colour_map[fam], size=8, opacity=0.45),
            hovertext=sub_ns["hover"], hoverinfo="text",
            legendgroup=fam,
        ))
 
        # Significativas en ambos -> estrella destacada
        sub_s = sub[both_sig]
        fig.add_trace(go.Scatter(
            x=sub_s["pf_2017_2023"], y=sub_s["pf_2023_2026"],
            mode="markers", name=f"{fam} (sig. en ambos periodos)",
            marker=dict(color=colour_map[fam], size=13, symbol="star",
                        line=dict(width=1, color="#333333")),
            hovertext=sub_s["hover"], hoverinfo="text",
            legendgroup=fam,
        ))
 
# --- Línea diagonal roja (PF idéntico en ambos periodos): y = x ---
    axis_lo = 0.5   # ajusta estos dos valores al rango que quieras mostrar
    axis_hi = 3.0

    fig.add_trace(go.Scatter(
        x=[axis_lo, axis_hi], y=[axis_lo, axis_hi],
        mode="lines",
        line=dict(color="red", width=2, dash="dash"),
        name="PF 2017-2023 = PF 2023-2026",
        showlegend=True,
    ))

    corr = df["pf_2017_2023"].replace([np.inf, -np.inf], np.nan).corr(
        df["pf_2023_2026"].replace([np.inf, -np.inf], np.nan)
    )

    fig.update_layout(
        title=f"Correlación Profit Factor 2017-2023 vs 2023-2026 (r = {corr:.3f})",
        template="plotly_white",
        xaxis_title="Profit Factor (2017-2023)",
        yaxis_title="Profit Factor (2023-2026)",
        height=650,
        width=800,
        showlegend=True,
        legend=dict(font=dict(size=9)),
    )
    fig.update_xaxes(range=[axis_lo, axis_hi])
    fig.update_yaxes(range=[axis_lo, axis_hi], scaleanchor="x", scaleratio=1)
 
    return fig