import copy
import numpy as np
import pandas as pd
import optuna
from optuna.logging import set_verbosity, WARNING

from src.models.models import Model
from .thresholds import ThresholdEvaluator
from .mcpt import MonteCarloPT

from src.evaluation.indicator_registry import INDICATOR_REGISTRY as DEFAULT_INDICATOR_REGISTRY


class OptunaWalkForwardFixed:
    """Walk-forward donde la selección de indicadores e hiperparámetros se
    re-optimiza de forma independiente en cada fold, usando solo el train
    de ese fold (sin ver nunca el test del fold, ni el de ningún otro).

    A diferencia de una selección de features por índice de columna (donde
    "rsi_14" y "rsi_20" serían dos variables independientes para Optuna),
    aquí la búsqueda es jerárquica: para cada familia de indicadores del
    registro, Optuna decide primero si la usa (True/False) y, si la usa,
    elige los parámetros concretos de esa familia (p.ej. window del RSI).
    Los indicadores se calculan dinámicamente en cada trial a partir de esa
    combinación familia→parámetros, en vez de seleccionarse de un conjunto
    de columnas precalculadas.

    Para evitar que Optuna vea el test real del fold durante la búsqueda,
    el train de cada fold se subdivide en:
      - inner_train: primer (1 - val_frac) % del train del fold.
      - inner_val:   último val_frac % del train del fold.

    Cada trial de Optuna se entrena con inner_train y se evalúa (mediante
    el profit factor del umbral óptimo) sobre inner_val. El trial ganador
    se reentrena con el train completo del fold y se evalúa, por fin, en
    el test real de ese fold.

    Args:
        data: DataFrame completo con OHLCV, target y close. Los indicadores
            se calculan a partir de este DataFrame, no de columnas precalculadas.
        model_builder: Función que recibe un dict de hiperparámetros y
            devuelve una instancia de Model lista para entrenar (sin fit).
        target: Nombre de la columna objetivo.
        param_space: Espacio de hiperparámetros del modelo (no de los
            indicadores). Mismo formato que antes:
            {nombre: (tipo, low, high[, step])} o {nombre: ("categorical", [...])}.
        indicator_registry: dict {familia: {"cls": Indicator, "params": {nombre: [valores]}}}.
            Por defecto usa DEFAULT_INDICATOR_REGISTRY con las 16 familias
            del proyecto. Se puede pasar un subconjunto o una versión
            personalizada para acotar el espacio de búsqueda.
        min_families: número mínimo de familias que debe activar un trial
            para considerarse válido (evita trials sin ninguna feature).
        train_window: Tamaño de la ventana de entrenamiento del walk-forward externo.
        test_window: Tamaño de la ventana de test del walk-forward externo.
        val_frac: Fracción del train de cada fold reservada como validación
            interna para las búsquedas de Optuna (por defecto 0.20).
        inner_metric: Métrica usada para evaluar cada trial sobre inner_val.
            Soportado: "pf_high" (profit factor del umbral óptimo, dirección long).
        n_trials_per_fold: Número de trials de Optuna a ejecutar en cada fold.
        step: Step del walk-forward externo (None = test_window).
        expanding: Modo expanding o rolling del walk-forward externo.
        min_kept: min_kept pasado a ThresholdEvaluator.find_optimized_threshold.
        mcpt: Si True, ejecuta un MCPT en modo "signal" (sin reentrenar) sobre
            el modelo final ya elegido por Optuna en cada fold, una sola vez.
        n_mcpt: Número de permutaciones del MCPT si mcpt=True.
        seed: Semilla para reproducibilidad.
        verbose: Si True, imprime el progreso de cada fold y cada trial.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model_builder,
        target: str,
        param_space: dict,
        train_window: int,
        test_window: int,
        indicator_registry: dict = None,
        min_families: int = 1,
        max_families: int = 5,
        val_frac: float = 0.20,
        inner_metric: str = "pf_high",
        n_trials_per_fold: int = 30,
        step: int = None,
        expanding: bool = False,
        min_kept: int = 300,
        mcpt: bool = False,
        n_mcpt: int = 200,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.data          = data
        self.model_builder  = model_builder
        self.target         = target
        self.param_space    = param_space
        self.indicator_registry = indicator_registry or DEFAULT_INDICATOR_REGISTRY
        self.min_families   = min_families
        self.max_families   = max_families

        self.train_window  = train_window
        self.test_window   = test_window
        self.step          = test_window if step is None else step
        self.expanding     = expanding

        self.val_frac          = val_frac
        self.inner_metric      = inner_metric
        self.n_trials_per_fold = n_trials_per_fold
        self.min_kept           = min_kept
        self.mcpt               = mcpt
        self.n_mcpt              = n_mcpt
        self.seed                = seed
        self.verbose              = verbose

        self.n      = len(data)
        self.splits = self._build_splits()

        self.fold_results = None

        if not self.verbose:
            set_verbosity(WARNING)

    # ------------------------------------------------------------------
    # Construcción de splits del walk-forward externo
    # ------------------------------------------------------------------

    def _build_splits(self):
        splits = []
        start = self.train_window

        while start + self.test_window <= self.n:
            train_start = 0 if self.expanding else start - self.train_window
            train_end   = start
            test_start  = start
            test_end    = start + self.test_window

            splits.append((train_start, train_end, test_start, test_end))
            start += self.step

        return splits

    # ------------------------------------------------------------------
    # Sugerencias jerárquicas de Optuna: familia -> parámetros
    # ------------------------------------------------------------------

    def _suggest_family_params(self, trial: optuna.Trial) -> dict:
        """Decide cuántas familias usar (dentro de [min_families, max_families])
        y cuáles, y para cada una sugiere sus parámetros concretos.

        A diferencia de una selección independiente familia por familia
        (donde el número de familias activas no está acotado y puede crecer
        sin control), aquí primero se fija cuántas familias se van a usar
        y luego se eligen sin repetición entre todas las disponibles. Esto
        acota el tamaño del espacio de búsqueda y favorece que Optuna
        converja a combinaciones más estables entre folds.

        Returns:
            dict {familia: {param: valor}} solo con las familias activadas
            en este trial.
        """
        family_names = list(self.indicator_registry.keys())

        low  = min(self.min_families, len(family_names))
        high = min(self.max_families, len(family_names))
        if low > high:
            low = high

        k = trial.suggest_int("n_families", low, high)

        available = list(family_names)
        selected = []
        for i in range(k):
            idx = trial.suggest_int(f"family_idx_{i}", 0, len(available) - 1)
            selected.append(available.pop(idx))

        chosen = {}
        for family_name in selected:
            spec = self.indicator_registry[family_name]
            params = {}
            for param_name, choices in spec["params"].items():
                params[param_name] = trial.suggest_categorical(
                    f"{family_name}_{param_name}", choices
                )
            chosen[family_name] = params

        return chosen

    def _build_feature_frame(self, family_params: dict) -> pd.DataFrame:
        """Construye el DataFrame de features a partir de una combinación
        familia->parámetros ya decidida (sin volver a preguntar a Optuna).

        Se usa tanto dentro de los trials (con family_params recién
        sugeridos) como en el reentrenamiento final del fold (con los
        family_params del mejor trial encontrado).

        Args:
            family_params: dict {familia: {param: valor}}.

        Returns:
            pd.DataFrame con todas las columnas generadas, indexado igual
            que self.data. Si family_params está vacío, devuelve un
            DataFrame vacío con el mismo índice.
        """
        if not family_params:
            return pd.DataFrame(index=self.data.index)

        frames = []
        for family_name, params in family_params.items():
            cls = self.indicator_registry[family_name]["cls"]
            indicator = cls(self.data, **params)
            frames.append(indicator.compute())

        feat_df = pd.concat(frames, axis=1)
        # Por si dos familias generasen columnas con el mismo nombre
        feat_df = feat_df.loc[:, ~feat_df.columns.duplicated()]
        return feat_df

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """Construye el diccionario de hiperparámetros del modelo a partir
        de param_space (esto es independiente de los indicadores)."""
        params = {}

        for name, spec in self.param_space.items():
            kind = spec[0]

            if kind == "int":
                low, high = spec[1], spec[2]
                step = spec[3] if len(spec) > 3 else 1
                params[name] = trial.suggest_int(name, low, high, step=step)

            elif kind == "float":
                low, high = spec[1], spec[2]
                step = spec[3] if len(spec) > 3 else None
                params[name] = trial.suggest_float(name, low, high, step=step)

            elif kind == "loguniform":
                low, high = spec[1], spec[2]
                params[name] = trial.suggest_float(name, low, high, log=True)

            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec[1])

            else:
                raise ValueError(f"Tipo de parámetro no soportado: {kind}")

        return params

    # ------------------------------------------------------------------
    # Optimización interna de un único fold
    # ------------------------------------------------------------------

    def _inner_objective(self, trial: optuna.Trial, train_start: int, train_end: int,
                          te_inner) -> float:
        """Construye dinámicamente los indicadores del trial, entrena con
        inner_train y evalúa con inner_val. Nunca toca el test real."""
        family_params = self._suggest_family_params(trial)

        if len(family_params) < self.min_families:
            trial.set_user_attr("error", "insufficient_families")
            return float("-inf")

        params = self._suggest_params(trial)

        feat_df = self._build_feature_frame(family_params)
        features = feat_df.columns.tolist()

        # Registramos family_params/features/params YA, antes de cualquier
        # intento de entrenamiento. Así, si el trial falla más adelante
        # (NaN, error del modelo, etc.), _optimize_fold sigue pudiendo
        # recuperar qué combinación se intentó, en vez de quedarse con un
        # family_params vacío que rompe la reconstrucción final en run().
        trial.set_user_attr("family_params", family_params)
        trial.set_user_attr("features", features)
        trial.set_user_attr("params", params)

        n_train = train_end - train_start
        n_val   = int(n_train * self.val_frac)
        inner_train_start = train_start
        inner_train_end   = train_end - n_val
        inner_val_start   = inner_train_end
        inner_val_end     = train_end

        X_inner_train = feat_df.iloc[inner_train_start:inner_train_end]
        y_inner_train = self.data[self.target].iloc[inner_train_start:inner_train_end]
        X_inner_val   = feat_df.iloc[inner_val_start:inner_val_end]
        y_inner_val   = self.data[self.target].iloc[inner_val_start:inner_val_end]

        # Los indicadores con ventana grande (p.ej. atr_window=252) no
        # tienen suficiente historia al principio de la serie y generan
        # NaN en esas filas. sklearn no admite NaN en el fit, así que
        # eliminamos las filas afectadas antes de entrenar/predecir.
        X_inner_train, y_inner_train = self._drop_nan_rows(X_inner_train, y_inner_train)
        X_inner_val = X_inner_val.dropna()

        min_rows = max(10, self.min_kept // 4)
        if len(X_inner_train) < min_rows or len(X_inner_val) < 5:
            trial.set_user_attr("error", "insufficient_valid_rows_after_dropna")
            return float("-inf")

        model = self.model_builder(params)

        try:
            model.fit(X_inner_train, y_inner_train)
            y_pred_val = pd.Series(
                model.predict(X_inner_val),
                index=X_inner_val.index,
                name="y_pred_inner_val"
            )

            te_inner.prepare(y_pred_val)
            opt = te_inner.find_optimized_threshold(min_kept=max(1, self.min_kept // 4))
            mse_val  = float(np.mean((y_inner_val.values - y_pred_val.values) ** 2))

        except Exception as e:
            trial.set_user_attr("error", str(e))
            return float("-inf")

        # Si optimizaramos por PF utilizariamos score
        score = opt.get(self.inner_metric, float("-inf"))
        if np.isinf(score) or np.isnan(score):
            score = float("-inf") if score != float("inf") else 1e6

        return score

    @staticmethod
    def _drop_nan_rows(X: pd.DataFrame, y: pd.Series):
        """Elimina filas con NaN en X o en y, manteniendo X e y alineados."""
        valid = X.notna().all(axis=1) & y.notna()
        return X.loc[valid], y.loc[valid]

    def _optimize_fold(self, train_start: int, train_end: int, fold: int) -> dict:
        """Lanza un mini-estudio de Optuna usando solo el train de un fold.

        Returns:
            dict con "family_params", "features" y "params" de la mejor
            combinación encontrada.
        """
        n_train = train_end - train_start
        n_val   = int(n_train * self.val_frac)
        inner_val_start = train_end - n_val
        inner_val_end   = train_end

        # ThresholdEvaluator interno: usa los returns del propio inner_val
        inner_data = self.data.iloc[inner_val_start:inner_val_end]
        te_inner = ThresholdEvaluator(
            np.log(inner_data["close"].shift(-1) / inner_data["close"])
        )

        sampler = optuna.samplers.TPESampler(seed=self.seed + fold)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        study.optimize(
            lambda trial: self._inner_objective(trial, train_start, train_end, te_inner),
            n_trials=self.n_trials_per_fold,
            show_progress_bar=False,
        )

        best = study.best_trial
        return {
            "family_params": best.user_attrs.get("family_params", {}),
            "features":      best.user_attrs.get("features", []),
            "params":        best.user_attrs.get("params", {}),
            "inner_score":   study.best_value,
        }

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(self) -> list:
        """Ejecuta el walk-forward completo, re-optimizando en cada fold.

        Returns:
            Lista de dicts (uno por fold) con: familia/parámetros de
            indicadores y del modelo elegidos, error de predicción
            (MSE/RMSE) en train/test, y las métricas habituales de profit
            factor en train/test.
        """
        returns = np.log(self.data["close"].shift(-1) / self.data["close"])
        te = ThresholdEvaluator(returns)

        fold_results = []

        for fold, (train_start, train_end, test_start, test_end) in enumerate(self.splits):

            if self.verbose:
                print(f"[Fold {fold}] optimizando con Optuna sobre train interno...")

            best = self._optimize_fold(train_start, train_end, fold)
            family_params = best["family_params"]
            features      = best["features"]
            params        = best["params"]

            # Reconstruye las features del fold completo (train + test) a
            # partir de la combinación familia->parámetros ganadora, sin
            # volver a preguntar a Optuna (ya está decidida).
            feat_df = self._build_feature_frame(family_params)

            if feat_df.shape[1] == 0:
                raise RuntimeError(
                    f"[Fold {fold}] Ningún trial de Optuna produjo una combinación "
                    f"válida de indicadores (todos los trials fallaron o quedaron en "
                    f"-inf). Revisa el log de errores por trial (trial.user_attrs['error']) "
                    f"o baja min_kept / min_families."
                )

            X_train = feat_df.iloc[train_start:train_end]
            y_train = self.data[self.target].iloc[train_start:train_end]
            X_test  = feat_df.iloc[test_start:test_end]
            y_test  = self.data[self.target].iloc[test_start:test_end]

            # Igual que en el objetivo interno: eliminamos filas con NaN
            # (por historia insuficiente al principio de la serie) antes
            # de entrenar y predecir.
            X_train, y_train = self._drop_nan_rows(X_train, y_train)
            valid_test = X_test.notna().all(axis=1) & y_test.notna()
            X_test, y_test = X_test.loc[valid_test], y_test.loc[valid_test]

            if len(X_train) == 0 or len(X_test) == 0:
                raise RuntimeError(
                    f"[Fold {fold}] No quedan filas válidas (sin NaN) en train o test "
                    f"tras eliminar NaN. Revisa que train_window sea mayor que la "
                    f"ventana máxima de los indicadores del registro."
                )

            # Reentrena con el TRAIN COMPLETO del fold (inner_train + inner_val)
            # usando la mejor combinación encontrada, y evalúa en el test real.
            model = self.model_builder(params)
            model.fit(X_train, y_train)

            y_pred_train = pd.Series(
                model.predict(X_train),
                index=X_train.index,
                name="y_pred_train"
            )
            te.prepare(y_pred_train)
            opt = te.find_optimized_threshold(min_kept=self.min_kept)

            y_pred_test = pd.Series(
                model.predict(X_test),
                index=X_test.index,
                name="y_pred_test"
            )
            eval_high = te.evaluate_threshold(y_pred_test, opt["high_thresh"])
            eval_low  = te.evaluate_threshold(y_pred_test, opt["low_thresh"])

            # Baseline: predecir siempre la media del target de train (naive)
            baseline_pred_train = np.full_like(y_train.values, y_train.mean(), dtype=float)
            mse_baseline_train   = float(np.mean((y_train.values - baseline_pred_train) ** 2))

            baseline_pred_test  = np.full_like(y_test.values, y_train.mean(), dtype=float)  # media de TRAIN aplicada a test, sin fuga de información
            mse_baseline_test   = float(np.mean((y_test.values - baseline_pred_test) ** 2))

            # --- Error de predicción (MSE / RMSE) train y test -------------------
            mse_train  = float(np.mean((y_train.values - y_pred_train.values) ** 2))
            rmse_train = float(np.sqrt(mse_train))
            mse_test   = float(np.mean((y_test.values - y_pred_test.values) ** 2))
            rmse_test  = float(np.sqrt(mse_test))

            # MCPT modo "signal" sobre el modelo final ya elegido (no reentrena)
            p_value_high = None
            p_value_low  = None
            if self.mcpt:
                _mc = MonteCarloPT(
                    self.data.loc[X_test.index].reset_index(drop=True),   # test filtrado, no train
                    seed=self.seed
                )
                pred_fold = y_pred_test.reset_index(drop=True).rename(y_pred_test.name)

                res_high = _mc.mcpt_threshold(pred_fold, n_test=self.n_mcpt,
                                            min_kept=self.min_kept, verbose=False)
                res_low  = _mc.mcpt_threshold(pred_fold, n_test=self.n_mcpt,
                                            min_kept=self.min_kept, flip_sign=True, verbose=False)

                p_value_high = res_high["p_value"]
                p_value_low  = res_low["p_value"]

            row = {
                "fold":                fold,
                "train_start":         train_start,
                "train_end":           train_end,
                "test_start":          test_start,
                "test_end":            test_end,
                "family_params":       family_params,
                "features":            features,
                "params":              params,
                "inner_score":         best["inner_score"],
                "model":               model,
                "mse_baseline_train": mse_baseline_train,
                "mse_baseline_test":  mse_baseline_test,
                "mse_train":           mse_train,
                "rmse_train":          rmse_train,
                "mse_test":            mse_test,
                "rmse_test":           rmse_test,
                "pf_train_high":       opt["pf_high"],
                "pf_train_low":        opt["pf_low"],
                "high_thresh":         opt["high_thresh"],
                "low_thresh":          opt["low_thresh"],
                "pf_test_long_above":  eval_high["pf_long_above"],
                "pf_test_short_above": eval_high["pf_short_above"],
                "pf_test_long_below":  eval_low["pf_long_below"],
                "pf_test_short_below": eval_low["pf_short_below"],
            }

            if self.mcpt:
                row["p_value_high"] = p_value_high
                row["p_value_low"]  = p_value_low

            fold_results.append(row)

            if self.verbose:
                p_str = f" p_hi={p_value_high:.3f}" if self.mcpt else ""
                families_str = ", ".join(family_params.keys()) if family_params else "(ninguna)"
                print(
                    f"[Fold {fold}] familias={families_str} "
                    f"inner_score={best['inner_score']:.3f} "
                    f"RMSE test={rmse_test:.5f} "
                    f"PF test long above={eval_high['pf_long_above']:.3f}{p_str}"
                )

        self.fold_results = fold_results
        return fold_results

    def summary(self):
        """Imprime un resumen agregado, el error de predicción medio y la
        evolución de familias de indicadores por fold."""
        if self.fold_results is None:
            raise RuntimeError("Llama a run() primero.")

        df = pd.DataFrame(self.fold_results)

        def safe_mean(col):
            return df[col].replace([np.inf, -np.inf], np.nan).mean()

        print("=" * 60)
        print("OPTUNA WALK-FORWARD SUMMARY")
        print("=" * 60)
        print(f"  Folds:                 {len(df)}")
        print(f"  Trials por fold:       {self.n_trials_per_fold}")
        print(f"  Val. interna (frac):   {self.val_frac}")
        print()
        print(f"  MSE baseline train medio:  {safe_mean('mse_baseline_train'):.6f}")
        print(f"  MSE baseline test medio:   {safe_mean('mse_baseline_test'):.6f}")
        print(f"  MSE train medio:           {safe_mean('mse_train'):.6f}")
        print(f"  RMSE train medio:          {safe_mean('rmse_train'):.6f}")
        print(f"  MSE test medio:            {safe_mean('mse_test'):.6f}")
        print(f"  RMSE test medio:           {safe_mean('rmse_test'):.6f}")
        print()
        print(f"  PF test long above medio:  {safe_mean('pf_test_long_above'):.4f}")
        print(f"  PF test long below medio:  {safe_mean('pf_test_long_below'):.4f}")
        print(f"  PF test short above medio: {safe_mean('pf_test_short_above'):.4f}")
        print(f"  PF test short below medio: {safe_mean('pf_test_short_below'):.4f}")
        print(f"  Folds PF test long above >1: {(df['pf_test_long_above'] > 1).sum()} / {len(df)}")
        print(f"  Folds PF test long below >1: {(df['pf_test_long_below'] > 1).sum()} / {len(df)}")
        print(f"  Folds PF test short above >1: {(df['pf_test_short_above'] > 1).sum()} / {len(df)}")
        print(f"  Folds PF test short below >1: {(df['pf_test_short_below'] > 1).sum()} / {len(df)}")
        if "p_value_high" in df.columns:
            print(f"  P-value high medio:        {safe_mean('p_value_high'):.4f}")
            print(f"  P-value low medio:         {safe_mean('p_value_low'):.4f}")
            print(f"  Folds p_value_high < 0.05: {(df['p_value_high'] < 0.05).sum()} / {len(df)}")
        print()
        print("--- Familias e indicadores elegidos por fold ---")
        for _, row in df.iterrows():
            fams = list(row["family_params"].keys()) if row["family_params"] else []
            print(f"  Fold {row['fold']}: familias={fams}")
            print(f"    features={row['features']}")
        print("=" * 60)