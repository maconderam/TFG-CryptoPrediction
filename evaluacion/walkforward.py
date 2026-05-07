import copy
import numpy as np
import pandas as pd

from modelos.models import Model
from .thresholds import ThresholdEvaluator

class WalkForwardEvaluator:
    def __init__(self, df, features, target, train_window, test_window, step=None, expanding=False):
        self.df       = df
        self.features = features
        self.target   = target

        self.X = df[features]
        self.y = df[target]

        self.train_window = train_window
        self.test_window  = test_window
        self.step         = test_window if step is None else step
        self.expanding    = expanding

        self.n      = len(self.X)
        self.splits = self._build_splits()

        self.te = ThresholdEvaluator(df)

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

    def run(self, model: Model):
        if not isinstance(model, Model):
            raise TypeError("model must be an instance of Model")

        fold_results    = []
        all_predictions = []

        for fold, (train_start, train_end, test_start, test_end) in enumerate(self.splits):

            X_train = self.X.iloc[train_start:train_end]
            y_train = self.y.iloc[train_start:train_end]
            X_test  = self.X.iloc[test_start:test_end]
            y_test  = self.y.iloc[test_start:test_end]

            # Fresh copy of the model for each fold
            current_model = copy.deepcopy(model)
            current_model.fit(X_train, y_train)

            # Optimizar umbral con predicciones sobre train
            y_pred_train = pd.Series(
                current_model.predict(X_train),
                index=X_train.index,
                name="y_pred_train"
            )
            self.te.prepare(y_pred_train)
            opt = self.te.find_optimized_threshold()

            # Evaluar ambos umbrales sobre test
            y_pred_test = pd.Series(
                current_model.predict(X_test),
                index=X_test.index,
                name="y_pred_test"
            )
            eval_high = self.te.evaluate_threshold(y_pred_test, opt["high_thresh"])
            eval_low  = self.te.evaluate_threshold(y_pred_test, opt["low_thresh"])

            fold_results.append({
                "fold":                    fold,
                "train_start":             train_start,
                "train_end":               train_end,
                "test_start":              test_start,
                "test_end":                test_end,
                "train_size":              train_end - train_start,
                "test_size":               test_end - test_start,
                "model":                   current_model,
                "pf_train_high":           opt["pf_high"],
                "pf_train_low":            opt["pf_low"],
                "high_thresh":             opt["high_thresh"],
                "low_thresh":              opt["low_thresh"],
                # high threshold
                "pf_test_long_above":      eval_high["pf_long_above"],
                "pf_test_short_above":     eval_high["pf_short_above"],
                # low threshold
                "pf_test_long_below":      eval_low["pf_long_below"],
                "pf_test_short_below":     eval_low["pf_short_below"],
            })

            for idx, y_true, y_pred in zip(X_test.index, y_test, y_pred_test):
                all_predictions.append({
                    "fold":   fold,
                    "index":  idx,
                    "y_true": y_true,
                    "y_pred": y_pred,
                })

        self.fold_results   = fold_results
        self.predictions_df = pd.DataFrame(all_predictions).set_index("index")

        return fold_results, self.predictions_df

    def summary(self):
        if not hasattr(self, "fold_results"):
            raise RuntimeError("Call run() first")

        df_metrics = pd.DataFrame(self.fold_results)

        print("=" * 60)
        print("WALK-FORWARD EVALUATOR SUMMARY")
        print("=" * 60)

        # Configuracion
        print("\n--- Configuracion ---")
        print(f"  Mode:         {'expanding' if self.expanding else 'rolling'}")
        print(f"  Train window: {self.train_window}")
        print(f"  Test window:  {self.test_window}")
        print(f"  Step:         {self.step}")
        print(f"  Total folds:  {len(self.fold_results)}")
        print(f"  Total obs:    {self.n}")

        # Datos
        print("\n--- Datos ---")
        print(f"  Target:   {self.target}")
        print(f"  Features: {', '.join(self.features)}")

        # Modelo
        print("\n--- Modelo ---")
        print(f"  {self.fold_results[0]['model'].name}")

        # Metricas agregadas
        def safe_mean(col):
            return df_metrics[col].replace([np.inf], np.nan).mean()

        print("\n--- Metricas agregadas ---")
        print(f"  PF train high medio:         {safe_mean('pf_train_high'):.4f}")
        print(f"  PF train low medio:          {safe_mean('pf_train_low'):.4f}")
        print(f"  PF test long  above medio:   {safe_mean('pf_test_long_above'):.4f}")
        print(f"  PF test short above medio:   {safe_mean('pf_test_short_above'):.4f}")
        print(f"  PF test long  below medio:   {safe_mean('pf_test_long_below'):.4f}")
        print(f"  PF test short below medio:   {safe_mean('pf_test_short_below'):.4f}")
        print(f"  Degradacion high (train/test long above):  {safe_mean('pf_train_high') / safe_mean('pf_test_long_above'):.4f}")
        print(f"  Degradacion low  (train/test short below): {safe_mean('pf_train_low')  / safe_mean('pf_test_short_below'):.4f}")
        print(f"  Folds PF test long  above > 1: {(df_metrics['pf_test_long_above'] > 1).sum()} / {len(df_metrics)}")
        print(f"  Folds PF test short above > 1: {(df_metrics['pf_test_short_above'] > 1).sum()} / {len(df_metrics)}")
        print(f"  Folds PF test long  below > 1: {(df_metrics['pf_test_long_below'] > 1).sum()} / {len(df_metrics)}")
        print(f"  Folds PF test short below > 1: {(df_metrics['pf_test_short_below'] > 1).sum()} / {len(df_metrics)}")

        # Detalle por fold
        def fmt(v):
            return "inf" if np.isinf(v) else f"{v:.3f}"

        print("\n--- Detalle por fold ---")
        header = (f"{'Fold':>4} | {'Train':>12} | {'Test':>12} | "
                  f"{'PF tr.hi':>8} | {'PF tr.lo':>8} | "
                  f"{'L.abv':>6} | {'S.abv':>6} | {'L.blw':>6} | {'S.blw':>6} | "
                  f"{'Hi thr':>7} | {'Lo thr':>7}")
        print(header)
        print("-" * len(header))
        for r in self.fold_results:
            print(
                f"{r['fold']:>4} | "
                f"{r['train_start']:>5}-{r['train_end']:<5} | "
                f"{r['test_start']:>5}-{r['test_end']:<5} | "
                f"{fmt(r['pf_train_high']):>8} | "
                f"{fmt(r['pf_train_low']):>8} | "
                f"{fmt(r['pf_test_long_above']):>6} | "
                f"{fmt(r['pf_test_short_above']):>6} | "
                f"{fmt(r['pf_test_long_below']):>6} | "
                f"{fmt(r['pf_test_short_below']):>6} | "
                f"{r['high_thresh']:>7.4f} | "
                f"{r['low_thresh']:>7.4f}"
            )
        print("=" * 60)
 