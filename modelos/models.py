from abc import ABC, abstractmethod
import pickle
import numpy as np
import pandas as pd
  
class Model(ABC):
    def __init__(self, name: str):
        self.name = name
        self.metrics = {}
        self.is_fitted = False
 
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass
 
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
 
    def add_metric(self, key: str, value):
        self.metrics[key] = value
 
    def summary(self):
        print("-" * 20)
        print(f"Model: {self.name}")
        print(f"Fitted: {self.is_fitted}")
 
        if not self.metrics:
            print("Metrics: none")
            return
 
        print("Metrics:")
        for k, v in self.metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
 
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")
 
    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        return model
 
 
class SklearnModel(Model):
    def __init__(self, model, name: str = None):
        model_name = name or type(model).__name__
        super().__init__(name=model_name)
        self.model = model
 
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
 
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")
        return self.model.predict(X)
 
    def feature_importance(self) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before feature_importance()")
 
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError(
                f"{self.name} does not expose feature_importances_"
            )
 
        return pd.Series(
            self.model.feature_importances_,
            index=self.model.feature_names_in_
        ).sort_values(ascending=False)