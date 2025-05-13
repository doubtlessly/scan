# modules/model.py
import os
import pickle
import pandas as pd
from .config import ENABLE_ML_MODEL, MODEL_PATH

class ModelHandler:
    def __init__(self):
        self.enabled = False
        if ENABLE_ML_MODEL:
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                self.enabled = True
                print(f"✔ Loaded ML model from {MODEL_PATH}")
            else:
                print(f"⚠️  ML model not found at {MODEL_PATH}, disabling ML ensemble")
        else:
            print("ℹ️  ML ensemble disabled via config")

    def predict_proba(self, fv: dict) -> float:
        """
        fv: feature_vector dict.
        Returns the model's probability, or 0 if disabled.
        """
        if not self.enabled:
            return 0.0
        # build DataFrame of features matching your training set:
        X = pd.DataFrame([{
            'imbalance':    fv['imbalance'],
            'spread':       fv['spread'],
            'atr':          fv['atr'],
            'funding_rate': fv.get('funding_rate', 0.0),
            'momentum':     (fv['mid_price'] - fv['vwap']) / fv['vwap'] if fv['vwap'] else 0.0
        }])
        proba = self.model.predict_proba(X)[:,1][0]
        return float(proba)
