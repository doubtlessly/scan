# modules/online_model.py
import os
import pickle
import numpy as np
from sklearn.linear_model import SGDClassifier
from .config import ENABLE_ONLINE_LEARNING, MODEL_ONLINE_PATH

class OnlineModelHandler:
    def __init__(self, n_features: int = 5):
        self.enabled    = ENABLE_ONLINE_LEARNING
        self.model_file = MODEL_ONLINE_PATH
        if not self.enabled:
            print("ℹ️  Online learning disabled via config")
            return

        if os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✔ Loaded online model from {self.model_file}")
        else:
            # Initialize a new SGDClassifier for logistic regression
            self.model = SGDClassifier(
                loss='log_loss',      # use the valid 'log_loss' alias
                max_iter=1,
                warm_start=True,
                learning_rate='optimal'
            )
            # Dummy partial_fit to initialize classes_
            X0 = np.zeros((1, n_features))
            self.model.partial_fit(X0, [0], classes=[0,1])
            print("ℹ️  Initialized new online model")

    def predict_proba(self, fv: dict) -> float:
        if not self.enabled:
            return 0.0
        momentum = (fv['mid_price'] - fv['vwap']) / fv['vwap'] if fv['vwap'] else 0.0
        X = np.array([[ 
            fv['imbalance'],
            fv['spread'],
            fv['atr'],
            fv.get('funding_rate', 0.0),
            momentum
        ]])
        return float(self.model.predict_proba(X)[0][1])

    def update_from_simulator(self, sim):
        if not self.enabled:
            return
        rows = sim.get_untrained_trades()
        if not rows:
            return

        for tid, fv, pnl in rows:
            y = 1 if pnl > 0 else 0
            momentum = (fv['mid_price'] - fv['vwap']) / fv['vwap'] if fv['vwap'] else 0.0
            X = np.array([[ 
                fv['imbalance'],
                fv['spread'],
                fv['atr'],
                fv.get('funding_rate', 0.0),
                momentum
            ]])
            self.model.partial_fit(X, [y])
            sim.mark_trained(tid)

        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✔ Online model updated with {len(rows)} new samples and saved")
