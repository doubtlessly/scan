# modules/signals.py
import numpy as np
from .features import FeatureEngineer
from .config   import (
    SIGNAL_THRESHOLDS,
    ENABLE_FUNDING_FEATURE,
    FUNDING_WEIGHT,
    SYMBOL,
    ENABLE_MULTITF,
    MTF_WEIGHTS,
    ENSEMBLE_WEIGHT,
)
from .model    import ModelHandler

class SignalGenerator:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or SIGNAL_THRESHOLDS
        self.mlmodel    = ModelHandler()

    async def generate_signal(self, symbol: str = None) -> dict:
        symbol = symbol or SYMBOL
        fv     = await FeatureEngineer.build_feature_vector(symbol)

        # core
        imb      = fv['imbalance']
        momentum = (fv['mid_price'] - fv['vwap']) / fv['vwap'] if fv['vwap'] else 0
        vol      = fv['atr'] or 1e-9
        spread   = fv['spread'] / vol

        # TF momentum sum
        mtf_score = 0.0
        if ENABLE_MULTITF:
            for tf, w in MTF_WEIGHTS.items():
                mtf_score += w * fv.get(f'{tf}_momentum', 0.0)

        # funding
        funding = fv['funding_rate'] if ENABLE_FUNDING_FEATURE else 0

        # heuristic raw
        score_h = 0.4*imb + 0.2*momentum -0.1*spread + (FUNDING_WEIGHT*funding) + mtf_score
        prob_h  = 1/(1+np.exp(-score_h))

        # ML leg
        prob_m  = self.mlmodel.predict_proba(fv)

        # ensemble
        prob = ENSEMBLE_WEIGHT*prob_m + (1-ENSEMBLE_WEIGHT)*prob_h
        side = 'buy' if prob > self.thresholds['entry'] else 'none'

        return {
            'symbol':        symbol,
            'probability':   prob,
            'side':          side,
            'feature_vector':fv
        }
