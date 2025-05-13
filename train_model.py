# train_model.py

import pandas as pd
import numpy as np
import ccxt
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.inspection import permutation_importance
from modules.features import build_feature_vector


def fetch_alternative_data(symbol: str, timeframe: str, limit: int):
    """
    Fetch on-chain and sentiment metrics (stubs).
    Returns two DataFrames with identical integer indices.
    """
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq='h')
    on_chain = pd.DataFrame({
        'net_flows': np.zeros(limit),
        'active_addresses': np.zeros(limit)
    }, index=dates)
    sentiment = pd.DataFrame({'sentiment': np.zeros(limit)}, index=dates)
    # Reset for positional alignment
    return on_chain.reset_index(drop=True), sentiment.reset_index(drop=True)


def fetch_price_and_orderbook(symbol: str, timeframe: str, limit: int):
    """
    Fetch historical OHLCV bars and an order book snapshot via CCXT.
    Returns DataFrame of price bars and order_book dict.
    """
    exchange = ccxt.kraken({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=None, limit=limit)
    order_book = exchange.fetch_order_book(symbol, limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df, order_book


def engineer_features(price_df: pd.DataFrame, order_book: dict, on_chain: pd.DataFrame, sentiment: pd.DataFrame):
    """
    Build a combined feature DataFrame from technical, on-chain, and sentiment data.
    """
    tech = build_feature_vector(price_df, order_book)
    tech.reset_index(drop=True, inplace=True)
    # Concatenate by position
    features = pd.concat([tech, on_chain, sentiment], axis=1)
    # Build target
    next_close = price_df['close'].shift(-1).reset_index(drop=True)
    curr_close = price_df['close'].reset_index(drop=True)
    features['target'] = (next_close > curr_close).astype(int)
    return features.dropna()


def split_data(features: pd.DataFrame):
    """
    Split features into train / val / test (60/20/20) sets.
    """
    n = len(features)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    X = features.drop(columns=['target'])
    y = features['target']
    return (
        X.iloc[:i1], y.iloc[:i1],
        X.iloc[i1:i2], y.iloc[i1:i2],
        X.iloc[i2:], y.iloc[i2:]
    )


def tune_histgb(X_train, y_train):
    """
    Hyperparameter tune a HistGradientBoostingClassifier via TimeSeriesSplit and RandomizedSearchCV.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_iter': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'l2_regularization': [0.0, 1.0, 10.0],
        'max_bins': [255]
    }
    base = HistGradientBoostingClassifier(loss='log_loss')
    search = RandomizedSearchCV(
        base, param_dist, n_iter=30,
        cv=tscv, scoring='accuracy', n_jobs=-1, random_state=0
    )
    search.fit(X_train, y_train)
    print("Best HGB params:", search.best_params_)
    return search.best_estimator_


def build_ensemble(hgb):
    """
    Construct a soft voting ensemble with HistGB, RandomForest, CatBoost, and KNN.
    """
    rf = RandomForestClassifier(n_estimators=300, max_depth=9, class_weight='balanced', random_state=0, n_jobs=-1)
    cb = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5)
    weights = [3, 1, 1, 1]
    return VotingClassifier(
        estimators=[('hgb', hgb), ('rf', rf), ('cb', cb), ('knn', knn)],
        voting='soft', weights=weights, n_jobs=-1
    )


def analyze_and_prune_features(hgb, X_train, y_train, X_val, y_val, clf):
    """
    Use permutation importance to identify and remove low-impact features.
    Returns updated classifier and list of dropped features.
    """
    print("Analyzing feature importances via permutation importance...")
    perm = permutation_importance(hgb, X_val, y_val, n_repeats=10, random_state=0, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=X_train.columns).sort_values(ascending=False)
    print(imp.head(20))
    low = imp[imp < 0.01].index.tolist()
    if low:
        print(f"Dropping low-importance features: {low}")
        # Drop from train and validation
        X_train.drop(columns=low, inplace=True)
        X_val.drop(columns=low, inplace=True)
        # Retrain ensemble without dropped features
        clf = build_ensemble(hgb)
        clf.fit(X_train, y_train)
    return clf, low


def optimize_threshold(clf, X_val, y_val):
    """
    Find classification threshold on validation set to maximize accuracy.
    """
    best_t, best_acc = 0.5, 0
    probs = clf.predict_proba(X_val)[:, 1]
    for t in np.linspace(0.2, 0.8, 61):
        acc = accuracy_score(y_val, (probs > t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    print(f"Optimal threshold: {best_t:.2f} -> accuracy: {best_acc:.4f}")
    return best_t


def evaluate(clf, threshold, X_test, y_test):
    """
    Evaluate ensemble on test set using optimized threshold.
    """
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs > threshold).astype(int)
    print(f"Test set accuracy at threshold {threshold:.2f}: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))


def main():
    # Pipeline orchestration
    symbol, timeframe, limit = 'BTC/USD', '1h', 3000
    on_chain, sentiment = fetch_alternative_data(symbol, timeframe, limit)
    price_df, order_book = fetch_price_and_orderbook(symbol, timeframe, limit)
    features = engineer_features(price_df, order_book, on_chain, sentiment)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(features)
    hgb = tune_histgb(X_train, y_train)
    clf = build_ensemble(hgb)
    clf.fit(X_train, y_train)
    clf, low_imp = analyze_and_prune_features(hgb, X_train, y_train, X_val, y_val, clf)
    # Drop low-importance features from test set to match trained model
    if low_imp:
        X_test.drop(columns=low_imp, inplace=True)
    threshold = optimize_threshold(clf, X_val, y_val)
    evaluate(clf, threshold, X_test, y_test)
    # Persist
    joblib.dump({'model': clf, 'threshold': threshold}, 'model.pkl')
    print("Model and threshold saved to model.pkl")

if __name__ == '__main__':
    main()
