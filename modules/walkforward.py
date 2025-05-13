# modules/walkforward.py
import pandas as pd

def generate_rolling_windows(
    df: pd.DataFrame,
    train_bars: int,
    test_bars: int,
    step_bars: int
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split a DataFrame of bars into rolling train/test windows.
    - train_bars: number of rows per in-sample window
    - test_bars:  number of rows per out-of-sample window
    - step_bars:  how many rows to advance each iteration
    """
    windows = []
    total = len(df)
    # walk through by step_bars, stop when there's not enough left
    for start in range(0, total - train_bars - test_bars + 1, step_bars):
        train = df.iloc[start       : start + train_bars]
        test  = df.iloc[start + train_bars : start + train_bars + test_bars]
        windows.append((train, test))
    return windows
