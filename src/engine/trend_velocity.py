import numpy as np
from typing import List, Tuple


def calculate_ema_trend_velocity(
    search_volume_history: List[float], lambda_decay: float = 0.35
) -> float:
    """
    Calculates Exponential Moving Average (EMA) trend velocity over a 7-day window.
    
    TVS = sum( e^(lambda * (t - 7)) * q(t) ) / sum( e^(lambda * (t - 7)) )
    """
    if not search_volume_history:
        return 0.0

    history = np.array(search_volume_history[-7:], dtype=float)
    n = len(history)
    time_indices = np.arange(1, n + 1)
    
    # Exponential weights favoring recent days
    weights = np.exp(lambda_decay * (time_indices - n))
    weighted_velocity = np.sum(weights * history) / np.sum(weights)
    
    return float(np.round(weighted_velocity, 4))


def calculate_zscore_anomaly(
    search_volume_history: List[float]
) -> Tuple[float, bool]:
    """
    Calculates Z-score anomaly metric for the current day search volume vs past history.
    Returns (z_score, is_significant_spike). Threshold: Z >= 1.75.
    """
    if len(search_volume_history) < 3:
        return 0.0, False

    history = np.array(search_volume_history, dtype=float)
    past_days = history[:-1]
    current_day = history[-1]

    mean = np.mean(past_days)
    std = np.std(past_days)

    if std < 1e-6:
        z_score = 0.0
    else:
        z_score = (current_day - mean) / std

    is_spike = bool(z_score >= 1.75)
    return float(np.round(z_score, 4)), is_spike
