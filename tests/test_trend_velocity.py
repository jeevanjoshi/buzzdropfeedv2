from src.engine.trend_velocity import calculate_ema_trend_velocity, calculate_zscore_anomaly
from src.engine.text_embeddings import calculate_rpm_cosine_similarity, calculate_semantic_novelty_index


def test_ema_trend_velocity():
    rising_history = [10.0, 20.0, 30.0, 40.0, 50.0, 80.0, 100.0]
    flat_history = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    tvs_rising = calculate_ema_trend_velocity(rising_history)
    tvs_flat = calculate_ema_trend_velocity(flat_history)

    assert tvs_rising > tvs_flat
    assert tvs_rising > 60.0
    assert abs(tvs_flat - 10.0) < 1e-3


def test_zscore_anomaly_detection():
    spiking_history = [10.0, 12.0, 11.0, 10.0, 11.0, 12.0, 100.0]
    z_score, is_spike = calculate_zscore_anomaly(spiking_history)

    assert is_spike is True
    assert z_score > 3.0


def test_rpm_cosine_similarity():
    high_rpm_text = "Federal Reserve benchmark interest rates and tech stock market portfolio"
    low_rpm_text = "how to bake chocolate chip cookies at home"

    rpm_high = calculate_rpm_cosine_similarity(high_rpm_text)
    rpm_low = calculate_rpm_cosine_similarity(low_rpm_text)

    assert rpm_high > rpm_low
    assert rpm_high >= 0.5


def test_semantic_novelty_index():
    past_texts = ["How Intel lost the chip market to TSMC"]
    cand_similar = "Intel semiconductor market share vs TSMC"
    cand_novel = "Federal Reserve interest rate inflation adjustments"

    idi_similar = calculate_semantic_novelty_index(cand_similar, past_texts)
    idi_novel = calculate_semantic_novelty_index(cand_novel, past_texts)

    assert idi_novel > idi_similar
