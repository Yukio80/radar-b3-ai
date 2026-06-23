import numpy as np
import pandas as pd
from qpe.regime_detector import RegimeDetector


def test_detect_bull():
    rd = RegimeDetector(lookback_days=252, vol_window=63)
    np.random.seed(42)
    r = pd.Series(np.random.normal(0.001, 0.015, 500))
    result = rd.detect(r)
    assert "regime" in result
    assert "classificacao" in result
    assert "confianca" in result


def test_detect_crisis():
    rd = RegimeDetector()
    np.random.seed(42)
    daily_ret = -0.002
    r = pd.Series(np.random.normal(daily_ret, 0.035, 500))
    result = rd.detect(r)
    assert result["regime"] == "crisis" or result["regime"] == "bear"


def test_detect_unknown_short_series():
    rd = RegimeDetector()
    r = pd.Series([0.01, 0.02, -0.01])
    result = rd.detect(r)
    assert result["regime"] == "unknown"


def test_adjust_weights_bull():
    weights = RegimeDetector.adjust_weights_for_regime("bull")
    assert weights["growth"] > 0.20
    assert weights["dividends"] < 0.20


def test_adjust_weights_bear():
    weights = RegimeDetector.adjust_weights_for_regime("bear")
    assert weights["safety"] > 0.15
    assert weights["growth"] < 0.20


def test_adjust_weights_high_rates():
    weights = RegimeDetector.adjust_weights_for_regime("high_rates")
    assert weights["dividends"] > 0.20
    assert weights["growth"] < 0.20


def test_adjust_weights_crisis():
    weights = RegimeDetector.adjust_weights_for_regime("crisis")
    assert weights["safety"] >= 0.20


def test_adjust_weights_with_base():
    base = {"quality": 0.30, "valuation": 0.20, "dividends": 0.20,
            "growth": 0.15, "safety": 0.15}
    weights = RegimeDetector.adjust_weights_for_regime("bull", base)
    assert abs(sum(weights.values()) - 1.0) < 0.01


def test_regime_description():
    desc = RegimeDetector.regime_description("bull")
    assert isinstance(desc, str)
    assert len(desc) > 10


def test_detect_rolling():
    rd = RegimeDetector(lookback_days=100, vol_window=30)
    np.random.seed(42)
    r = pd.Series(np.random.normal(0.001, 0.02, 300),
                  index=pd.date_range("2020-01-01", periods=300, freq="B"))
    df = rd.detect_rolling(r, window_size=100, step=20)
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert "regime" in df.columns


def test_adjust_weights_unknown():
    weights = RegimeDetector.adjust_weights_for_regime("unknown")
    assert abs(sum(weights.values()) - 1.0) < 0.01


def test_detect_with_cdi():
    rd = RegimeDetector()
    np.random.seed(42)
    r = pd.Series(np.random.normal(0.001, 0.02, 500))
    result = rd.detect(r, cdi_rate=0.15)
    assert "regime" in result


def test_detect_recovery():
    rd = RegimeDetector()
    np.random.seed(42)
    crisis_part = pd.Series(np.random.normal(-0.005, 0.04, 200))
    recovery_part = pd.Series(np.random.normal(0.003, 0.02, 200))
    r = pd.concat([crisis_part, recovery_part]).reset_index(drop=True)
    result = rd.detect(r)
    assert isinstance(result["regime"], str)
