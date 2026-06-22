import pandas as pd
import numpy as np
from qpe.outlier_detection import detect_outliers_iqr, detect_outliers_zscore, clean_outliers


def test_iqr_basic():
    s = pd.Series([1, 2, 3, 4, 5, 100])
    mask = detect_outliers_iqr(s)
    assert mask.iloc[-1] == True
    assert mask.iloc[:-1].sum() == 0


def test_zscore_basic():
    s = pd.Series([5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 1000])
    result = detect_outliers_zscore(s)
    assert result.loc[10, "is_outlier"] == True
    assert result["reason"].str.contains("zscore").any()


def test_clean_outliers():
    df = pd.DataFrame({"a": [1, 2, 3, 100], "b": [5, 6, 7, 8]})
    cleaned = clean_outliers(df, columns=["a"])
    assert pd.isna(cleaned.loc[3, "a"])
    assert cleaned.loc[3, "b"] == 8


def test_empty_series():
    s = pd.Series([], dtype=float)
    mask = detect_outliers_iqr(s)
    assert len(mask) == 0


def test_constant_series():
    s = pd.Series([5, 5, 5, 5])
    result = detect_outliers_zscore(s)
    assert not result["is_outlier"].any()
