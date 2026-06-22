import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union


def detect_outliers_iqr(
    series: pd.Series,
    k: float = 1.5,
) -> pd.Series:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Parameters
    ----------
    series : pd.Series
        Input numeric data.
    k : float, default=1.5
        Multiplier for the IQR range. Standard is 1.5; use 3.0 for extreme outliers.

    Returns
    -------
    pd.Series
        Boolean mask where True indicates an outlier.
    """
    if series.dropna().empty:
        return pd.Series([False] * len(series), index=series.index)

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return (series < lower) | (series > upper)


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detect outliers using the Z-score method.

    Parameters
    ----------
    series : pd.Series
        Input numeric data.
    threshold : float, default=3.0
        Z-score threshold above which a point is considered an outlier.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: value (original), zscore, is_outlier, reason.
    """
    clean = series.dropna()
    if clean.empty:
        result = pd.DataFrame({
            "value": series,
            "zscore": np.nan,
            "is_outlier": False,
            "reason": "",
        })
        return result

    mean = clean.mean()
    std = clean.std()
    if std == 0:
        result = pd.DataFrame({
            "value": series,
            "zscore": 0.0,
            "is_outlier": False,
            "reason": "",
        })
        return result

    zscores = (series - mean) / std
    is_outlier = zscores.abs() > threshold

    reasons = []
    for val, z, out in zip(series, zscores, is_outlier):
        if out:
            direction = "alto" if z > 0 else "baixo"
            reasons.append(f"zscore={abs(z):.2f} > {threshold} ({direction})")
        else:
            reasons.append("")

    return pd.DataFrame({
        "value": series,
        "zscore": zscores.round(3),
        "is_outlier": is_outlier,
        "reason": reasons,
    })


def clean_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    k: float = 1.5,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Remove outliers from specified columns by setting them to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : list of str, optional
        Columns to clean. If None, uses all numeric columns.
    method : str, default='iqr'
        Detection method: 'iqr' or 'zscore'.
    k : float, default=1.5
        IQR multiplier (only for IQR method).
    threshold : float, default=3.0
        Z-score threshold (only for Z-score method).

    Returns
    -------
    pd.DataFrame
        Copy of df with outliers replaced by NaN.
    """
    result = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in result.columns:
            continue
        if method == "iqr":
            mask = detect_outliers_iqr(result[col], k=k)
        else:
            info = detect_outliers_zscore(result[col], threshold=threshold)
            mask = info["is_outlier"]
        result.loc[mask, col] = np.nan

    return result


def outlier_report(
    df: pd.DataFrame,
    columns: List[str],
) -> Dict[str, Dict[str, Union[int, float, List]]]:
    """
    Generate a detailed outlier report for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : list of str
        Columns to analyse.

    Returns
    -------
    dict
        Per-column outlier statistics.
    """
    report: Dict[str, Dict] = {}
    for col in columns:
        if col not in df.columns:
            continue
        ser = df[col].dropna()
        if ser.empty:
            continue
        mask = detect_outliers_iqr(ser)
        outliers = ser[mask]
        report[col] = {
            "total": int(len(ser)),
            "outliers": int(mask.sum()),
            "pct_outliers": round(float(mask.mean() * 100), 2),
            "min": float(ser.min()),
            "max": float(ser.max()),
            "q1": float(ser.quantile(0.25)),
            "q3": float(ser.quantile(0.75)),
            "outlier_values": [round(float(v), 4) for v in outliers.head(10)],
        }
    return report
