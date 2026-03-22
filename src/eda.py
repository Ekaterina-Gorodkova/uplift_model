import numpy as np
import pandas as pd

def basic_overview(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "table": [name],
        "rows": [df.shape[0]],
        "cols": [df.shape[1]],
        "duplicates": [df.duplicated().sum()],
        "missing_total": [int(df.isna().sum().sum())],
    })

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.isna().mean().rename("missing_ratio").sort_values(ascending=False).reset_index()
           .rename(columns={"index": "feature"}))
    out["missing_pct"] = (out["missing_ratio"] * 100).round(2)
    return out

def conversion_by_group(df: pd.DataFrame, treatment_col: str, target_col: str) -> pd.DataFrame:
    return (df.groupby(treatment_col)[target_col].agg(["mean", "count", "sum"])
            .rename(columns={"mean": "conversion_rate", "sum": "positives"}).reset_index())

def observed_uplift(df: pd.DataFrame, treatment_col: str, target_col: str) -> float:
    treated = df.loc[df[treatment_col] == 1, target_col].mean()
    control = df.loc[df[treatment_col] == 0, target_col].mean()
    return float(treated - control)

def standardized_mean_diff(df: pd.DataFrame, feature: str, treatment_col: str) -> float:
    x1 = df.loc[df[treatment_col] == 1, feature].dropna()
    x0 = df.loc[df[treatment_col] == 0, feature].dropna()
    if len(x1) == 0 or len(x0) == 0:
        return np.nan
    m1, m0 = x1.mean(), x0.mean()
    s1, s0 = x1.std(ddof=1), x0.std(ddof=1)
    pooled = np.sqrt((s1**2 + s0**2) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((m1 - m0) / pooled)

def smd_report(df: pd.DataFrame, numeric_features: list[str], treatment_col: str) -> pd.DataFrame:
    rows = [{"feature": c, "smd": standardized_mean_diff(df, c, treatment_col)} for c in numeric_features]
    out = pd.DataFrame(rows)
    out["abs_smd"] = out["smd"].abs()
    return out.sort_values("abs_smd", ascending=False)
