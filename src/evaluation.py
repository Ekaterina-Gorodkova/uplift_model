import numpy as np
import pandas as pd

def uplift_at_k(y_true, treatment, uplift_score, k=0.3):
    df = pd.DataFrame({"y": y_true, "t": treatment, "score": uplift_score}).sort_values("score", ascending=False)
    n = max(1, int(len(df) * k))
    top = df.head(n)
    tr = top.loc[top["t"] == 1, "y"].mean()
    ct = top.loc[top["t"] == 0, "y"].mean()
    tr = 0 if pd.isna(tr) else tr
    ct = 0 if pd.isna(ct) else ct
    return float(tr - ct)

def qini_curve(y_true, treatment, uplift_score):
    df = pd.DataFrame({"y": y_true, "t": treatment, "score": uplift_score}).sort_values("score", ascending=False).reset_index(drop=True)
    df["treated_outcome"] = np.where(df["t"] == 1, df["y"], 0)
    df["control_outcome"] = np.where(df["t"] == 0, df["y"], 0)
    df["cum_treated"] = df["t"].cumsum()
    df["cum_control"] = (1 - df["t"]).cumsum()
    df["cum_treated_outcome"] = df["treated_outcome"].cumsum()
    df["cum_control_outcome"] = df["control_outcome"].cumsum()
    df["qini"] = df["cum_treated_outcome"] - df["cum_control_outcome"] * (df["cum_treated"] / df["cum_control"].replace(0, np.nan))
    df["qini"] = df["qini"].fillna(0)
    df["population_share"] = (np.arange(len(df)) + 1) / len(df)
    return df[["population_share", "qini"]]

def auuc(y_true, treatment, uplift_score):
    curve = qini_curve(y_true, treatment, uplift_score)
    return float(np.trapz(curve["qini"], curve["population_share"]))
