source .venv/bin/activatefrom pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .evaluation import uplift_at_k
from .utils import load_config, ensure_dir

def build_preprocessor(df: pd.DataFrame, exclude_cols):
    features = [c for c in df.columns if c not in exclude_cols]
    X = df[features]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

def train_two_models(train_df: pd.DataFrame, config: dict):
    cols = config["columns"]
    target_col = cols["target"]
    treatment_col = cols["treatment"]
    client_col = cols["client_id"]

    exclude = [target_col, treatment_col, client_col]
    preprocessor = build_preprocessor(train_df, exclude)
    features = [c for c in train_df.columns if c not in exclude]
    X = train_df[features]
    y = train_df[target_col]
    t = train_df[treatment_col]

    treated_idx = t == 1
    control_idx = t == 0

    model_t = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
    model_c = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
    model_t.fit(X.loc[treated_idx], y.loc[treated_idx])
    model_c.fit(X.loc[control_idx], y.loc[control_idx])
    return model_t, model_c, features

def predict_uplift(model_t, model_c, df, features):
    p_t = model_t.predict_proba(df[features])[:, 1]
    p_c = model_c.predict_proba(df[features])[:, 1]
    return p_t - p_c

def main():
    config = load_config()
    cols = config["columns"]
    processed_dir = Path(config["paths"]["processed_dir"])

    train_feat = pd.read_parquet(processed_dir / "train_features.parquet")
    test_feat = pd.read_parquet(processed_dir / "test_features.parquet")

    model_t, model_c, features = train_two_models(train_feat, config)
    uplift_pred_train = predict_uplift(model_t, model_c, train_feat, features)
    print("Train uplift@30%:", uplift_at_k(train_feat[cols["target"]].values, train_feat[cols["treatment"]].values, uplift_pred_train, 0.3))

    uplift_pred_test = predict_uplift(model_t, model_c, test_feat, features)
    submission = test_feat[[cols["client_id"]]].copy()
    submission["uplift_score"] = uplift_pred_test

    ensure_dir("artifacts/models")
    ensure_dir("artifacts/submissions")
    joblib.dump(model_t, "artifacts/models/two_models_treated.pkl")
    joblib.dump(model_c, "artifacts/models/two_models_control.pkl")
    submission.to_csv("artifacts/submissions/submission_two_models.csv", index=False)
    print("Saved models and submission.")

if __name__ == "__main__":
    main()
