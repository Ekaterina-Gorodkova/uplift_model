from pathlib import Path
import pandas as pd
from .utils import load_config, ensure_dir

def get_paths(config: dict) -> dict:
    raw_dir = Path(config["paths"]["raw_dir"])
    files = config["files"]
    return {
        "raw_dir": raw_dir,
        "clients": raw_dir / files["clients"],
        "products": raw_dir / files["products"],
        "purchases": raw_dir / files["purchases"],
        "uplift_train": raw_dir / files["uplift_train"],
        "uplift_test": raw_dir / files["uplift_test"],
    }

def read_raw_tables() -> dict:
    config = load_config()
    paths = get_paths(config)
    cols = config["columns"]
    tables = {
        "clients": pd.read_csv(paths["clients"], parse_dates=[cols["first_issue_date"], cols["first_redeem_date"]], low_memory=False),
        "products": pd.read_csv(paths["products"], low_memory=False),
        "uplift_train": pd.read_csv(paths["uplift_train"], low_memory=False),
        "uplift_test": pd.read_csv(paths["uplift_test"], low_memory=False),
    }
    return tables

def save_table_profiles(tables: dict) -> pd.DataFrame:
    profiles = []
    for name, df in tables.items():
        profiles.append({
            "table": name,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "missing_total": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        })
    out = pd.DataFrame(profiles)
    out.to_csv("data/processed/table_profiles.csv", index=False)
    return out

def save_data_dictionary(tables: dict) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        for col, dtype in df.dtypes.items():
            rows.append({
                "table": name,
                "column": col,
                "dtype": str(dtype),
                "missing": int(df[col].isna().sum()),
                "n_unique": int(df[col].nunique(dropna=True)),
            })
    out = pd.DataFrame(rows)
    out.to_csv("data/processed/data_dictionary.csv", index=False)
    return out

def main() -> None:
    ensure_dir("data/processed")
    tables = read_raw_tables()
    profiles = save_table_profiles(tables)
    dictionary = save_data_dictionary(tables)
    print(profiles)
    print(dictionary.head())

if __name__ == "__main__":
    main()
