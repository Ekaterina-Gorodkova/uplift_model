from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import load_config, ensure_dir

def infer_campaign_date(clients: pd.DataFrame, config: dict) -> pd.Timestamp:
    first_issue_col = config["columns"]["first_issue_date"]
    candidate = pd.to_datetime(clients[first_issue_col], errors="coerce").max()
    if pd.notna(candidate):
        return candidate.normalize() + pd.Timedelta(days=1)
    return pd.Timestamp("2019-03-15")

def preprocess_clients(clients: pd.DataFrame, campaign_date: pd.Timestamp, config: dict) -> pd.DataFrame:
    cols = config["columns"]

    client_col = cols["client_id"]
    first_issue_col = cols["first_issue_date"]
    first_redeem_col = cols["first_redeem_date"]
    age_col = cols["age"]
    gender_col = cols["gender"]

    df = clients.copy()

    print("[clients] preprocessing started...")

    df[first_issue_col] = pd.to_datetime(df[first_issue_col], errors="coerce")
    df[first_redeem_col] = pd.to_datetime(df[first_redeem_col], errors="coerce")

    df["days_since_first_issue"] = (campaign_date - df[first_issue_col]).dt.days
    df["days_since_first_redeem"] = (campaign_date - df[first_redeem_col]).dt.days
    df["has_redeem_history"] = df[first_redeem_col].notna().astype(int)
    df["issue_to_redeem_days"] = (df[first_redeem_col] - df[first_issue_col]).dt.days

    df["age_missing"] = df[age_col].isna().astype(int)
    df[age_col] = df[age_col].fillna(df[age_col].median())

    df[gender_col] = df[gender_col].fillna("U").astype(str)
    df["gender_is_female"] = (df[gender_col] == "F").astype(int)
    df["gender_is_male"] = (df[gender_col] == "M").astype(int)
    df["gender_is_unknown"] = (df[gender_col] == "U").astype(int)

    keep_cols = [c for c in df.columns if c not in [first_issue_col, first_redeem_col]]
    out = df[keep_cols].copy()

    if client_col not in out.columns:
        raise KeyError(f"Column `{client_col}` not found in clients features")

    print(f"[clients] done, shape={out.shape}")
    return out

def _safe_div(numerator, denominator):
    result = numerator / denominator.replace({0: np.nan})
    return result.replace([np.inf, -np.inf], np.nan)

def build_purchase_features_chunked(
    purchases_path: str | Path,
    products: pd.DataFrame,
    campaign_date: pd.Timestamp,
    config: dict,
) -> pd.DataFrame:
    cols = config["columns"]
    fe_cfg = config["feature_engineering"]

    client_col = cols["client_id"]
    transaction_col = cols["transaction_id"]
    dt_col = cols["transaction_datetime"]
    product_col = cols["product_id"]
    qty_col = cols["product_quantity"]
    sum_col = cols["purchase_sum"]
    store_col = cols["store_id"]

    chunk_size = int(fe_cfg.get("chunk_size", 500_000))
    recent_windows = fe_cfg.get("recent_windows_days", [7, 30, 90])

    product_lookup_cols = [product_col, "level_1", "is_own_trademark", "is_alcohol"]
    product_lookup_cols = [c for c in product_lookup_cols if c in products.columns]
    products_small = products[product_lookup_cols].copy()

    totals = defaultdict(float)
    first_dt = {}
    last_dt = {}
    uniq_products = defaultdict(set)
    uniq_stores = defaultdict(set)
    uniq_days = defaultdict(set)
    uniq_transactions = defaultdict(set)
    level1_counts = defaultdict(lambda: defaultdict(float))
    own_tm_counts = defaultdict(float)
    alcohol_counts = defaultdict(float)
    recent_sum = {w: defaultdict(float) for w in recent_windows}
    recent_trx = {w: defaultdict(set) for w in recent_windows}
    recent_products = {w: defaultdict(set) for w in recent_windows}

    usecols = [
        client_col,
        transaction_col,
        dt_col,
        "regular_points_received",
        "express_points_received",
        "regular_points_spent",
        "express_points_spent",
        sum_col,
        store_col,
        product_col,
        qty_col,
        "trn_sum_from_iss",
        "trn_sum_from_red",
    ]

    print(f"[purchases] reading in chunks from {purchases_path}")
    print(f"[purchases] chunk_size={chunk_size}")

    chunk_counter = 0
    rows_seen = 0

    for chunk in pd.read_csv(
        purchases_path,
        usecols=usecols,
        parse_dates=[dt_col],
        chunksize=chunk_size,
        low_memory=False,
    ):
        chunk_counter += 1
        rows_seen += len(chunk)

        print(
            f"[purchases] chunk #{chunk_counter} loaded, "
            f"rows={len(chunk):,}, total_rows_seen={rows_seen:,}"
        )

        chunk = chunk[chunk[dt_col] < campaign_date].copy()

        print(
            f"[purchases] chunk #{chunk_counter} after date filter: "
            f"{len(chunk):,} rows"
        )

        if chunk.empty:
            continue

        chunk = chunk.merge(products_small, on=product_col, how="left")
        chunk["date_only"] = chunk[dt_col].dt.date
        chunk["days_before_campaign"] = (campaign_date - chunk[dt_col]).dt.days
        chunk["points_received"] = chunk["regular_points_received"].fillna(0) + chunk["express_points_received"].fillna(0)
        chunk["points_spent"] = chunk["regular_points_spent"].fillna(0) + chunk["express_points_spent"].fillna(0)

        for client_id, sub in chunk.groupby(client_col, sort=False):
            rows_cnt = float(len(sub))
            total_spent = float(sub[sum_col].fillna(0).sum())
            total_qty = float(sub[qty_col].fillna(0).sum())

            totals[(client_id, "rows_cnt")] += rows_cnt
            totals[(client_id, "total_spent")] += total_spent
            totals[(client_id, "total_quantity")] += total_qty
            totals[(client_id, "points_received")] += float(sub["points_received"].sum())
            totals[(client_id, "points_spent")] += float(sub["points_spent"].sum())
            totals[(client_id, "trn_sum_from_iss")] += float(sub["trn_sum_from_iss"].fillna(0).sum())
            totals[(client_id, "trn_sum_from_red")] += float(sub["trn_sum_from_red"].fillna(0).sum())

            cur_min_dt = sub[dt_col].min()
            cur_max_dt = sub[dt_col].max()
            if client_id not in first_dt or cur_min_dt < first_dt[client_id]:
                first_dt[client_id] = cur_min_dt
            if client_id not in last_dt or cur_max_dt > last_dt[client_id]:
                last_dt[client_id] = cur_max_dt

            uniq_products[client_id].update(sub[product_col].dropna().astype(str).tolist())
            uniq_stores[client_id].update(sub[store_col].dropna().astype(str).tolist())
            uniq_days[client_id].update(sub["date_only"].tolist())
            uniq_transactions[client_id].update(sub[transaction_col].dropna().astype(str).tolist())

            if "level_1" in sub.columns:
                for cat, cnt in sub["level_1"].value_counts(dropna=True).items():
                    level1_counts[client_id][str(cat)] += float(cnt)
            if "is_own_trademark" in sub.columns:
                own_tm_counts[client_id] += float(sub["is_own_trademark"].fillna(0).sum())
            if "is_alcohol" in sub.columns:
                alcohol_counts[client_id] += float(sub["is_alcohol"].fillna(0).sum())

            for window in recent_windows:
                recent_sub = sub[sub["days_before_campaign"] <= window]
                if not recent_sub.empty:
                    recent_sum[window][client_id] += float(recent_sub[sum_col].fillna(0).sum())
                    recent_trx[window][client_id].update(recent_sub[transaction_col].dropna().astype(str).tolist())
                    recent_products[window][client_id].update(recent_sub[product_col].dropna().astype(str).tolist())

        print(f"[purchases] chunk #{chunk_counter} processed")

    all_clients = sorted(
        set(first_dt.keys()) | set(last_dt.keys()) | set(uniq_products.keys()) | {k[0] for k in totals.keys()}
    )
    print(f"[purchases] all chunks processed, clients with history={len(all_clients):,}")

    rows = []
    for client_id in all_clients:
        rows_cnt = totals.get((client_id, "rows_cnt"), 0.0)
        transaction_cnt = len(uniq_transactions[client_id])
        total_spent = totals.get((client_id, "total_spent"), 0.0)
        total_quantity = totals.get((client_id, "total_quantity"), 0.0)
        active_days = len(uniq_days[client_id])
        unique_product_cnt = len(uniq_products[client_id])
        unique_store_cnt = len(uniq_stores[client_id])
        days_since_last_purchase = (campaign_date - last_dt[client_id]).days if client_id in last_dt else np.nan
        period_days = (last_dt[client_id] - first_dt[client_id]).days if client_id in last_dt and client_id in first_dt else np.nan

        top1_category_share = np.nan
        if rows_cnt > 0 and len(level1_counts[client_id]) > 0:
            top1_category_share = max(level1_counts[client_id].values()) / rows_cnt

        row = {
            client_col: client_id,
            "rows_cnt": rows_cnt,
            "transaction_cnt": transaction_cnt,
            "unique_product_cnt": unique_product_cnt,
            "unique_store_cnt": unique_store_cnt,
            "active_days": active_days,
            "period_days": period_days,
            "days_since_last_purchase": days_since_last_purchase,
            "total_spent": total_spent,
            "total_quantity": total_quantity,
            "avg_spent_per_line": total_spent / rows_cnt if rows_cnt else np.nan,
            "avg_spent_per_transaction": total_spent / transaction_cnt if transaction_cnt else np.nan,
            "avg_qty_per_line": total_quantity / rows_cnt if rows_cnt else np.nan,
            "avg_lines_per_transaction": rows_cnt / transaction_cnt if transaction_cnt else np.nan,
            "purchase_frequency_per_active_day": transaction_cnt / active_days if active_days else np.nan,
            "category_top1_share": top1_category_share,
            "own_trademark_share": own_tm_counts[client_id] / rows_cnt if rows_cnt else np.nan,
            "alcohol_share": alcohol_counts[client_id] / rows_cnt if rows_cnt else np.nan,
            "total_points_received": totals.get((client_id, "points_received"), 0.0),
            "total_points_spent": totals.get((client_id, "points_spent"), 0.0),
            "total_trn_sum_from_iss": totals.get((client_id, "trn_sum_from_iss"), 0.0),
            "total_trn_sum_from_red": totals.get((client_id, "trn_sum_from_red"), 0.0),
            "category_diversity_proxy": unique_product_cnt / rows_cnt if rows_cnt else np.nan,
        }

        for window in recent_windows:
            row[f"spent_last_{window}d"] = recent_sum[window][client_id]
            row[f"trx_last_{window}d"] = len(recent_trx[window][client_id])
            row[f"unique_products_last_{window}d"] = len(recent_products[window][client_id])

        rows.append(row)

    feat = pd.DataFrame(rows)

    if feat.empty:
        raise ValueError("Purchase features are empty. Check campaign_date or CSV parsing.")

    feat["log_total_spent"] = np.log1p(feat["total_spent"])
    feat["log_transaction_cnt"] = np.log1p(feat["transaction_cnt"])

    if "spent_last_30d" in feat.columns and "spent_last_90d" in feat.columns:
        feat["spend_trend_30_to_90"] = _safe_div(feat["spent_last_30d"], feat["spent_last_90d"])

    feat["high_value_user"] = (feat["total_spent"] >= feat["total_spent"].quantile(0.75)).astype(int)
    feat["inactive_user"] = (feat["days_since_last_purchase"] >= feat["days_since_last_purchase"].quantile(0.75)).astype(int)
    feat["frequent_user"] = (feat["transaction_cnt"] >= feat["transaction_cnt"].quantile(0.75)).astype(int)
    feat["diverse_user"] = (feat["unique_product_cnt"] >= feat["unique_product_cnt"].quantile(0.75)).astype(int)

    print(f"[purchases] final purchase features shape={feat.shape}")
    return feat

def merge_all_features(
    uplift_train: pd.DataFrame,
    uplift_test: pd.DataFrame,
    clients_feat: pd.DataFrame,
    purchases_feat: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    client_col = config["columns"]["client_id"]

    train = uplift_train.copy()
    test = uplift_test.copy()

    train["dataset_split"] = "train"
    test["dataset_split"] = "test"

    full_base = pd.concat([train, test], ignore_index=True, sort=False)
    print(f"[merge] full base shape before features={full_base.shape}")

    df = full_base.merge(clients_feat, on=client_col, how="left")
    df = df.merge(purchases_feat, on=client_col, how="left")

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in object_cols:
        df[col] = df[col].fillna("missing")
    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    print(f"[merge] full feature table shape={df.shape}")
    return df

def main() -> None:
    config = load_config()
    cols = config["columns"]
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = ensure_dir(config["paths"]["processed_dir"])

    print("[main] loading small tables...")
    clients = pd.read_csv(
        raw_dir / config["files"]["clients"],
        parse_dates=[cols["first_issue_date"], cols["first_redeem_date"]],
        low_memory=False,
    )
    products = pd.read_csv(raw_dir / config["files"]["products"], low_memory=False)
    uplift_train = pd.read_csv(raw_dir / config["files"]["uplift_train"], low_memory=False)
    uplift_test = pd.read_csv(raw_dir / config["files"]["uplift_test"], low_memory=False)
    print(f"[main] clients={clients.shape}, products={products.shape}, uplift_train={uplift_train.shape}, uplift_test={uplift_test.shape}")

    campaign_date = infer_campaign_date(clients, config)
    print(f"[main] inferred campaign_date={campaign_date}")

    clients_feat = preprocess_clients(clients, campaign_date, config)
    purchases_feat = build_purchase_features_chunked(
        raw_dir / config["files"]["purchases"],
        products,
        campaign_date,
        config,
    )

    full_feat = merge_all_features(uplift_train, uplift_test, clients_feat, purchases_feat, config)
    out_path = processed_dir / "full_features.parquet"
    full_feat.to_parquet(out_path, index=False)

    print(f"[save] saved full feature store to {out_path}")
    print("[main] feature engineering completed successfully")

if __name__ == "__main__":
    main()
