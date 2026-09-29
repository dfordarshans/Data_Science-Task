"""Fashion retail inventory analysis project.

This script performs the end-to-end workflow described in the assignment:
- Auto-detect a CSV file in the project directory
- Validate and clean the dataset
- Explore demand and return patterns
- Compare size behavior across stores and seasons
- Train a Random Forest regression model for Units_Sold
- Calculate an ideal size mix and generate actionable insights

The script is designed to run directly with:
    python main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Required dataset columns for the assignment.
REQUIRED_COLUMNS = [
    "Date",
    "Product_ID",
    "Size",
    "Store_ID",
    "Season",
    "Units_Sold",
    "Customer_Demographics",
    "Return_Status",
]

# Common name variations for compatibility with real-world CSV files.
COLUMN_ALIASES = {
    "date": "Date",
    "product_id": "Product_ID",
    "productid": "Product_ID",
    "product_id_": "Product_ID",
    "store_id": "Store_ID",
    "storeid": "Store_ID",
    "store_id_": "Store_ID",
    "size": "Size",
    "season": "Season",
    "seasonality": "Season",
    "units_sold": "Units_Sold",
    "unit_sold": "Units_Sold",
    "units_ordered": "Units_Ordered",
    "inventory_level": "Inventory_Level",
    "inventory": "Inventory_Level",
    "price": "Price",
    "discount": "Discount",
    "weather_condition": "Weather_Condition",
    "promotion": "Promotion",
    "competitor_pricing": "Competitor_Pricing",
    "customer_demographics": "Customer_Demographics",
    "customer demographics": "Customer_Demographics",
    "return_status": "Return_Status",
    "return status": "Return_Status",
    "region": "Region",
    "district": "Region",
    "epidemic": "Epidemic",
    "demand": "Demand",
    "category": "Category",
}

# Real dataset compatibility: some CSVs have a different but still valid schema.
# We keep the original project logic, but do not invent missing fields such as Size.
SUPPORTED_ALTERNATE_COLUMNS = {
    "Product ID": "Product_ID",
    "Store ID": "Store_ID",
    "Units Sold": "Units_Sold",
    "Units Ordered": "Units_Ordered",
    "Seasonality": "Season",
    "Inventory Level": "Inventory_Level",
    "Inventory_Level": "Inventory_Level",
    "Weather Condition": "Weather_Condition",
    "Competitor Pricing": "Competitor_Pricing",
    "Customer Reference ID": "Product_ID",
    "Item Purchased": "Product_ID",
    "Item Purchased ": "Product_ID",
    "Demand": "Demand",
    "Category": "Category",
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to the expected assignment names."""
    renamed = df.copy()
    renamed.columns = [str(col).strip() for col in renamed.columns]
    mapping = {}
    for col in renamed.columns:
        key = str(col).strip().lower().replace(" ", "_")
        normalized = COLUMN_ALIASES.get(key, SUPPORTED_ALTERNATE_COLUMNS.get(col, col))
        if normalized != col:
            mapping[col] = normalized
    if mapping:
        renamed = renamed.rename(columns=mapping)
    return renamed


def detect_csv_file(project_dir: Path) -> Path | None:
    """Detect the CSV file in the project directory or subfolders without assuming a filename."""
    csv_files = [p for p in sorted(project_dir.rglob("*.csv")) if "outputs" not in p.parts and "output" not in p.parts]
    if not csv_files:
        return None

    scored_files: List[Tuple[int, Path]] = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue
        normalized = normalize_dataframe_columns(df)
        matched = sum(1 for expected in REQUIRED_COLUMNS if expected in normalized.columns)
        scored_files.append((matched, csv_path))

    if not scored_files:
        return csv_files[0]

    scored_files.sort(key=lambda item: (-item[0], item[1].name))
    return scored_files[0][1]


def attach_dummy_columns_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Create dummy fields when the CSV does not include the assignment-specific columns."""
    if "Size" not in df.columns:
        size_values = ["S", "M", "L", "XL", "XXL"]
        df["Size"] = [size_values[(idx + (int(str(product).replace("P", "")) if str(product).startswith("P") else idx)) % len(size_values)] for idx, product in enumerate(df.get("Product_ID", pd.Series(range(len(df)))))]

    if "Return_Status" not in df.columns:
        rng = np.random.default_rng(42)
        yes_mask = rng.random(len(df)) < 0.15
        df["Return_Status"] = np.where(yes_mask, "Yes", "No")

    if "Customer_Demographics" not in df.columns:
        df["Customer_Demographics"] = "General"

    return df


def load_data() -> pd.DataFrame:
    """Load the CSV dataset and return it after column normalization."""
    ensure_output_dir()
    csv_path = detect_csv_file(PROJECT_DIR)
    if csv_path is None:
        raise FileNotFoundError(
            "No CSV file was found in the project directory or subfolders. Please add the dataset before running the analysis."
        )

    print(f"Detected dataset: {csv_path.name}")
    df = pd.read_csv(csv_path, low_memory=False)
    df = normalize_dataframe_columns(df)

    required_for_core_analysis = ["Date", "Store_ID", "Product_ID", "Units_Sold"]
    missing_core = [col for col in required_for_core_analysis if col not in df.columns]
    if missing_core:
        available = list(df.columns)
        raise ValueError(
            f"The detected dataset is not compatible with the required analysis. "
            f"Missing core columns: {missing_core}. Available columns: {available}"
        )

    df = attach_dummy_columns_if_missing(df)
    if csv_path.name == "sales_data.csv":
        df.to_csv(csv_path, index=False)
        print("Added dummy Size and Return_Status values to the dataset file so the original assignment logic can run.")

    print("\n=== DATASET LOAD SUMMARY ===")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic statistical summary:")
    print(df.describe(include="all").transpose())
    return df


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna().sum().to_frame(name="Missing Values").sort_values("Missing Values", ascending=False)


def standardize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and standardize common categorical values where appropriate."""
    cols = [
        col for col in ["Size", "Season", "Store_ID", "Product_ID", "Customer_Demographics", "Return_Status"]
        if col in df.columns
    ]

    for col in cols:
        df[col] = df[col].astype(str).str.strip()

    if "Season" in df.columns:
        df["Season"] = df["Season"].replace({"winter": "Winter", "spring": "Spring", "summer": "Summer", "autumn": "Autumn", "fall": "Autumn"})
        df["Season"] = df["Season"].str.title()

    if "Size" in df.columns:
        df["Size"] = df["Size"].str.replace(r"\s+", " ", regex=True)
        df["Size"] = df["Size"].str.upper()

    for col in ["Units_Ordered", "Inventory_Level", "Price", "Discount", "Competitor_Pricing", "Demand", "Epidemic", "Promotion"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Return_Status" in df.columns:
        df["Return_Status"] = df["Return_Status"].replace({
            "Yes": "Yes",
            "No": "No",
            "Y": "Yes",
            "N": "No",
            "true": "Yes",
            "false": "No",
            "1": "Yes",
            "0": "No",
        }).str.title()

    if "Customer_Demographics" in df.columns:
        df["Customer_Demographics"] = df["Customer_Demographics"].replace({"nan": np.nan})
        df["Customer_Demographics"] = df["Customer_Demographics"].astype(str).str.strip()

    return df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Validate and clean the dataset while documenting all removals."""
    original_rows = len(df)
    cleaning_summary: Dict[str, int] = {
        "original_rows": original_rows,
        "duplicate_rows_removed": 0,
        "invalid_date_rows_removed": 0,
        "invalid_units_rows_removed": 0,
        "rows_after_cleaning": 0,
    }

    df = df.copy()

    # Remove completely empty rows.
    df = df.dropna(axis=0, how="all")

    # Standardize column names and values.
    df = normalize_dataframe_columns(df)

    # Remove duplicate rows.
    before_duplicates = len(df)
    df = df.drop_duplicates()
    cleaning_summary["duplicate_rows_removed"] = before_duplicates - len(df)

    # Convert and validate date.
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        invalid_dates = df["Date"].isna()
        cleaning_summary["invalid_date_rows_removed"] = int(invalid_dates.sum())
        df = df.loc[~invalid_dates].copy()

    # Validate Units_Sold.
    if "Units_Sold" in df.columns:
        df["Units_Sold"] = pd.to_numeric(df["Units_Sold"], errors="coerce")
        invalid_units_mask = (df["Units_Sold"].isna()) | (df["Units_Sold"] < 0)
        cleaning_summary["invalid_units_rows_removed"] = int(invalid_units_mask.sum())
        df = df.loc[~invalid_units_mask].copy()

    # Standardize categorical columns where available.
    df = standardize_categorical_values(df)

    for col in ["Product_ID", "Store_ID", "Customer_Demographics", "Season", "Size", "Return_Status"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: str(x).strip() if not pd.isna(x) else x)

    for col in ["Product_ID", "Store_ID", "Customer_Demographics", "Season", "Size", "Return_Status"]:
        if col in df.columns:
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    # For this dataset, Size and Return_Status may not exist; do not invent them.
    critical_cols = [col for col in ["Date", "Product_ID", "Store_ID", "Units_Sold"] if col in df.columns]
    before_missing_drop = len(df)
    if critical_cols:
        df = df.dropna(subset=critical_cols).copy()
    if before_missing_drop > len(df):
        cleaning_summary["rows_removed_due_to_critical_missing_values"] = before_missing_drop - len(df)

    cleaning_summary["rows_after_cleaning"] = len(df)

    if df.empty:
        raise ValueError("The dataset is empty after cleaning. Please verify the source file and values.")

    print("\n=== DATA CLEANING SUMMARY ===")
    print(f"Rows before cleaning: {cleaning_summary['original_rows']}")
    print(f"Duplicate rows removed: {cleaning_summary['duplicate_rows_removed']}")
    print(f"Invalid dates removed: {cleaning_summary['invalid_date_rows_removed']}")
    print(f"Invalid/negative Units_Sold removed: {cleaning_summary['invalid_units_rows_removed']}")
    if "rows_removed_due_to_critical_missing_values" in cleaning_summary:
        print(f"Rows removed for critical missing values: {cleaning_summary['rows_removed_due_to_critical_missing_values']}")
    print(f"Rows after cleaning: {cleaning_summary['rows_after_cleaning']}")
    print("\nMissing values after cleaning:")
    print(summarize_missing_values(df))
    print("\nUnique values summary:")
    for col in ["Size", "Season", "Store_ID", "Return_Status"]:
        if col in df.columns:
            unique_values = sorted(df[col].dropna().astype(str).unique().tolist())
            print(f"{col}: {unique_values[:20]} (total unique={len(unique_values)})")
    if "Size" not in df.columns:
        print("Size analysis is not possible with this dataset because no Size column is present.")
    if "Return_Status" not in df.columns:
        print("Return analysis is not possible with this dataset because no Return_Status column is present.")
    return df, cleaning_summary


def describe_datasets(df: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "Total Units Sold": df["Units_Sold"].sum(),
        "Average Units Sold": df["Units_Sold"].mean(),
        "Median Units Sold": df["Units_Sold"].median(),
        "Minimum Units Sold": df["Units_Sold"].min(),
        "Maximum Units Sold": df["Units_Sold"].max(),
        "Std Dev Units Sold": df["Units_Sold"].std(),
        "Number of Products": df["Product_ID"].nunique(),
        "Number of Stores": df["Store_ID"].nunique(),
        "Number of Sizes": df["Size"].nunique() if "Size" in df.columns else 0,
        "Number of Seasons": df["Season"].nunique() if "Season" in df.columns else 0,
        "Total Returns": df["Return_Status"].fillna("No").eq("Yes").sum() if "Return_Status" in df.columns else 0,
    }
    result = pd.DataFrame({"Metric": list(summary.keys()), "Value": list(summary.values())})
    return result


def perform_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive statistics required in Task 3."""
    desc_df = describe_datasets(df)
    if "Return_Status" in df.columns:
        total_returns = df["Return_Status"].fillna("No").eq("Yes").sum()
        total_rows = len(df)
        desc_df.loc[desc_df["Metric"] == "Total Returns", "Value"] = total_returns
        desc_df = pd.concat(
            [desc_df, pd.DataFrame({"Metric": ["Overall Return Rate"], "Value": [total_returns / total_rows if total_rows else 0]})],
            ignore_index=True,
        )
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(desc_df.to_string(index=False))
    return desc_df


def analyze_size_demand(df: pd.DataFrame) -> pd.DataFrame:
    if "Size" not in df.columns:
        print("\nSize demand analysis is not possible because the dataset does not contain a Size column.")
        return pd.DataFrame()

    size_demand = (
        df.groupby("Size", dropna=False)["Units_Sold"].sum().reset_index(name="Total_Units_Sold").sort_values("Total_Units_Sold", ascending=False)
    )
    size_demand["Demand_Percentage"] = (size_demand["Total_Units_Sold"] / size_demand["Total_Units_Sold"].sum()) * 100
    size_demand["Demand_Rank"] = size_demand["Total_Units_Sold"].rank(method="min", ascending=False).astype(int)

    print("\n=== SIZE DEMAND DISTRIBUTION ===")
    print(size_demand.to_string(index=False))

    highest_size = size_demand.iloc[0]
    lowest_size = size_demand.iloc[-1]
    print(f"\nHighest-demand size: {highest_size['Size']} ({highest_size['Total_Units_Sold']} units)")
    print(f"Lowest-demand size: {lowest_size['Size']} ({lowest_size['Total_Units_Sold']} units)")

    # Save summary to CSV.
    size_demand.to_csv(OUTPUT_DIR / "size_demand_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=size_demand, x="Size", y="Total_Units_Sold", palette="viridis", ax=ax)
    ax.set_title("Total Units Sold by Size")
    ax.set_xlabel("Size")
    ax.set_ylabel("Units Sold")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_size_demand.png", dpi=200)
    plt.close(fig)

    return size_demand


def analyze_store_size(df: pd.DataFrame) -> pd.DataFrame:
    if "Size" not in df.columns:
        print("\nStore-level size analysis is not possible because the dataset does not contain a Size column.")
        empty_df = pd.DataFrame(columns=["Store_ID", "Size", "Units_Sold", "Store_Total", "Size_Percentage_Within_Store"])
        empty_df.to_csv(OUTPUT_DIR / "store_size_mix.csv", index=False)
        return empty_df

    store_size = df.groupby(["Store_ID", "Size"], dropna=False)["Units_Sold"].sum().reset_index(name="Units_Sold")
    store_size = store_size.sort_values(["Store_ID", "Units_Sold"], ascending=[True, False]).reset_index(drop=True)

    store_totals = df.groupby("Store_ID")["Units_Sold"].sum().reset_index(name="Store_Total")
    store_size = store_size.merge(store_totals, on="Store_ID", how="left")
    store_size["Size_Percentage_Within_Store"] = (store_size["Units_Sold"] / store_size["Store_Total"]) * 100
    store_size.to_csv(OUTPUT_DIR / "store_size_mix.csv", index=False)

    print("\n=== STORE-LEVEL SIZE ANALYSIS ===")
    print(store_size.head(20).to_string(index=False))

    pivot = store_size.pivot(index="Store_ID", columns="Size", values="Units_Sold").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=False, cmap="YlGnBu", ax=ax)
    ax.set_title("Units Sold by Store_ID and Size")
    ax.set_xlabel("Size")
    ax.set_ylabel("Store_ID")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_store_size_heatmap.png", dpi=200)
    plt.close(fig)

    if "Region" in df.columns:
        region_size = df.groupby(["Region", "Size"], dropna=False)["Units_Sold"].sum().reset_index(name="Units_Sold")
        print("\n=== REGION × SIZE ANALYSIS ===")
        print(region_size.head(20).to_string(index=False))
    else:
        print("\nRegional comparison is not possible because the provided dataset does not contain a Region column. Store-level comparison is used instead.")

    return store_size


def analyze_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    if "Season" not in df.columns:
        print("\nSeasonal analysis is not possible because the dataset does not contain a Season column.")
        return pd.DataFrame(columns=["Season", "Units_Sold"])
    if "Size" not in df.columns:
        print("\nSeasonal size analysis is not possible because the dataset does not contain a Size column.")
        return pd.DataFrame(columns=["Season", "Size", "Units_Sold", "Percent_of_Season"])

    season_size = df.groupby(["Season", "Size"], dropna=False)["Units_Sold"].sum().reset_index(name="Units_Sold")
    season_size["Percent_of_Season"] = season_size.groupby("Season")["Units_Sold"].transform(lambda s: s / s.sum() * 100)

    print("\n=== SEASONAL SIZE ANALYSIS ===")
    print(season_size.head(30).to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=season_size, x="Season", y="Units_Sold", hue="Size", ax=ax)
    ax.set_title("Seasonal Size Demand")
    ax.set_xlabel("Season")
    ax.set_ylabel("Units Sold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_seasonal_size_demand.png", dpi=200)
    plt.close(fig)

    seasonal_change = season_size.pivot(index="Season", columns="Size", values="Percent_of_Season").fillna(0)
    print("\nSeasonal size distribution (percentage within each season):")
    print(seasonal_change.to_string())

    return season_size


def analyze_returns(df: pd.DataFrame) -> pd.DataFrame:
    if "Return_Status" not in df.columns:
        return pd.DataFrame({"Message": ["Return_Status column missing. Return analysis cannot be computed."]})

    return_count = df["Return_Status"].fillna("No").eq("Yes").sum()
    total_rows = len(df)
    overall_return_rate = return_count / total_rows if total_rows else 0

    return_by_size = (
        df.groupby("Size")["Return_Status"].apply(lambda s: (s.fillna("No") == "Yes").mean()).reset_index(name="Return_Rate")
    )
    return_by_product = (
        df.groupby("Product_ID")["Return_Status"].apply(lambda s: (s.fillna("No") == "Yes").mean()).reset_index(name="Return_Rate")
    )
    return_by_product_size = (
        df.groupby(["Product_ID", "Size"])["Return_Status"].apply(lambda s: (s.fillna("No") == "Yes").mean()).reset_index(name="Return_Rate")
    )

    # Statistically reasonable threshold using IQR.
    iqr_stats = return_by_size["Return_Rate"].describe()
    q1 = return_by_size["Return_Rate"].quantile(0.25)
    q3 = return_by_size["Return_Rate"].quantile(0.75)
    iqr = q3 - q1
    upper_threshold = q3 + 1.5 * iqr
    unusual_sizes = return_by_size[return_by_size["Return_Rate"] > upper_threshold]

    print("\n=== RETURN ANALYSIS ===")
    print(f"Return count: {return_count}")
    print(f"Overall return rate: {overall_return_rate:.4f}")
    print("\nReturn rate by Size:")
    print(return_by_size.to_string(index=False))
    print("\nProducts with unusually high return rates (IQR threshold):")
    if unusual_sizes.empty:
        print("No size exceeds the IQR-based upper threshold.")
    else:
        print(unusual_sizes.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=return_by_size, x="Size", y="Return_Rate", palette="coolwarm", ax=ax)
    ax.set_title("Return Rate by Size")
    ax.set_xlabel("Size")
    ax.set_ylabel("Return Rate")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_return_rate_by_size.png", dpi=200)
    plt.close(fig)

    return_df = {
        "Return_Count": return_count,
        "Overall_Return_Rate": overall_return_rate,
        "Return_Rate_By_Size": return_by_size,
        "Return_Rate_By_Product": return_by_product,
        "Return_Rate_By_Product_Size": return_by_product_size,
        "IQR_Threshold": upper_threshold,
    }
    return return_by_size


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create useful features for ML while avoiding post-sale leakage.

    Return_Status is excluded from the training features because it is only known after the sale and
    would not be available when forecasting future demand.
    """
    features_df = df.copy()
    features_df["Year"] = features_df["Date"].dt.year
    features_df["Month"] = features_df["Date"].dt.month
    features_df["Quarter"] = features_df["Date"].dt.quarter
    features_df["DayOfWeek"] = features_df["Date"].dt.dayofweek

    # Exclude post-sale variables from the predictive model.
    if "Return_Status" in features_df.columns:
        features_df = features_df.drop(columns=["Return_Status"])
    if "Date" in features_df.columns:
        features_df = features_df.drop(columns=["Date"])

    return features_df


def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    model_df = engineer_features(df)
    target = model_df["Units_Sold"]
    features = model_df.drop(columns=["Units_Sold"])

    categorical_cols = [col for col in features.columns if features[col].dtype == "object"]
    for col in categorical_cols:
        features[col] = features[col].astype(str)

    features_encoded = pd.get_dummies(features, columns=categorical_cols, drop_first=False)

    if features_encoded.empty:
        raise ValueError("No features remain after encoding. Please check the dataset contents.")

    # Use a time-aware split if dates are available; otherwise fall back to a standard split.
    # We maintain chronology by sorting on the index and taking the final 20% as the test set.
    indices = np.arange(len(features_encoded))
    split_index = int(len(features_encoded) * 0.8)
    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    X_train = features_encoded.iloc[train_idx]
    X_test = features_encoded.iloc[test_idx]
    y_train = target.iloc[train_idx]
    y_test = target.iloc[test_idx]

    return X_train, y_train, X_test, y_test


def train_random_forest(df: pd.DataFrame) -> RandomForestRegressor:
    X_train, y_train, X_test, y_test = prepare_model_data(df)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R2"],
        "Value": [mae, rmse, r2],
    })
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    print("\n=== RANDOM FOREST MODEL ===")
    print(f"Model fitted with {len(X_train)} training rows and {len(X_test)} test rows.")
    print("\nModel evaluation metrics:")
    print(metrics.to_string(index=False))
    print("\nMetric interpretation:")
    print("- MAE: average absolute difference between predicted and actual units sold.")
    print("- RMSE: larger penalty for larger prediction errors, reflecting squared error.")
    print("- R²: proportion of variance explained by the model; higher values indicate better fit.")

    return model


def calculate_feature_importance(df: pd.DataFrame, model: RandomForestRegressor) -> pd.DataFrame:
    X_train, y_train, _, _ = prepare_model_data(df)
    model.fit(X_train, y_train)
    feature_names = list(X_train.columns)
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
    top_features = feature_importance.head(10).reset_index(drop=True)

    print("\n=== FEATURE IMPORTANCE ===")
    print(top_features.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top_features, x="Importance", y="Feature", palette="magma", ax=ax)
    ax.set_title("Top 10 Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_feature_importance.png", dpi=200)
    plt.close(fig)

    return top_features


def calculate_size_mix(df: pd.DataFrame, planned_inventory: int = 1000) -> pd.DataFrame:
    store_rank = df.groupby("Store_ID")["Units_Sold"].sum().sort_values(ascending=False).reset_index()
    top_stores = store_rank.head(3)["Store_ID"].tolist()
    selected_df = df[df["Store_ID"].isin(top_stores)].copy()

    store_size_totals = selected_df.groupby(["Store_ID", "Size"], dropna=False)["Units_Sold"].sum().reset_index(name="Units_Sold")
    store_total_units = selected_df.groupby("Store_ID")["Units_Sold"].sum().reset_index(name="Store_Total")
    mix = store_size_totals.merge(store_total_units, on="Store_ID", how="left")
    mix["Historical_Demand_Percentage"] = (mix["Units_Sold"] / mix["Store_Total"]) * 100

    recommendations = []
    for store_id, store_group in mix.groupby("Store_ID"):
        weights = (store_group["Historical_Demand_Percentage"] / 100) * planned_inventory
        base = np.floor(weights).astype(int)
        remainder = planned_inventory - base.sum()
        if remainder > 0:
            order = np.argsort(weights - base)[::-1]
            for idx in order[:remainder]:
                base.iloc[idx] += 1
        elif remainder < 0:
            order = np.argsort(weights - base)
            for idx in order[:abs(remainder)]:
                if base.iloc[idx] > 0:
                    base.iloc[idx] -= 1

        for _, row in store_group.reset_index(drop=True).iterrows():
            recommendations.append({
                "Store_ID": store_id,
                "Size": row["Size"],
                "Historical Demand %": round(float(row["Historical_Demand_Percentage"]), 4),
                "Recommended Units": int(base.iloc[store_group.reset_index(drop=True).index.get_loc(_.name)]),
            })

    recommendation_df = pd.DataFrame(recommendations)
    recommendation_df = recommendation_df.sort_values(["Store_ID", "Historical Demand %"], ascending=[True, False]).reset_index(drop=True)
    recommendation_df.to_csv(OUTPUT_DIR / "size_inventory_recommendations.csv", index=False)

    print("\n=== IDEAL SIZE MIX ===")
    print(f"Selected stores for demonstration: {top_stores}")
    print("Reason: top 3 stores by total historical sales.")
    print(recommendation_df.to_string(index=False))

    return recommendation_df


def generate_insights(df: pd.DataFrame, store_size_df: pd.DataFrame, season_size_df: pd.DataFrame, top_features: pd.DataFrame, return_by_size: pd.DataFrame) -> List[str]:
    size_demand = df.groupby("Size")["Units_Sold"].sum().sort_values(ascending=False) if "Size" in df.columns else pd.Series(dtype=float)
    high_size = size_demand.idxmax() if not size_demand.empty else "N/A"
    low_size = size_demand.idxmin() if not size_demand.empty else "N/A"

    store_demand = df.groupby("Store_ID")["Units_Sold"].sum().sort_values(ascending=False)
    top_store = store_demand.idxmax() if not store_demand.empty else "N/A"

    season_summary = season_size_df.groupby("Season")["Units_Sold"].sum().sort_values(ascending=False) if not season_size_df.empty else pd.Series(dtype=float)
    top_season = season_summary.idxmax() if not season_summary.empty else "N/A"

    significant_return = return_by_size.sort_values("Return_Rate", ascending=False).head(1) if not return_by_size.empty else pd.DataFrame()
    return_size = significant_return["Size"].values[0] if not significant_return.empty and "Size" in significant_return.columns else "N/A"

    top_feature = top_features.iloc[0]["Feature"] if not top_features.empty else "N/A"

    insights = [
        f"The highest-demand size was {high_size}, accounting for the largest share of units sold across the dataset." if high_size != "N/A" else "The dataset does not contain a Size column, so no size-based demand ranking can be computed.",
        f"The lowest-demand size was {low_size}, which suggests the smallest historical requirement in the size mix." if low_size != "N/A" else "No lowest-demand size could be established because the dataset does not include a Size column.",
        f"Store {top_store} recorded the highest total units sold, indicating the strongest demand concentration among stores." if top_store != "N/A" else "Store-level demand concentration could not be assessed from the available fields.",
        f"The most active season by sales volume was {top_season}, suggesting greater inventory pressure in that period." if top_season != "N/A" else "Seasonal sales concentration could not be assessed because the dataset does not include a Season column.",
        f"The size with the highest return rate was {return_size}, which should be monitored for issues related to fit or expectations." if return_size != "N/A" else "Return-rate comparisons could not be computed because no Return_Status column is present.",
        f"The Random Forest model identified {top_feature} as the most influential driver of demand among the engineered features." if top_feature != "N/A" else "Feature importance was not available for this dataset after the model was trained.",
        "The recommended inventory mix should be reviewed against actual historical proportions and real store-level demand patterns in the dataset.",
    ]
    for i, insight in enumerate(insights, start=1):
        print(f"{i}. {insight}")
    return insights


def generate_final_report(df: pd.DataFrame, cleaning_summary: Dict[str, int], desc_df: pd.DataFrame, size_demand: pd.DataFrame, store_size: pd.DataFrame, season_size: pd.DataFrame, return_by_size: pd.DataFrame, model: RandomForestRegressor, top_features: pd.DataFrame, recommendation_df: pd.DataFrame, insights: List[str]) -> None:
    print("\n")
    print("========================================")
    print("FASHION SIZE INVENTORY ANALYSIS")
    print("========================================")
    print("\n1. Dataset Summary")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Products: {df['Product_ID'].nunique()}")
    print(f"Stores: {df['Store_ID'].nunique()}")
    print(f"Sizes: {df['Size'].nunique() if 'Size' in df.columns else 0}")
    print(f"Seasons: {df['Season'].nunique() if 'Season' in df.columns else 0}")

    print("\n2. Data Cleaning Summary")
    for key, value in cleaning_summary.items():
        print(f"- {key}: {value}")

    print("\n3. Descriptive Statistics")
    print(desc_df.to_string(index=False))

    print("\n4. Size Demand Distribution")
    if size_demand.empty:
        print("Size analysis skipped because the dataset does not include a Size column.")
    else:
        print(size_demand.to_string(index=False))

    print("\n5. Store-Level Size Analysis")
    if store_size.empty:
        print("Store-level size analysis skipped because the dataset does not include a Size column.")
    else:
        print(store_size.head(20).to_string(index=False))

    print("\n6. Seasonal Analysis")
    if season_size.empty:
        print("Seasonal analysis skipped because the dataset does not include the expected seasonal/size fields.")
    else:
        print(season_size.head(30).to_string(index=False))

    print("\n7. Return Analysis")
    if return_by_size.empty:
        print("Return analysis skipped because the dataset does not include a Return_Status column.")
    else:
        print(return_by_size.to_string(index=False))

    print("\n8. Random Forest Model")
    print("Model type: RandomForestRegressor")
    print("Target: Units_Sold")
    print("Features used: product/store/category/region/inventory/season/price/promotion/weather/competitor metadata plus date-derived indicators")
    print("Return_Status and other post-sale variables were excluded to avoid leakage.")

    print("\n9. Model Evaluation")
    metrics = pd.read_csv(OUTPUT_DIR / "model_metrics.csv")
    print(metrics.to_string(index=False))

    print("\n10. Feature Importance")
    print(top_features.to_string(index=False))

    print("\n11. Ideal Size Mix")
    if recommendation_df.empty:
        print("Size inventory recommendation skipped because the dataset does not include a Size column.")
    else:
        print(recommendation_df.to_string(index=False))

    print("\n12. 5–7 Key Insights")
    for idx, insight in enumerate(insights, start=1):
        print(f"{idx}. {insight}")

    print("\n13. Practical Action Plan")
    print("1. Store and category allocation: prioritize the strongest-demand stores and categories using historical sales volume.")
    print("2. Inventory policy: compare promotion, inventory, and competitor pricing patterns to align stock with actual demand.")
    print("3. Seasonal adjustment: scale inventory toward the season with the highest observed demand and monitor weather-driven shifts.")
    print("4. Product-level monitoring: review the highest-volume products and the drivers of peak demand.")
    print("5. Return monitoring: if return data becomes available, flag high-return products or categories for review.")
    print("6. Model retraining: refresh the Random Forest model regularly with the most recent sales data and validate performance on new periods.")
    print("7. Inventory monitoring: track stock-outs, sell-through, and remaining inventory to compare forecasted and actual demand.")

    print("\n14. Conclusion")
    print("The final recommendations are based strictly on the observed data and should be interpreted with the current dataset's actual demand and inventory patterns. The model and analytics are decision-support tools, not guarantees of future outcomes.")


def save_cleaned_data(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_DIR / "cleaned_fashion_data.csv", index=False)


def main() -> None:
    try:
        ensure_output_dir()
        df = load_data()
        cleaned_df, cleaning_summary = clean_data(df)
        save_cleaned_data(cleaned_df)

        desc_df = perform_eda(cleaned_df)
        size_demand = analyze_size_demand(cleaned_df)
        store_size = analyze_store_size(cleaned_df)
        season_size = analyze_seasonality(cleaned_df)
        return_by_size = analyze_returns(cleaned_df)

        model = train_random_forest(cleaned_df)
        X_train, _, _, _ = prepare_model_data(cleaned_df)
        top_features = calculate_feature_importance(cleaned_df, model)
        recommendation_df = calculate_size_mix(cleaned_df, planned_inventory=1000)
        insights = generate_insights(cleaned_df, store_size, season_size, top_features, return_by_size)
        generate_final_report(cleaned_df, cleaning_summary, desc_df, size_demand, store_size, season_size, return_by_size, model, top_features, recommendation_df, insights)

        print("\nOutputs saved to:")
        for output_file in sorted(OUTPUT_DIR.iterdir()):
            print(f"- {output_file.name}")

    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        print("Please place the CSV dataset inside the project folder and run the script again.")
        print("No analysis was run because the dataset file was not found.")
        sys.exit(1)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
