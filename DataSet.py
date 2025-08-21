from __future__ import annotations

# ===== Imports =====
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ===== Configuration =====
DEFAULT_FILENAME = "superstore_sample.csv"
FALLBACK_FILENAME = "superstore_sample_generated.csv"
FORECAST_STEPS = 6  # months
RANDOM_STATE = 42

# ===== Utility / Validation =====
REQUIRED_COLUMNS = {"Order Date", "Category", "Region", "Sales", "Profit"}


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"


def generate_sample_dataset(path: Path, n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a small Superstore-like dataset and save to `path`."""
    rng = np.random.default_rng(seed)
    categories = ["Furniture", "Office Supplies", "Technology"]
    regions = ["East", "West", "Central", "South"]
    df = pd.DataFrame({
        "Order Date": pd.date_range(start="2021-01-01", periods=n, freq="7D"),
        "Category": rng.choice(categories, n),
        "Region": rng.choice(regions, n),
        "Sales": np.round(rng.uniform(50, 1000, n), 2),
        "Profit": np.round(rng.uniform(-100, 300, n), 2),
    })
    df.to_csv(path, index=False)
    return df


def possible_paths(cli_arg: Optional[str]) -> List[Path]:
    """Return a list of candidate dataset paths to try in order."""
    candidates: List[Path] = []
    # 1) CLI arg
    if cli_arg:
        candidates.append(Path(cli_arg))
    # 2) ENV var
    env_path = os.environ.get("DATASET_PATH")
    if env_path:
        candidates.append(Path(env_path))
    # 3) Common locations
    candidates.extend([
        Path(DEFAULT_FILENAME),
        Path("/mnt/data") / DEFAULT_FILENAME,
        Path.cwd() / DEFAULT_FILENAME,
    ])
    # 4) Fallback (if previously generated)
    candidates.append(Path(FALLBACK_FILENAME))
    # Deduplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def resolve_or_create_dataset(cli_arg: Optional[str]) -> Tuple[pd.DataFrame, Path]:
    """Try to resolve a dataset path. If none found, generate one and return it."""
    for p in possible_paths(cli_arg):
        if p.exists():
            df = load_data(p)
            return df, p
    # Nothing found → generate
    gen_path = Path.cwd() / FALLBACK_FILENAME
    print(f"⚠️  Dataset not found. Generating a sample at: {gen_path.resolve()}")
    df = generate_sample_dataset(gen_path, seed=RANDOM_STATE)
    return df, gen_path


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")
    df = pd.read_csv(path)
    # Parse dates safely
    if "Order Date" not in df.columns:
        raise KeyError("'Order Date' column is required in the dataset")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    # Drop rows with invalid dates or sales
    df = df.dropna(subset=["Order Date", "Sales"]).copy()
    # Ensure numeric types
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df["Profit"] = pd.to_numeric(df.get("Profit"), errors="coerce")
    df = df.dropna(subset=["Sales"]).copy()
    return df


# ===== EDA =====

def plot_category_sales(df: pd.DataFrame) -> pd.Series:
    category_sales = (
        df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    )
    plt.figure()
    sns.barplot(x=category_sales.index, y=category_sales.values)
    plt.title("Total Sales by Category")
    plt.xlabel("Category")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.show()
    return category_sales


def plot_region_profit(df: pd.DataFrame) -> pd.Series:
    region_profit = (
        df.groupby("Region")["Profit"].sum().sort_values(ascending=False)
    )
    plt.figure()
    sns.barplot(x=region_profit.index, y=region_profit.values)
    plt.title("Total Profit by Region")
    plt.xlabel("Region")
    plt.ylabel("Profit")
    plt.tight_layout()
    plt.show()
    return region_profit


def monthly_sales_series(df: pd.DataFrame) -> pd.Series:
    monthly_sales = df.set_index("Order Date").sort_index().resample("M")["Sales"].sum()
    monthly_sales = monthly_sales.dropna()
    assert isinstance(monthly_sales.index, pd.DatetimeIndex)
    assert len(monthly_sales) > 0, "No monthly sales could be computed — check dates."
    return monthly_sales


def plot_monthly_sales(monthly_sales: pd.Series) -> None:
    plt.figure()
    plt.plot(monthly_sales, marker="o")
    plt.title("Monthly Sales Over Time")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.show()


# ===== Forecasting =====

def fit_best_arima(y: pd.Series, candidates: Optional[Iterable[Tuple[int,int,int]]] = None):
    if candidates is None:
        candidates = [
            (2, 1, 2), (1, 1, 1), (1, 1, 0), (0, 1, 1), (2, 1, 1), (1, 1, 2)
        ]
    best_res = None
    best_order = None
    best_aic = np.inf
    for order in candidates:
        try:
            res = ARIMA(y, order=order).fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_res = res
                best_order = order
        except Exception:
            continue
    return best_res, best_order


def _coerce_forecast_index(fc: pd.Series, history: pd.Series) -> pd.Series:
    # Ensure forecast has a DatetimeIndex continuing from the last observed month
    if not isinstance(fc.index, pd.DatetimeIndex):
        start = history.index[-1] + pd.offsets.MonthEnd(1)
        idx = pd.date_range(start=start, periods=len(fc), freq="M")
        fc = pd.Series(fc.values, index=idx)
    return fc


def forecast_sales(monthly_sales: pd.Series, steps: int = 6) -> pd.Series:
    model_res, order = fit_best_arima(monthly_sales)
    if model_res is not None:
        fc = model_res.forecast(steps=steps)
    else:
        # Fallback: naive forecast (repeat last observed value)
        last = float(monthly_sales.iloc[-1])
        idx = pd.date_range(monthly_sales.index[-1] + pd.offsets.MonthEnd(1), periods=steps, freq="M")
        fc = pd.Series([last] * steps, index=idx)
        order = ("naive",)
    fc = _coerce_forecast_index(fc, monthly_sales)
    # Plot
    plt.figure()
    plt.plot(monthly_sales, label="History")
    plt.plot(fc.index, fc.values, label=f"Forecast {order}")
    plt.legend()
    plt.title(f"Sales Forecast — Next {steps} Months")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.show()
    return fc


# ===== Business Insights =====

def print_insights(category_sales: pd.Series, region_profit: pd.Series) -> None:
    top_category = category_sales.index[0] if len(category_sales) else "N/A"
    top_region = region_profit.index[0] if len(region_profit) else "N/A"
    print("\nBusiness Insights:")
    print(f"1) Category with highest sales: {top_category}")
    print(f"2) Most profitable region: {top_region}")
    print("3) Use forecast to plan inventory and marketing for upcoming months.")


# ===== Tests (existing + additional) =====

def run_tests(df: pd.DataFrame, monthly_sales: pd.Series, forecast: pd.Series, steps: int) -> None:
    # Existing tests
    ensure_columns(df, REQUIRED_COLUMNS)
    assert pd.api.types.is_datetime64_any_dtype(df["Order Date"]), "Order Date must be datetime"
    assert len(monthly_sales) >= 6, "Expected at least 6 months of data for forecasting"
    assert len(forecast) == steps, f"Forecast must have {steps} steps"
    assert not forecast.isna().any(), "Forecast contains NaNs"
    assert (df.groupby("Category")["Sales"].sum() > 0).any(), "Category sales empty"
    assert df["Profit"].notna().any(), "Profit column has only NaNs"

    # Additional tests
    # 1) Monthly sales are non-negative
    assert (monthly_sales >= 0).all(), "Monthly sales should be non-negative"
    # 2) Forecast index strictly after last history index
    assert isinstance(forecast.index, pd.DatetimeIndex), "Forecast index must be DatetimeIndex"
    assert forecast.index.min() > monthly_sales.index.max(), "Forecast must start after last historical month"
    # 3) Forecast index monotonic
    assert forecast.index.is_monotonic_increasing, "Forecast index must be increasing"
    # 4) Category and Region series not empty
    assert df["Category"].nunique() > 0 and df["Region"].nunique() > 0, "Category/Region should not be empty"
    # 5) Inferred frequency is monthly (best effort)
    inferred = pd.infer_freq(monthly_sales.index)
    assert inferred in ("M", "ME", "MS", None) or inferred and inferred.endswith("M"), "Monthly sales should be monthly frequency"

    print("\n✅ All tests passed.")


# ===== Main =====

def main():
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None
    df, used_path = resolve_or_create_dataset(cli_arg)

    print("Using dataset:", used_path.resolve())
    print("\nDataset info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())

    # EDA
    category_sales = plot_category_sales(df)
    region_profit = plot_region_profit(df)

    # Monthly series
    m_sales = monthly_sales_series(df)
    plot_monthly_sales(m_sales)

    # Forecast
    fc = forecast_sales(m_sales, steps=FORECAST_STEPS)

    # Insights
    print_insights(category_sales, region_profit)

    # Tests
    run_tests(df, m_sales, fc, FORECAST_STEPS)


if __name__ == "__main__":
    main()
