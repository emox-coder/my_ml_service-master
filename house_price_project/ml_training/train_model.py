"""
Machine Learning Training Script for House Price Prediction.

This version is "defense-ready" (more professional + more accurate):
- Adds more PH-relevant features (lot size, condition, property type, neighborhood rating, distance to CBD).
- Uses proper categorical encoding (OneHotEncoder, NOT LabelEncoder).
- Derives PropertyAge from YearBuilt.
- Trains stronger models (Ridge / RandomForest / HistGradientBoosting).
- Uses cross-validation (5-fold) + a holdout test split.
- Saves a single end-to-end scikit-learn Pipeline (preprocessing + model) to ml_models/model.pkl
- Saves metrics + model metadata to ml_models/model_metrics.json

Usage:
  python ml_training/train_model.py
  python ml_training/train_model.py --data data/house_prices.csv
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PRICE_CURRENCY = "PHP"

# Canonical columns used by the model (inputs + target).
# Your CSV can use common variants; `normalize_columns()` maps them.
FEATURE_COLUMNS = [
    "Location",
    "FloorAreaSqm",
    "LotSizeSqm",
    "Bedrooms",
    "Bathrooms",
    "ParkingSpaces",
    "PropertyAge",
    "Condition",
    "PropertyType",
    "NeighborhoodRating",
    "DistanceToCBDKm",
]
TARGET_COLUMN = "Price"


def _project_path(*parts: str) -> str:
    # ml_training/.. = house_price_project/
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", *parts))


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic house price data (kept to make the project runnable out-of-the-box).
    Replace this by providing your own CSV via --data.
    """
    rng = np.random.default_rng(42)

    # Philippines-focused locations (also used in the UI dropdown)
    locations = np.array(
        [
            "Metro Manila",
            "Cebu",
            "Davao",
            "Baguio",
            "Iloilo",
            "Bacolod",
            "Cagayan de Oro",
            "General Santos",
        ]
    )
    location = rng.choice(locations, size=n_samples, replace=True)

    # Basic property features
    floor_area_sqm = rng.integers(20, 450, size=n_samples)  # condos to large houses
    lot_size_sqm = rng.integers(0, 800, size=n_samples)  # condos may be 0
    bedrooms = rng.integers(0, 7, size=n_samples)
    bathrooms = rng.integers(1, 6, size=n_samples)
    parking_spaces = rng.integers(0, 5, size=n_samples)
    year_built = rng.integers(1950, 2026, size=n_samples)
    property_age = 2026 - year_built

    property_types = np.array(["Condo", "House & Lot", "Townhouse"])
    conditions = np.array(["New", "Good", "Needs Repair"])
    property_type = rng.choice(property_types, size=n_samples, replace=True, p=[0.45, 0.45, 0.10])
    condition = rng.choice(conditions, size=n_samples, replace=True, p=[0.25, 0.60, 0.15])

    # Neighborhood / amenities proxies (synthetic)
    neighborhood_rating = rng.integers(3, 11, size=n_samples)  # 3..10
    distance_to_cbd_km = np.clip(rng.normal(8, 6, size=n_samples), 0.1, 40)

    # Prices are in PHP (₱). Tune these coefficients to match your market.
    base_price = 900_000
    location_multiplier = {
        "Metro Manila": 2.6,
        "Cebu": 1.9,
        "Davao": 1.7,
        "Baguio": 1.8,
        "Iloilo": 1.4,
        "Bacolod": 1.3,
        "Cagayan de Oro": 1.25,
        "General Santos": 1.15,
    }
    loc_mult = np.vectorize(location_multiplier.get)(location).astype(float)

    type_multiplier = {"Condo": 1.2, "House & Lot": 1.35, "Townhouse": 1.15}
    cond_multiplier = {"New": 1.15, "Good": 1.0, "Needs Repair": 0.85}
    t_mult = np.vectorize(type_multiplier.get)(property_type).astype(float)
    c_mult = np.vectorize(cond_multiplier.get)(condition).astype(float)

    price = (
        base_price
        + (floor_area_sqm * 55_000)
        + (lot_size_sqm * 22_000)
        + (bedrooms * 320_000)
        + (bathrooms * 240_000)
        + (parking_spaces * 180_000)
        + (neighborhood_rating * 180_000)
        - (distance_to_cbd_km * 28_000)
        - (property_age * 12_000)
    ) * loc_mult * t_mult * c_mult
    price = price + rng.normal(0, 450_000, size=n_samples)
    price = np.maximum(price, 450_000)

    df = pd.DataFrame(
        {
            "Location": location,
            "FloorAreaSqm": floor_area_sqm,
            "LotSizeSqm": lot_size_sqm,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "ParkingSpaces": parking_spaces,
            "YearBuilt": year_built,
            "PropertyAge": property_age,
            "Condition": condition,
            "PropertyType": property_type,
            "NeighborhoodRating": neighborhood_rating,
            "DistanceToCBDKm": distance_to_cbd_km,
            "Price": price,
        }
    )

    # Inject a few missing values so the imputer logic is exercised.
    for col in ["Location", "FloorAreaSqm", "Bathrooms", "Condition", "LotSizeSqm"]:
        missing_idx = rng.choice(df.index, size=max(1, n_samples // 100), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts common column naming variants and normalizes to canonical columns.
    Missing optional columns are created with NaN (they will be imputed).
    """
    normalized = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "").replace("_", "")
        normalized[key] = col

    def pick(*candidates: str) -> str | None:
        for c in candidates:
            if c in normalized:
                return normalized[c]
        return None

    # Minimum needed to train a usable model.
    required_mapping = {
        "Location": pick("location", "city", "municipality"),
        "FloorAreaSqm": pick("floorareasqm", "floorarea", "area", "squarefeet", "squarefootage", "sqft"),
        "Bedrooms": pick("bedrooms", "beds"),
        "Bathrooms": pick("bathrooms", "baths"),
        "ParkingSpaces": pick("parkingspaces", "parking", "garage"),
        "Price": pick("price", "pricephp", "target"),
    }

    missing_required = [k for k, v in required_mapping.items() if v is None]
    if missing_required:
        raise ValueError(
            "Dataset is missing required columns. "
            f"Missing: {', '.join(missing_required)}. "
            "Required minimum: Location, FloorAreaSqm (or SquareFeet), Bedrooms, Bathrooms, ParkingSpaces, Price."
        )

    optional_mapping = {
        "LotSizeSqm": pick("lotsizesqm", "lotsize", "lot"),
        "YearBuilt": pick("yearbuilt", "built", "year"),
        "PropertyAge": pick("propertyage", "age"),
        "Condition": pick("condition", "housecondition"),
        "PropertyType": pick("propertytype", "type"),
        "NeighborhoodRating": pick("neighborhoodrating", "rating", "neighborhoodscore"),
        "DistanceToCBDKm": pick("distancetocbdkm", "distancetocbd", "distancecbd", "distance"),
    }

    out = pd.DataFrame()
    # Required
    for canonical, source in required_mapping.items():
        out[canonical] = df[source]
    # Optional (create if missing)
    for canonical, source in optional_mapping.items():
        out[canonical] = df[source] if source is not None else np.nan

    # Derive PropertyAge if we have YearBuilt
    if out["PropertyAge"].isna().all() and not out["YearBuilt"].isna().all():
        current_year = datetime.now().year
        out["PropertyAge"] = current_year - pd.to_numeric(out["YearBuilt"], errors="coerce")

    # Keep only expected columns
    cols = ["Location"] + [c for c in FEATURE_COLUMNS if c != "Location"] + [TARGET_COLUMN]
    out = out[cols]
    return out


@dataclass(frozen=True)
class ModelResult:
    name: str
    pipeline: Pipeline
    # Cross-validation (mean ± std)
    cv_r2_mean: float
    cv_r2_std: float
    cv_mae_mean: float
    cv_mae_std: float
    # Holdout test
    test_r2: float
    test_mae: float


def build_preprocessor() -> ColumnTransformer:
    categorical_features = ["Location", "Condition", "PropertyType"]
    numeric_features = [
        "FloorAreaSqm",
        "LotSizeSqm",
        "Bedrooms",
        "Bathrooms",
        "ParkingSpaces",
        "PropertyAge",
        "NeighborhoodRating",
        "DistanceToCBDKm",
    ]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features),
        ]
    )


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list[ModelResult]:
    preprocessor = build_preprocessor()

    candidates = [
        ("Ridge", Ridge(alpha=1.0, random_state=42)),
        ("Random Forest", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
        ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=42)),
    ]

    results: list[ModelResult] = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, regressor in candidates:
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", regressor),
            ]
        )

        # Cross-validation on the training split
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=-1,
            error_score="raise",
        )
        cv_mae = -scores["test_mae"]
        cv_r2 = scores["test_r2"]

        # Fit on full training and evaluate on holdout test
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        test_mae = float(mean_absolute_error(y_test, preds))
        test_r2 = float(r2_score(y_test, preds))

        results.append(
            ModelResult(
                name=name,
                pipeline=pipe,
                cv_r2_mean=float(np.mean(cv_r2)),
                cv_r2_std=float(np.std(cv_r2)),
                cv_mae_mean=float(np.mean(cv_mae)),
                cv_mae_std=float(np.std(cv_mae)),
                test_r2=test_r2,
                test_mae=test_mae,
            )
        )
    return results


def save_artifacts(best: ModelResult, model_path: str, metrics_path: str, card_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best.pipeline, model_path)
    now = datetime.now(timezone.utc).isoformat()
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best.name,
                "currency": PRICE_CURRENCY,
                "trained_at_utc": now,
                "feature_columns": FEATURE_COLUMNS,
                "cv": {
                    "r2_mean": best.cv_r2_mean,
                    "r2_std": best.cv_r2_std,
                    "mae_mean": best.cv_mae_mean,
                    "mae_std": best.cv_mae_std,
                },
                "holdout_test": {"r2": best.test_r2, "mae": best.test_mae},
            },
            f,
            indent=2,
        )

    # Simple "model card" for defense / documentation.
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(
            f"""# House Price Prediction Model Card

## Summary
- Task: house price regression (prices in {PRICE_CURRENCY})
- Best model: {best.name}
- Trained at (UTC): {now}

## Features Used
{", ".join(FEATURE_COLUMNS)}

## Evaluation
- Cross-validation (5-fold) MAE: {best.cv_mae_mean:,.2f} ± {best.cv_mae_std:,.2f}
- Cross-validation (5-fold) R²: {best.cv_r2_mean:.4f} ± {best.cv_r2_std:.4f}
- Holdout test MAE: {best.test_mae:,.2f}
- Holdout test R²: {best.test_r2:.4f}

## Notes / Limitations
- If you use synthetic data, accuracy reflects the synthetic assumptions. For real accuracy, train on real PH listings.
- Always validate units (sqm vs sqft) and location granularity (city/barangay).
"""
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a house price regression model.")
    parser.add_argument(
        "--data",
        default=_project_path("data", "house_prices.csv"),
        help="Path to CSV dataset. If missing, a sample dataset will be generated at this path.",
    )
    parser.add_argument(
        "--overwrite-sample",
        action="store_true",
        help="Overwrite the dataset with a newly generated Philippines/PHP sample dataset.",
    )
    parser.add_argument(
        "--model-out",
        default=_project_path("ml_models", "model.pkl"),
        help="Path to write the trained model pipeline (joblib .pkl).",
    )
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.data)
    model_path = os.path.abspath(args.model_out)
    metrics_path = os.path.join(os.path.dirname(model_path), "model_metrics.json")
    card_path = os.path.join(os.path.dirname(model_path), "model_card.md")

    print("=" * 60)
    print("HOUSE PRICE PREDICTION - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Model output: {model_path}")
    print()

    if os.path.exists(dataset_path) and not args.overwrite_sample:
        df_raw = pd.read_csv(dataset_path)
        print(f"Loaded dataset with shape: {df_raw.shape}")
    else:
        print("Generating sample dataset...")
        df_raw = generate_sample_data(n_samples=1500)
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        df_raw.to_csv(dataset_path, index=False)
        print(f"Sample dataset saved to: {dataset_path}")

    df = normalize_columns(df_raw)

    # Split X/y
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Train & evaluate
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    # Select best by CV MAE (lower is better); tie-break on CV R² (higher better)
    results_sorted = sorted(results, key=lambda r: (r.cv_mae_mean, -r.cv_r2_mean))

    print("\n" + "-" * 60)
    print("MODEL EVALUATION (CV is 5-fold; lower MAE is better)")
    print("-" * 60)
    for r in results_sorted:
        print(
            f"{r.name:>18} | CV MAE: {r.cv_mae_mean:,.2f} ± {r.cv_mae_std:,.2f} "
            f"| CV R²: {r.cv_r2_mean:.4f} ± {r.cv_r2_std:.4f} "
            f"| Test MAE: {r.test_mae:,.2f} | Test R²: {r.test_r2:.4f}"
        )

    best = results_sorted[0]
    print("\n" + "=" * 60)
    print(
        f"BEST MODEL: {best.name} | CV MAE: {best.cv_mae_mean:,.2f} | CV R²: {best.cv_r2_mean:.4f}"
    )
    print("=" * 60)

    save_artifacts(best, model_path, metrics_path, card_path)
    print(f"\nSaved: {model_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {card_path}")


if __name__ == "__main__":
    main()
