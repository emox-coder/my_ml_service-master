# House Price Prediction Model Card

## Summary
- Task: house price regression (prices in PHP)
- Best model: HistGradientBoosting
- Trained at (UTC): 2026-05-20T13:11:23.078787+00:00

## Features Used
Location, FloorAreaSqm, LotSizeSqm, Bedrooms, Bathrooms, ParkingSpaces, PropertyAge, Condition, PropertyType, NeighborhoodRating, DistanceToCBDKm

## Evaluation
- Cross-validation (5-fold) MAE: 3,457,536.29 ± 176,309.93
- Cross-validation (5-fold) R²: 0.9542 ± 0.0140
- Holdout test MAE: 2,953,969.77
- Holdout test R²: 0.9618

## Notes / Limitations
- If you use synthetic data, accuracy reflects the synthetic assumptions. For real accuracy, train on real PH listings.
- Always validate units (sqm vs sqft) and location granularity (city/barangay).
