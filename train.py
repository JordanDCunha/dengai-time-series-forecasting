# ======================
# Imports
# ======================
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

# ======================
# Load Data
# ======================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ======================
# Preprocessing
# ======================
for df in [train, test]:
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    df.sort_values(["city", "week_start_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

# Fill missing values
train = train.groupby("city").apply(lambda x: x.ffill()).reset_index(drop=True)
test = test.groupby("city").apply(lambda x: x.ffill()).reset_index(drop=True)

# Tag datasets
train["dataset"] = "train"
test["dataset"] = "test"

# Add placeholder target
test["total_cases"] = np.nan

# ======================
# Combine
# ======================
full = pd.concat([train, test], axis=0).reset_index(drop=True)
full = full.sort_values(["city", "week_start_date"])

# ======================
# Feature Engineering
# ======================

# Seasonal features
full["weekofyear"] = full["week_start_date"].dt.isocalendar().week.astype(int)
full["week_sin"] = np.sin(2 * np.pi * full["weekofyear"] / 52)
full["week_cos"] = np.cos(2 * np.pi * full["weekofyear"] / 52)

# Lag features
lags = [1, 2, 4, 8]
for lag in lags:
    full[f"cases_lag_{lag}"] = full.groupby("city")["total_cases"].shift(lag)

# Rolling features
full["cases_roll_mean_4"] = full.groupby("city")["total_cases"].shift(1).rolling(4).mean()
full["cases_roll_std_4"] = full.groupby("city")["total_cases"].shift(1).rolling(4).std()

# ======================
# Split back
# ======================
train_processed = full[full["dataset"] == "train"].copy()
test_processed = full[full["dataset"] == "test"].copy()

features_to_drop = ["id", "city", "year", "weekofyear", "week_start_date", "total_cases", "dataset"]

models = {}
preds_list = []
cv_mae_list = []

# ======================
# Training Loop
# ======================
for city in train_processed["city"].unique():

    print(f"\nTraining for city: {city}")

    train_city = train_processed[train_processed["city"] == city]
    test_city = test_processed[test_processed["city"] == city]

    X_train = train_city.drop(columns=features_to_drop, errors="ignore")
    y_train = train_city["total_cases"]

    X_test = test_city.drop(columns=features_to_drop, errors="ignore")

    # Fill missing values
    X_train = X_train.ffill().bfill()
    X_test = X_test.ffill().bfill()

    # ======================
    # Time Series CV
    # ======================
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        temp_model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

        temp_model.fit(X_tr, y_tr)
        preds_val = temp_model.predict(X_val)

        mae = mean_absolute_error(y_val, preds_val)
        mae_scores.append(mae)

    cv_mae = np.mean(mae_scores)
    cv_mae_list.append(cv_mae)

    print(f"{city} CV MAE: {cv_mae:.2f}")

    # ======================
    # Final Model
    # ======================
    final_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    final_model.fit(X_train, y_train)

    preds = final_model.predict(X_test)

    temp = pd.DataFrame({
        "id": test_city["id"],
        "total_cases": preds
    })

    preds_list.append(temp)
    models[city] = final_model

# ======================
# Final Score
# ======================
final_mae = np.mean(cv_mae_list)
print("\nOverall CV MAE:", final_mae)

# ======================
# Submission
# ======================
submission = pd.concat(preds_list)
submission["total_cases"] = submission["total_cases"].round().astype(int).clip(lower=0)
submission = submission.sort_values("id")

submission.to_csv("submission.csv", index=False)

print("\nSubmission file saved!")
