# =============================================================================
# House Price Prediction System
# Using Random Forest Regressor on a Synthetic Dataset
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Set random seed for reproducibility ──────────────────────────────────────
SEED = 42
np.random.seed(SEED)

print("=" * 60)
print("   HOUSE PRICE PREDICTION SYSTEM")
print("   Model: Random Forest Regressor")
print("=" * 60)


# =============================================================================
# 1. GENERATE SYNTHETIC DATASET
# =============================================================================
print("\n[1] Generating synthetic dataset...")

N = 1000  # Number of houses

square_footage  = np.random.randint(500,   5000, N)
bedrooms        = np.random.randint(1,     7,    N)
location_score  = np.random.uniform(1,    10,    N).round(2)   # 1–10 desirability
land_value      = np.random.randint(10_000, 300_000, N)        # USD
age_of_house    = np.random.randint(1,    100,   N)             # years

# Realistic price formula + Gaussian noise
price = (
    square_footage  * 150
    + bedrooms      * 8_000
    + location_score * 20_000
    + land_value    * 0.5
    - age_of_house  * 500
    + np.random.normal(0, 20_000, N)          # noise
).clip(50_000)                                # floor at $50 k

df = pd.DataFrame({
    "SquareFootage":  square_footage,
    "Bedrooms":       bedrooms,
    "LocationScore":  location_score,
    "LandValue":      land_value,
    "AgeOfHouse":     age_of_house,
    "Price":          price.astype(int),
})

print(f"   Dataset shape : {df.shape}")
print(f"   Price range   : ${df['Price'].min():,} – ${df['Price'].max():,}")
print(f"   Mean price    : ${df['Price'].mean():,.0f}")


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n[2] Exploratory Data Analysis")
print(df.describe().to_string())

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()
print("   Saved → correlation_heatmap.png")

# Price distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["Price"], bins=40, kde=True, color="steelblue")
plt.title("House Price Distribution")
plt.xlabel("Price (USD)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("price_distribution.png", dpi=150)
plt.close()
print("   Saved → price_distribution.png")


# =============================================================================
# 3. PREPARE FEATURES & TARGET
# =============================================================================
print("\n[3] Preparing features and target variable...")

FEATURES = ["SquareFootage", "Bedrooms", "LocationScore", "LandValue", "AgeOfHouse"]
TARGET   = "Price"

X = df[FEATURES]
y = df[TARGET]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")


# =============================================================================
# 4. TRAIN RANDOM FOREST REGRESSOR
# =============================================================================
print("\n[4] Training Random Forest Regressor...")

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
print("   Model training complete.")


# =============================================================================
# 5. EVALUATE THE MODEL
# =============================================================================
print("\n[5] Evaluating model performance...")

y_pred = rf_model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n   ── Test Set Metrics ────────────────────────")
print(f"   MAE   : ${mae:>12,.2f}")
print(f"   MSE   : ${mse:>12,.2f}")
print(f"   RMSE  : ${rmse:>12,.2f}")
print(f"   R²    : {r2:>13.4f}")
print(f"   MAPE  : {mape:>12.2f}%")
print("   ─────────────────────────────────────────────")

# Cross-validation R² (5-fold)
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="r2")
print(f"\n   5-Fold CV R²  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================
print("\n[6] Feature importances...")

importance_df = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": rf_model.feature_importances_,
}).sort_values("Importance", ascending=False)

print(importance_df.to_string(index=False))

plt.figure(figsize=(7, 4))
sns.barplot(data=importance_df, x="Importance", y="Feature",
            palette="viridis", orient="h")
plt.title("Random Forest – Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("   Saved → feature_importance.png")


# =============================================================================
# 7. ACTUAL vs PREDICTED PLOT
# =============================================================================
results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})

plt.figure(figsize=(7, 6))
plt.scatter(results_df["Actual"], results_df["Predicted"],
            alpha=0.4, color="steelblue", edgecolors="none", s=25)
lims = [results_df.min().min(), results_df.max().max()]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price (USD)")
plt.ylabel("Predicted Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()
print("   Saved → actual_vs_predicted.png")

# Residuals
residuals = y_test.values - y_pred
plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals, alpha=0.35, color="coral", s=25)
plt.axhline(0, color="black", linewidth=1.2)
plt.title("Residual Plot")
plt.xlabel("Predicted Price (USD)")
plt.ylabel("Residual (Actual – Predicted)")
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150)
plt.close()
print("   Saved → residual_plot.png")


# =============================================================================
# 8. SAVE RESULTS
# =============================================================================
results_df.to_csv("prediction_results.csv", index=False)
df.to_csv("synthetic_dataset.csv", index=False)
print("\n   Saved → prediction_results.csv")
print("   Saved → synthetic_dataset.csv")


# =============================================================================
# 9. SAMPLE PREDICTIONS
# =============================================================================
print("\n[9] Sample predictions on new data")

new_houses = pd.DataFrame({
    "SquareFootage": [1200, 2500, 4000],
    "Bedrooms":      [2,    4,    5   ],
    "LocationScore": [5.5,  7.8,  9.2 ],
    "LandValue":     [50_000, 120_000, 250_000],
    "AgeOfHouse":    [20,   10,    3  ],
})

new_predictions = rf_model.predict(new_houses)
new_houses["PredictedPrice"] = new_predictions.astype(int)
print(new_houses.to_string(index=False))

print("\n" + "=" * 60)
print("   Pipeline complete — all outputs saved.")
print("=" * 60)
