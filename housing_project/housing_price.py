# =======================
# California Housing Price Prediction
# =======================

# 1. load the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =======================
# 2. load the data
# =======================
california = fetch_california_housing()

# turn to dataframe for easier viewing
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

print("Dataset size：", X.shape)
print("Target length：", len(y))
print("\nFeature names：")
for i, feature in enumerate(california.feature_names):
    print(f"{i+1}. {feature}")
print("\nThe first five rows：\n", X.head())
print("Target samples：", y[:5])

# =======================
# 3. Data preprocess
# =======================
# split the dataset（80%:20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================
# 4. Definite the models
# =======================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}  # save the results

# =======================
# 5. Training & Testing
# =======================
for name, model in models.items():
    # Tips:Tree models do not require standardization. Original data is OK.
    if name == "Random Forest":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    # Calculate MSE 和 R²
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}

# Transform to DataFrame to print
results_df = pd.DataFrame(results).T
print("\nModel performance comparison：\n", results_df)

# =======================
# 6. Visualization - Comparison of Model R² Scores
# =======================
plt.figure(figsize=(8, 5))
results_df["R2"].plot(kind="bar", color="skyblue")
plt.title("Model Comparison (R² Score)")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
for i, v in enumerate(results_df["R2"]):        #show the values above the bar
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
plt.show()

# =======================
# 7. Visualization - Predicted Values vs. Actual Values
#    (Taking Random Forest as an Example)
# =======================
rf = models["Random Forest"]
y_pred_rf = rf.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest: Actual vs Predicted")
plt.show()

# =======================
# 8. Feature importance (random forest)
# =======================
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices], color="orange")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
plt.title("Feature Importances (Random Forest)")
for i, v in enumerate(importances[indices]):
    plt.text(i, v + 0.005, f"{v:.2f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()
