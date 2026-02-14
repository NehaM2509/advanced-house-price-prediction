import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_csv("data/housing.csv")

# Select important numeric columns
data = data[[
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "SalePrice"
]]

# Remove missing values
data = data.dropna()

# Features (X)
X = data.drop("SalePrice", axis=1)

# Target (y)
y = data["SalePrice"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestRegressor(random_state=42)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R2 Score:", score)
print("MSE:", mse)

# Save model and score
joblib.dump(model, "models/house_price_model.pkl")
joblib.dump(score, "models/model_score.pkl")

print("Model and score saved successfully!")
