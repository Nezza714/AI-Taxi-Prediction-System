# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset

file_path = 'C:/Users/neary/Taxi_System/yellow_tripdata_2024-02.parquet'
df = pd.read_parquet(file_path)
print(" Dataset loaded! Shape:", df.shape)

# Data Cleaning

# Drop rows with missing critical values
df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'passenger_count', 'fare_amount'])

# Calculate trip duration (in seconds)
df['trip_duration'] = (pd.to_datetime(df['tpep_dropoff_datetime']) - pd.to_datetime(df['tpep_pickup_datetime'])).dt.total_seconds()

# Filter for realistic trips
df = df[
    (df['trip_duration'] > 60) & (df['trip_duration'] <= 7200) &
    (df['trip_distance'] > 0) & (df['trip_distance'] < 100) &
    (df['passenger_count'] > 0) &
    (df['fare_amount'] > 0)
]
print(" After cleaning, shape:", df.shape)

# Feature Engineering

df['pickup_hour'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
df['pickup_day_of_week'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.dayofweek
print(" Feature engineering complete.")

# Define Features and Target

features = ['trip_distance', 'passenger_count', 'pickup_hour', 'pickup_day_of_week']
target = 'trip_duration'

X = df[features]
y = df[target]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(" Data split complete.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Train Random Forest Regressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

print(" Training Random Forest Regressor...")
model.fit(X_train, y_train)
print(" Model training complete.")

# Make Predictions and Evaluate

y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f} seconds")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} seconds")
print(f"R² Score: {r2:.4f}")

# Visualization 

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.2)
plt.xlabel('True Trip Duration (seconds)')
plt.ylabel('Predicted Trip Duration (seconds)')
plt.title('True vs Predicted Trip Duration')
plt.grid(True)
plt.show()
