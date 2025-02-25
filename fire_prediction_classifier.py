import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data 
data = pd.read_csv("")

# Extract features from data and convert them to a df.
# Features that should be included: long,lat,humidity,temperature,
# wind speed, vegeation information, temporal (data and time of fires).

# Split the data into training and testing sets
X_train, X_test, y_train_lat, y_test_lat = train_test_split(X, y_lat, test_size=0.2, random_state=42)
X_train, X_test, y_train_lon, y_test_lon = train_test_split(X, y_lon, test_size=0.2, random_state=42)

# Train an XGBoost regressor for latitude
model_lat = xgb.XGBRegressor()
model_lat.fit(X_train, y_train_lat)

# Train an XGBoost regressor for longitude
model_lon = xgb.XGBRegressor()
model_lon.fit(X_train, y_train_lon)

# Predict on the test set
y_pred_lat = model_lat.predict(X_test)
y_pred_lon = model_lon.predict(X_test)

# Evaluate using RMSE
rmse_lat = mean_squared_error(y_test_lat, y_pred_lat, squared=False)
rmse_lon = mean_squared_error(y_test_lon, y_pred_lon, squared=False)
print("Latitude RMSE:", rmse_lat)
print("Longitude RMSE:", rmse_lon)
