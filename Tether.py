import pandas as pd
import numpy as np
import xgboost as xgb
import talib
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta

# Step 1: Load Tether dataset
df = pd.read_csv('Dataset/coin_Tether.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)

# Step 2: Feature Engineering
df['Return'] = df['Close'].pct_change()
df['RollingMean'] = df['Close'].rolling(window=5).mean()
df['RollingStd'] = df['Close'].rolling(window=5).std()
df['Lag1'] = df['Close'].shift(1)
df['Lag2'] = df['Close'].shift(2)
df['Lag3'] = df['Close'].shift(3)
df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
df['MACD'], df['MACDSignal'], _ = talib.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
df['EMA'] = talib.EMA(df['Close'].values, timeperiod=14)
df.dropna(inplace=True)

# Step 3: Log Transformation on Target (Close Price)
df['LogClose'] = np.log(df['Close'])

# Step 4: Prepare features and labels
X = df[['Open', 'High', 'Low', 'Volume', 'Return', 'RollingMean', 'RollingStd',
        'Lag1', 'Lag2', 'Lag3', 'RSI', 'MACD', 'EMA']].values
y = df['LogClose'].values

# Step 5: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Hyperparameter tuning with K-Fold using both XGBoost and manual GBM
param_grid = {
    'n_estimators': [100],
    'learning_rate': [0.1],
    'max_depth': [5],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0]
}

best_score = float('inf')
best_params = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for n_est in param_grid['n_estimators']:
    for lr in param_grid['learning_rate']:
        for depth in param_grid['max_depth']:
            for subsample in param_grid['subsample']:
                for colsample in param_grid['colsample_bytree']:
                    for gamma in param_grid['gamma']:
                        xgb_fold_mse = []
                        gbm_fold_mse = []
                        for train_idx, val_idx in kf.split(X_scaled):
                            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]

                            # XGBoost model
                            xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                                                         n_estimators=n_est,
                                                         learning_rate=lr,
                                                         max_depth=depth,
                                                         subsample=subsample,
                                                         colsample_bytree=colsample,
                                                         gamma=gamma,
                                                         random_state=42)
                            xgb_model.fit(X_train, y_train)
                            y_pred_xgb = xgb_model.predict(X_val)
                            xgb_fold_mse.append(np.mean((y_val - y_pred_xgb) ** 2))

                            # Manual GBM model
                            gbm_model = GradientBoostingRegressor(n_estimators=n_est,
                                                                  learning_rate=lr,
                                                                  max_depth=depth,
                                                                  subsample=subsample,
                                                                  random_state=42)
                            gbm_model.fit(X_train, y_train)
                            y_pred_gbm = gbm_model.predict(X_val)
                            gbm_fold_mse.append(np.mean((y_val - y_pred_gbm) ** 2))

                        avg_xgb_mse = np.mean(xgb_fold_mse)
                        avg_gbm_mse = np.mean(gbm_fold_mse)
                        
                        # Combine scores for hyperparameter selection
                        combined_score = (avg_xgb_mse + avg_gbm_mse) / 2

                        if combined_score < best_score:
                            best_score = combined_score
                            best_params = {
                                'n_estimators': n_est,
                                'learning_rate': lr,
                                'max_depth': depth,
                                'subsample': subsample,
                                'colsample_bytree': colsample,
                                'gamma': gamma
                            }

# Step 7: Train Final XGBoost Model with best params
final_model = xgb.XGBRegressor(objective='reg:squarederror',
                               **best_params,
                               random_state=42)
final_model.fit(X_scaled, y)

# Step 8: Predict on Training Data
y_pred = final_model.predict(X_scaled)

# Step 9: Evaluation
mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y - y_pred))
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\nEvaluation Metrics:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R²):", r2)

# Step 10: Inverse Log Transform
y_pred_actual = np.exp(y_pred)
y_actual = np.exp(y)

# Step 11: DataFrame of Actual vs Predicted
predicted_df = pd.DataFrame({
    'Actual': y_actual.ravel(),
    'Predicted': y_pred_actual.ravel()
}, index=df.index)
print("\nSample Predictions:")
print(predicted_df.tail())

# Step 12: Plot Actual vs Predicted
plt.figure(figsize=(14, 6))
plt.plot(df.index, y_actual, label='Actual Price', color='blue')
plt.plot(df.index, y_pred_actual, label='Predicted Price', color='orange')

# Step 13: Save the Model
os.makedirs('models', exist_ok=True)
joblib.dump(final_model, 'models/best_xgboost_tether_model.pkl')
print("\nModel saved to models/best_xgboost_tether_model.pkl")

# Step 14: Predict future prices dynamically updating lag and rolling features

last_row = df.iloc[-1].copy()
current_date = datetime(2025, 5, 15)

# Initialize rolling window with last 5 close prices (for rolling mean/std)
rolling_window = list(df['Close'].tail(5))
# Initialize lag variables
lag1 = last_row['Close']
lag2 = df['Close'].iloc[-2] if len(df) > 1 else lag1
lag3 = df['Close'].iloc[-3] if len(df) > 2 else lag2

# For simplicity keep RSI, MACD, EMA, Volume, Open, High, Low constant as last known values
rsi = last_row['RSI']
macd = last_row['MACD']
ema = last_row['EMA']
volume = last_row['Volume']
open_ = last_row['Open']
high_ = last_row['High']
low_ = last_row['Low']

future_days = [1, 7, 30, 365]
simulated_dates = []
simulated_prices = []

prev_close = lag1

for day in range(1, max(future_days) + 1):
    # Calculate features based on predicted price history
    
    # Return from previous day (use simple return from prev_close to lag1)
    ret = (prev_close - lag1) / lag1 if lag1 != 0 else 0
    
    # Rolling mean and std based on rolling_window plus new predicted price
    rolling_window.append(prev_close)
    if len(rolling_window) > 5:
        rolling_window.pop(0)
    rolling_mean = np.mean(rolling_window)
    rolling_std = np.std(rolling_window)
    
    # Update lag variables for this prediction
    # lag1 is previous close, lag2 previous day of lag1, lag3 day before lag2
    lag3 = lag2
    lag2 = lag1
    lag1 = prev_close
    
    features = np.array([
        open_, high_, low_, volume,
        ret,
        rolling_mean, rolling_std,
        lag1, lag2, lag3,
        rsi, macd, ema
    ]).reshape(1, -1)
    
    features_scaled = scaler.transform(features)
    log_price_pred = final_model.predict(features_scaled)[0]
    price_pred = np.exp(log_price_pred)
    
    sim_date = current_date + timedelta(days=day)
    
    # Save predictions if day in future_days
    if day in future_days:
        simulated_dates.append(sim_date)
        simulated_prices.append(price_pred)
    
    # Update prev_close for next iteration
    prev_close = price_pred

# Step 15: Print predicted future prices
print("\nFuture Price Predictions:")
for date, price, day in zip(simulated_dates, simulated_prices, future_days):
    print(f"Predicted price after {day} day(s): ${price:.4f}")

# Step 16: Plot future predictions as green dots only (no line)
plt.scatter(simulated_dates, simulated_prices, color='green', label='Future Predictions', marker='o')

plt.title('Tether (USDT) Price Prediction: Actual, Predicted (Train), and Future Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
