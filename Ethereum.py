import pandas as pd
import numpy as np
import xgboost as xgb
import talib
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from datetime import datetime, timedelta

# Custom Gradient Boosting Regressor
class SimpleGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.init_prediction = None

    def fit(self, X, y):
        self.models = []
        self.init_prediction = np.mean(y)
        residuals = y - self.init_prediction
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            prediction = tree.predict(X)
            residuals -= self.learning_rate * prediction
            self.models.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.init_prediction)
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# Load Ethereum dataset
df = pd.read_csv('Dataset/coin_Ethereum.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)

# Feature Engineering
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

# Log Transformation on Close
df['LogClose'] = np.log(df['Close'])

# Prepare features and target
X = df[['Open', 'High', 'Low', 'Volume', 'Return', 'RollingMean', 'RollingStd',
        'Lag1', 'Lag2', 'Lag3', 'RSI', 'MACD', 'EMA']].values
y = df['LogClose'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model Definitions
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                              n_estimators=100,
                              learning_rate=0.1,
                              max_depth=5,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              gamma=0)

gbm_model = SimpleGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# Fit both models
xgb_model.fit(X_scaled, y)
gbm_model.fit(X_scaled, y)

# Predict and evaluate
def evaluate_model(model, name):
    y_pred = model.predict(X_scaled)
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - y_pred))
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    print(f"\n{name} Evaluation:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    return y_pred, rmse

y_pred_xgb, rmse_xgb = evaluate_model(xgb_model, "XGBoost")
y_pred_gbm, rmse_gbm = evaluate_model(gbm_model, "SimpleGBM")

# Choose best model
best_model = xgb_model if rmse_xgb < rmse_gbm else gbm_model
best_pred = y_pred_xgb if best_model == xgb_model else y_pred_gbm
model_name = "xgboost" if best_model == xgb_model else "simple_gbm"

# Save best model
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, f'models/best_{model_name}_ethereum_model.pkl')
print(f"\nBest model ({model_name}) saved to models/best_{model_name}_ethereum_model.pkl")

# Inverse transform
y_pred_actual = np.exp(best_pred)
y_actual = np.exp(y)

# Create prediction DataFrame
predicted_df = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred_actual}, index=df.index)
print("\nSample Predictions:")
print(predicted_df.tail())

# Plot training fit
plt.figure(figsize=(14, 6))
plt.plot(df.index, y_actual, label='Actual Price', color='blue')
plt.plot(df.index, y_pred_actual, label='Predicted Price (Train)', color='orange')

# Simulate Future Prediction
last_row = df.iloc[-1].copy()
current_date = datetime(2025, 5, 16)
future_days = [1, 7, 30, 365]
simulated_dates = []
simulated_prices = []

rolling_window = df['Close'].tail(5).tolist()
lag1, lag2, lag3 = last_row['Close'], df['Close'].iloc[-2], df['Close'].iloc[-3]
rsi, macd, ema = last_row['RSI'], last_row['MACD'], last_row['EMA']
volume, open_, high_, low_ = last_row['Volume'], last_row['Open'], last_row['High'], last_row['Low']
prev_close = lag1

for day in range(1, max(future_days) + 1):
    ret = (prev_close - lag1) / lag1 if lag1 != 0 else 0
    rolling_window.append(prev_close)
    if len(rolling_window) > 5:
        rolling_window.pop(0)
    rolling_mean = np.mean(rolling_window)
    rolling_std = np.std(rolling_window)

    lag3, lag2, lag1 = lag2, lag1, prev_close

    features = np.array([
        open_, high_, low_, volume,
        ret, rolling_mean, rolling_std,
        lag1, lag2, lag3,
        rsi, macd, ema
    ]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    log_price_pred = best_model.predict(features_scaled)[0]
    price_pred = np.exp(log_price_pred)
    sim_date = current_date + timedelta(days=day)
    if day in future_days:
        simulated_dates.append(sim_date)
        simulated_prices.append(price_pred)
    prev_close = price_pred

# Show predictions
print("\nFuture Price Predictions:")
for date, price, day in zip(simulated_dates, simulated_prices, future_days):
    print(f"Predicted price after {day} day(s): ${price:.4f}")

# Plot future predictions
plt.scatter(simulated_dates, simulated_prices, color='green', label='Future Predictions', marker='o')
plt.title('Ethereum (ETH) Price Prediction: Actual, Predicted (Train), and Future Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
