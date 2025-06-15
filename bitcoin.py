import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# --- Step 1: Custom SimpleGBMRegressor ---
class SimpleGBMRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_pred = None

    def fit(self, X, y):
        self.base_pred = np.mean(y)
        y_pred = np.full(y.shape, self.base_pred)
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full((X.shape[0],), self.base_pred)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# --- Step 2: Load Dataset ---
df = pd.read_csv('Dataset/coin_Bitcoin.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)

# --- Step 3: Feature Engineering ---
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

# --- Step 4: Log Transform Target ---
df['LogClose'] = np.log(df['Close'])

# --- Step 5: Feature & Target Prep ---
X = df[['Open', 'High', 'Low', 'Volume', 'Return', 'RollingMean', 'RollingStd',
        'Lag1', 'Lag2', 'Lag3', 'RSI', 'MACD', 'EMA']].values
y = df['LogClose'].values

# --- Step 6: Scale Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 7: KFold Cross Validation with SimpleGBM ---
param_grid = {
    'n_estimators': [100],
    'learning_rate': [0.1],
    'max_depth': [5]
}

best_score = float('inf')
best_params = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for n_est in param_grid['n_estimators']:
    for lr in param_grid['learning_rate']:
        for depth in param_grid['max_depth']:
            fold_mse = []
            for train_idx, val_idx in kf.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model = SimpleGBMRegressor(n_estimators=n_est,
                                           learning_rate=lr,
                                           max_depth=depth)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                mse = np.mean((y_val - y_pred) ** 2)
                fold_mse.append(mse)
            avg_mse = np.mean(fold_mse)
            if avg_mse < best_score:
                best_score = avg_mse
                best_params = {
                    'n_estimators': n_est,
                    'learning_rate': lr,
                    'max_depth': depth
                }

# --- Step 8: Train Final XGB Model ---
final_model = XGBRegressor(objective='reg:squarederror',
                           n_estimators=best_params['n_estimators'],
                           learning_rate=best_params['learning_rate'],
                           max_depth=best_params['max_depth'],
                           subsample=0.8,
                           colsample_bytree=0.8,
                           gamma=0)
final_model.fit(X_scaled, y)

# --- Step 9: Evaluation on Training Data ---
y_pred = final_model.predict(X_scaled)
mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y - y_pred))
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\nEvaluation Metrics:")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)

# --- Step 10: Inverse Log Transform ---
y_pred_actual = np.exp(y_pred)
y_actual = np.exp(y)

# --- Step 11: Predictions DataFrame ---
predicted_df = pd.DataFrame({
    'Actual': y_actual.ravel(),
    'Predicted': y_pred_actual.ravel()
}, index=df.index)
print("\nSample Predictions:")
print(predicted_df.tail())

# --- Step 12: Plot Actual vs Predicted ---
plt.figure(figsize=(14, 6))
plt.plot(df.index, y_actual, label='Actual Price', color='blue')
plt.plot(df.index, y_pred_actual, label='Predicted Price', color='orange')

# --- Step 13: Save Model ---
os.makedirs('models', exist_ok=True)
joblib.dump(final_model, 'models/best_xgboost_bitcoin_model.pkl')
print("\nModel saved to models/best_xgboost_bitcoin_model.pkl")

# --- Step 14: Future Forecasting ---
last_row = df.iloc[-1].copy()
current_date = datetime(2025, 5, 15)

rolling_window = list(df['Close'].tail(5))
lag1 = last_row['Close']
lag2 = df['Close'].iloc[-2] if len(df) > 1 else lag1
lag3 = df['Close'].iloc[-3] if len(df) > 2 else lag2

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
    ret = (prev_close - lag1) / lag1 if lag1 != 0 else 0
    rolling_window.append(prev_close)
    if len(rolling_window) > 5:
        rolling_window.pop(0)
    rolling_mean = np.mean(rolling_window)
    rolling_std = np.std(rolling_window)

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

    if day in future_days:
        simulated_dates.append(sim_date)
        simulated_prices.append(price_pred)

    prev_close = price_pred

# --- Step 15: Print Predictions ---
print("\nFuture Price Predictions:")
for date, price, day in zip(simulated_dates, simulated_prices, future_days):
    print(f"Predicted price after {day} day(s): ${price:.4f}")

# --- Step 16: Plot Predictions ---
plt.scatter(simulated_dates, simulated_prices, color='green', label='Future Predictions', marker='o')
plt.title('Bitcoin Price Prediction: Actual, Predicted (Train), and Future Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
