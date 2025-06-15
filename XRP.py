import pandas as pd
import numpy as np
import xgboost as xgb
import talib
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from datetime import datetime, timedelta

# Define SimpleGBMRegressor here:
class SimpleGBMRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.init_pred = None
        np.random.seed(random_state)

    def fit(self, X, y):
        self.init_pred = np.mean(y)
        residuals = y - self.init_pred
        self.trees = []

        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[indices]
            residuals_sample = residuals[indices]
            # Fit a decision tree regressor on residuals
            tree = xgb.XGBRegressor(objective='reg:squarederror',
                                    max_depth=self.max_depth,
                                    n_estimators=1,
                                    learning_rate=1,
                                    random_state=self.random_state)
            tree.fit(X_sample, residuals_sample)
            pred = tree.predict(X)
            residuals = residuals - self.learning_rate * pred
            self.trees.append(tree)

    def predict(self, X):
        pred = np.full(shape=(X.shape[0],), fill_value=self.init_pred)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred


def main():
    # === Step 1: Load XRP dataset ===
    df = pd.read_csv('Dataset/coin_XRP.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)

    # === Step 2: Feature Engineering ===
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

    # === Step 3: Log Transformation on Target (Close Price) ===
    df['LogClose'] = np.log(df['Close'])

    # === Step 4: Prepare features and labels ===
    feature_cols = ['Open', 'High', 'Low', 'Volume', 'Return', 'RollingMean', 'RollingStd',
                    'Lag1', 'Lag2', 'Lag3', 'RSI', 'MACD', 'EMA']
    X = df[feature_cols].values
    y = df['LogClose'].values

    # === Step 5: Feature Scaling ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === Step 6: Hyperparameter tuning with K-Fold (simplified grid) ===
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

    print("Starting hyperparameter tuning with XGBoost + SimpleGBM ...")
    for n_est in param_grid['n_estimators']:
        for lr in param_grid['learning_rate']:
            for depth in param_grid['max_depth']:
                for subsample in param_grid['subsample']:
                    for colsample in param_grid['colsample_bytree']:
                        for gamma in param_grid['gamma']:
                            fold_mse = []
                            for train_idx, val_idx in kf.split(X_scaled):
                                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                                y_train, y_val = y[train_idx], y[val_idx]

                                # Train XGBoost
                                model_xgb = xgb.XGBRegressor(objective='reg:squarederror',
                                                             n_estimators=n_est,
                                                             learning_rate=lr,
                                                             max_depth=depth,
                                                             subsample=subsample,
                                                             colsample_bytree=colsample,
                                                             gamma=gamma,
                                                             random_state=42)
                                model_xgb.fit(X_train, y_train)
                                y_pred_xgb = model_xgb.predict(X_val)
                                mse_xgb = np.mean((y_val - y_pred_xgb) ** 2)

                                # Train SimpleGBM
                                model_gbm = SimpleGBMRegressor(n_estimators=n_est,
                                                               learning_rate=lr,
                                                               max_depth=depth,
                                                               random_state=42)
                                model_gbm.fit(X_train, y_train)
                                y_pred_gbm = model_gbm.predict(X_val)
                                mse_gbm = np.mean((y_val - y_pred_gbm) ** 2)

                                # Average mse of both models
                                fold_mse.append((mse_xgb + mse_gbm) / 2)

                            avg_mse = np.mean(fold_mse)
                            if avg_mse < best_score:
                                best_score = avg_mse
                                best_params = {
                                    'n_estimators': n_est,
                                    'learning_rate': lr,
                                    'max_depth': depth,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample,
                                    'gamma': gamma
                                }

    print("Best hyperparameters found:", best_params)
    print("Best CV MSE:", best_score)

    # === Step 7: Train final XGBoost model ===
    final_model = xgb.XGBRegressor(objective='reg:squarederror',
                                   **best_params,
                                   random_state=42)
    final_model.fit(X_scaled, y)

    # === Step 8: Predict on training data ===
    y_pred = final_model.predict(X_scaled)

    # === Step 9: Evaluation metrics ===
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - y_pred))
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print("\nEvaluation Metrics on Training Data:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R-squared: {r2:.6f}")

    # === Step 10: Inverse log transform for plotting ===
    y_pred_actual = np.exp(y_pred)
    y_actual = np.exp(y)

    # === Step 11: Plot actual vs predicted prices ===
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, y_actual, label='Actual Price', color='blue')
    plt.plot(df.index, y_pred_actual, label='Predicted Price', color='orange')

    # === Step 12: Save model and scaler ===
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_model, 'models/xgboost_xrp_model.pkl')
    joblib.dump(scaler, 'models/xrp_scaler.pkl')
    print("\nModel and scaler saved to 'models/' directory.")

    # === Step 13: Future prediction from TODAY (2025-05-16) ===
    today = datetime(2025, 5, 16)

    close_history = list(df['Close'].iloc[-30:].values)
    volume_history = list(df['Volume'].iloc[-30:].values)

    if df.index[-1] > today:
        valid_dates = df.index[df.index <= today]
        close_history = list(df.loc[valid_dates[-30:], 'Close'].values)
        volume_history = list(df.loc[valid_dates[-30:], 'Volume'].values)
    elif df.index[-1] < today:
        pass

    prev_close = close_history[-1]

    future_days = [1, 7, 30, 365]
    simulated_dates = []
    simulated_prices = []

    for day in range(1, max(future_days) + 1):
        lag1 = close_history[-1]
        lag2 = close_history[-2] if len(close_history) > 1 else lag1
        lag3 = close_history[-3] if len(close_history) > 2 else lag2

        ret = (prev_close - lag1) / lag1 if lag1 != 0 else 0

        rolling_window = close_history[-5:]
        rolling_mean = np.mean(rolling_window)
        rolling_std = np.std(rolling_window)

        np_close = np.array(close_history)
        rsi_arr = talib.RSI(np_close, timeperiod=14)
        macd_arr, macd_signal_arr, _ = talib.MACD(np_close, fastperiod=12, slowperiod=26, signalperiod=9)
        ema_arr = talib.EMA(np_close, timeperiod=14)

        rsi = rsi_arr[-1] if not np.isnan(rsi_arr[-1]) else 50
        macd = macd_arr[-1] if not np.isnan(macd_arr[-1]) else 0
        ema = ema_arr[-1] if not np.isnan(ema_arr[-1]) else prev_close

        open_ = prev_close * (1 + np.random.normal(0, 0.002))
        high_ = open_ * (1 + np.random.uniform(0, 0.005))
        low_ = open_ * (1 - np.random.uniform(0, 0.005))
        volume = volume_history[-1] * (1 + np.random.normal(0, 0.05))
        volume = max(volume, 1)

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

        sim_date = today + timedelta(days=day)
        if day in future_days:
            simulated_dates.append(sim_date)
            simulated_prices.append(price_pred)

        close_history.append(price_pred)
        volume_history.append(volume)
        prev_close = price_pred

    # === Step 14: Print future price predictions ===
    print("\nFuture Price Predictions:")
    for date, price, day in zip(simulated_dates, simulated_prices, future_days):
        print(f"Price after {day} day(s): ${price:.4f}")

    # === Step 15: Plot future predictions ===
    plt.scatter(simulated_dates, simulated_prices, color='green', label='Future Predictions', marker='o')
    plt.title('XRP Price Prediction: Actual, Predicted (Train), and Future')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
