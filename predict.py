import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import talib
import os

def predict_future_prices(coin_name, target_day):
    file_path = f'Dataset/coin_{coin_name}.csv'
    df = pd.read_csv(file_path, parse_dates=['Date'])
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

    # Compute natural log of Close price to stabilize variance and prepare for log-return based models
    df['LogClose'] = np.log(df['Close'])

    # Prepare Features
    X = df[['Open', 'High', 'Low', 'Volume', 'Return', 'RollingMean', 'RollingStd',
            'Lag1', 'Lag2', 'Lag3', 'RSI', 'MACD', 'EMA']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Load Model
    model_path = f'models/best_xgboost_{coin_name.lower()}_model.pkl'
    if not os.path.exists(model_path):
        return None

    model = joblib.load(model_path)

    # Last known values
    last_row = df.iloc[-1].copy()
    current_date = datetime.today()
    rolling_window = list(df['Close'].tail(5))
    lag1 = last_row['Close']
    lag2 = df['Close'].iloc[-2]
    lag3 = df['Close'].iloc[-3]

    rsi = last_row['RSI']
    macd = last_row['MACD']
    ema = last_row['EMA']
    volume = last_row['Volume']
    open_ = last_row['Open']
    high_ = last_row['High']
    low_ = last_row['Low']
    prev_close = lag1

    for day in range(1, target_day + 1):
        ret = (prev_close - lag1) / lag1 if lag1 != 0 else 0
        rolling_window.append(prev_close)
        if len(rolling_window) > 5:
            rolling_window.pop(0)
        rolling_mean = np.mean(rolling_window)
        rolling_std = np.std(rolling_window)

        lag3 = lag2
        lag2 = lag1
        lag1 = prev_close

        features = np.array([[
            open_, high_, low_, volume, ret, rolling_mean, rolling_std,
            lag1, lag2, lag3, rsi, macd, ema
        ]])
        features_scaled = scaler.transform(features)
        log_pred = model.predict(features_scaled)[0]
        prev_close = np.exp(log_pred)

    return prev_close
