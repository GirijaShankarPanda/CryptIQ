
# 🪙 CryptIQ: Cryptocurrency Price Prediction and Forecasting Platform

CryptIQ is a web-based application that leverages machine learning techniques—particularly XGBoost—for short-term price prediction of major cryptocurrencies. It combines robust data preprocessing, advanced feature engineering, and predictive modeling into a user-friendly interface to provide real-time insights and forecasts.

---

## 📌 Objective

The main objectives of this project are:

- To develop a machine learning-based model for short-term price prediction of major cryptocurrencies.
- To evaluate the efficacy of XGBoost in handling high-frequency, non-linear, and noisy financial data.
- To design a user-friendly web platform, CryptIQ, that delivers real-time predictive analytics.
- To establish a complete pipeline integrating data collection, preprocessing, feature engineering, model training, evaluation, and deployment.

---

## 🔬 Original Contribution

This project presents several key contributions:

- **Custom Implementation:** A custom-tuned XGBoost regression framework specifically optimized for cryptocurrency market behavior.
- **Advanced Feature Engineering:** Inclusion of lag features and log-transformed prices to enhance model accuracy.
- **Interactive Platform:** Development of a web-based application (CryptIQ) for real-time user interaction and visualization.
- **Robust Validation:** Comparative analysis across five different cryptocurrencies to evaluate model generalizability and robustness.

---

## 📚 Literature Survey

Over time, financial time series forecasting has evolved significantly. Traditional models like ARIMA and GARCH, though initially effective in stock prediction, fall short in modeling the highly volatile and non-linear nature of cryptocurrencies. To overcome this, deep learning approaches such as LSTM and GRU have been adopted due to their strength in capturing temporal dependencies [1], [2]. However, they require large datasets and heavy computational resources.

Ensemble methods like **XGBoost** have gained attention for their scalability, built-in feature selection, and regularization capabilities, making them effective for financial prediction tasks [3], [4]. 

**References:**

[1] Ghadiri, H., & Hajizadeh, E. (2025). Designing a cryptocurrency trading system with deep reinforcement learning utilizing LSTM neural networks and XGBoost feature selection. *Applied Soft Computing, 175*, 113029.

[2] Islam, M. S., et al. (2025). Machine Learning-Based Cryptocurrency Prediction: Enhancing Market Forecasting with Advanced Predictive Models. *Journal of Ecohumanism, 4*(2), 2498–2519.

[3] Qureshi, S. M., et al. (2025). Evaluating machine learning models for predictive accuracy in cryptocurrency price forecasting. *PeerJ Computer Science, 11*, e2626.

[4] Kumar, K. S., et al. (2024). Comparative Analysis of LSTM and XGBoost Models for Short-Term Bitcoin Price Prediction. *2024 3rd ICAAI Conference*, IEEE.

[5] Zubair, M., et al. (2024). An improved machine learning-driven framework for cryptocurrencies price prediction with sentimental cautioning. *IEEE Access*.

---

## ⚙️ Features

- 📈 Real-time price prediction using XGBoost
- 🔍 Feature-rich models with lag features, RSI, MACD, EMA, etc.
- 🌐 Web app built using Flask and HTML/CSS
- 📊 Graphs showing predicted vs actual prices
- 🔄 Rolling mean and standard deviation plots for data smoothing
- 🔐 User login and session management

---

## 🛠️ Technologies Used

| Component         | Tools & Technologies                                |
|------------------|------------------------------------------------------|
| Programming Lang | Python                                               |
| Libraries        | Pandas, NumPy, XGBoost, Scikit-learn, Matplotlib     |
| Web Framework    | Flask, HTML, CSS                                     |
| Deployment       | Localhost / Production-ready Flask app               |
| Visualization    | Matplotlib, Seaborn                                  |

---

## 🧠 Algorithms Used

CryptIQ employs advanced machine learning algorithms to ensure robust and accurate cryptocurrency price prediction:

- **XGBoost (Extreme Gradient Boosting):**
  - Core regression model used for price prediction.
  - Efficient at handling high-frequency, noisy, and non-linear financial data.
  - Incorporates regularization to reduce overfitting and improve generalization.

- **Gradient Boosting (Custom Implementation):**
  - Ensemble learning technique that sequentially builds weak learners.
  - Used for comparative evaluation against XGBoost to validate consistency and accuracy.

## 🗃️ Dataset

The datasets used include historical data for 10 major cryptocurrencies:
- **Bitcoin (BTC)**
- **Stellar (XLM)**
- **Ethereum (ETH)**
- **Cardano (ADA)**
- **Monero (XMR)**
- **ChainLink (LINK)**
- **Litecoin (LTC)**
- **Tether (USDT)**
- **USDCoin (USDC)**
- **XRP**

### Columns Used

- **Less Features Dataset:** `Sl No`, `Open`, `High`, `Low`, `Close`, `Volume`, `MarketCap`
- **More Features Dataset (after preprocessing):** `Open`, `High`, `Low`, `Close`, `Volume`, `MarketCap`, `lag1`, `lag2`, `lag3`, `rolling_mean`, `rolling_std`, `RSI`, `MACD`, `EMA`

---

## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/CryptIQ.git
cd CryptIQ
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## 🧪 Results & Evaluation

Comparative evaluation showed that:

- **XGBoost outperformed LSTM** in execution speed and robustness with limited data.
- **More features led to better accuracy**, as demonstrated by RMSE and R² scores.
- Model generalized well across different cryptocurrencies.

---

## 📩 Contact

For questions or suggestions, feel free to open an issue or reach out to the team.
