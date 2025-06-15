from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_session import Session
from flask_cors import CORS
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd
import importlib
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from predict import predict_future_prices

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
CORS(app)
COINS = ['Bitcoin', 'Cardano', 'ChainLink', 'Ethereum', 'Litecoin',
         'Monero', 'Stellar', 'Tether', 'USDCoin', 'XRP']

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'sql@girija'
app.config['MYSQL_DB'] = 'cryptiq'

mysql = MySQL(app)
CORS(app)

# Dataset directory
DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")


# Utility: Get filename from crypto name
def get_csv_filename(crypto):
    return f"coin_{crypto}.csv"


# Utility: Dynamic import of prediction module
def get_prediction_module(crypto):
    try:
        return importlib.import_module(crypto)
    except ModuleNotFoundError:
        return None


# Utility: Generate graph for crypto
def generate_crypto_graph(crypto):
    csv_file = os.path.join(DATASET_DIR, get_csv_filename(crypto))
    if not os.path.exists(csv_file):
        return None, f"Data for {crypto} not found."

    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').tail(30)

    plt.figure(figsize=(8, 4))
    plt.plot(df['Date'], df['Close'], marker="o", label=f"{crypto} Close Price")
    plt.title(f"{crypto} Closing Price (Last 30 Days)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return "data:image/png;base64," + img_base64, None


# ---------------- Routes ---------------- #

# Home
@app.route('/')
def home():
    if 'user' in session:
        return render_template('index.html', username=session['user'],userid = session['name'])
    else:
        return redirect(url_for('login'))


# Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        phone=request.form['phone']
        dob=request.form['dob']
        address=request.form['address']
        hashed_password = generate_password_hash(password)

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            return render_template('register.html', error="Email already exists")

        cursor.execute("INSERT INTO users (username, password,email,phone,address,dob) VALUES (%s, %s,%s, %s,%s,%s)", (username, hashed_password, email,phone,address,dob))
        mysql.connection.commit()
        cursor.close()

        return redirect(url_for('login'))

    return render_template('register.html')


#Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']  # changed from username to email
        password = request.form['password']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        cursor.close()

        if user and check_password_hash(user[2], password):  # password at index 2
            session['user'] = user[3]  # store username (or you can store email if you want)
            session['name']=user[1]
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid email or password")

    return render_template('login.html')


# Fixed Logout
@app.route('/logout')
def logout():
    print(">>> LOGOUT TRIGGERED <<<")
    session.pop('user', None)
    return redirect(url_for('home'))


# Account Page
@app.route('/account')
def account():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('account.html', user=session['user'],userid = session['name'])


# Graph Page
@app.route('/graph.html')
def graph_page():
    return render_template('graph.html',user=session['user'],userid = session['name'])


# About Page
@app.route('/about')
def about():
    return render_template('about.html')


# Graph API
@app.route("/graph", methods=["POST"])
def graph():
    data = request.json
    crypto = data.get("crypto")
    if not crypto:
        return jsonify({"error": "Missing crypto name"}), 400

    graph_img, err = generate_crypto_graph(crypto)
    if err:
        return jsonify({"error": err}), 404
    return jsonify({"graph": graph_img})



@app.route("/forecast", methods=["POST"])
def forecast():
    print("Content-Type:", request.content_type)
    print("Raw data:", request.data)
    print("Parsed JSON:", request.json)
    data = request.get_json()
    print(">>> /forecast data received:", data)  # Debug line

    crypto = data.get("crypto")
    days = data.get("days")

    if not crypto or days is None:
        return jsonify({"error": "Missing crypto or days"}), 400

    module = get_prediction_module(crypto)
    if module is None or not hasattr(module, "predict_price"):
        return jsonify({"error": f"Prediction logic for {crypto} not found."}), 404

    try:
        forecast_date, predicted_price = module.predict_price(int(days))
        return jsonify({"date": forecast_date, "price": predicted_price})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        coin = data.get('coin')
        # days = int(data.get('days', 1))
        days = data.get('days')

        if not coin:
            return jsonify({'error': 'Coin not provided'}), 400

        predicted_price = predict_future_prices(coin, days)
        return jsonify({'price': float(predicted_price)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
# Run Server
if __name__ == "__main__":
    app.run(debug=True)
