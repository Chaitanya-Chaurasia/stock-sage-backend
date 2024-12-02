from flask import Flask, request, jsonify
from utils import init_clients
from db import init_db
from data_collection import collect_reddit_posts, calculate_sentiment_scores, collect_stock_prices
from trading_strategy import train_model, get_historical_data, prepare_dataset
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


app = Flask(__name__)
CORS(app)
finnhub, yf = init_clients()
init_db()

@app.route("/")
def home():
    return "StockSage API is running!"

@app.route("/news")
def get_news():
    return finnhub.general_news('general', min_id=0)[:4]


@app.route("/prices")
def get_prices():
    stocks = ['AAPL', 'TSLA', 'PLTR', 'NVDA', 'AMZN', 'GOOG']
    return [finnhub.quote(i) for i in stocks]


@app.route("/collect_reddit_posts",  methods=['POST'])
def reddit_posts():
    try:
        data = request.json
        if not data:
            return {"error": "Missing 'company_name' in request body"}
        company_name = data['company']
        res = collect_reddit_posts(company_name)
        return res

    except Exception as e:
        return {"error": str(e)}

@app.route("/market_sentiment", methods=['POST'])
def market_sentiment():
    data = request.get_json()
    symbol = data.get('symbol')
    start_date = data.get('start_date', '2024-01-01')
    end_date = data.get('end_date', '2024-11-20')
    
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400

    sentiment_data = finnhub.stock_insider_sentiment(symbol, start_date, end_date)
    sentiment_df = pd.DataFrame(sentiment_data['data'])
    stock_data = get_historical_data(symbol, start_date, end_date, yf)
    dataset = prepare_dataset(sentiment_df, stock_data, symbol)
    
    if dataset.empty:
        return jsonify({'error': 'Insufficient data for modeling'}), 400
    
    model, report = train_model(dataset)
    
    latest_data = dataset.tail(1)
    X_new = latest_data[['mspr', 'change']]
    
    prediction = model.predict(X_new)
    prediction_proba = model.predict_proba(X_new)
    
    result = {
        'symbol': symbol,
        'prediction': 'Up' if prediction[0] == 1 else 'Down',
        'confidence': float(max(prediction_proba[0])) * 100,
        'report': report
    }
    
    return jsonify(result)

@app.route("/trade_decisions", methods=['POST'])
def trade_decisions():
    try:
        data = request.json
        if not data or 'threshold_high' not in data or 'threshold_low' not in data:
            return {"error": "Missing threshold parameters in request body"}, 400
        
        threshold_high = float(data['threshold_high'])
        threshold_low = float(data['threshold_low'])
        sentiment_scores = data.get('sentiment_scores', [])
        closing_prices = data.get('closing_prices', [])
        if len(sentiment_scores) < 2 or len(closing_prices) < 2:
            return {"error": "Insufficient data for training"}, 400

        min_length = min(len(sentiment_scores), len(closing_prices))
        sentiment_scores = sentiment_scores[-min_length:]
        closing_prices = closing_prices[-min_length:]

        X = np.array(sentiment_scores[:-1]).reshape(-1, 1)
        y = []
        for i in range(1, len(closing_prices)):
            if closing_prices[i] > closing_prices[i - 1]:
                y.append(1) 
            elif closing_prices[i] < closing_prices[i - 1]:
                y.append(-1)  
            else:
                y.append(0) 

        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        latest_score = sentiment_scores[-1]
        prediction = model.predict([[latest_score]])[0]
        if latest_score > threshold_high:
            action = "buy"
        elif latest_score < threshold_low:
            action = "sell"
        else:
            action = "hold"
        decision_log = {
            "threshold_high": threshold_high,
            "threshold_low": threshold_low,
            "latest_score": latest_score,
            "model_prediction": "buy" if prediction == 1 else "sell" if prediction == -1 else "hold",
            "action": action
        }
        
        return jsonify(decision_log)

    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/daily_sentiment_scores", methods=['GET'])
def get_daily_sentiment_scores():
   result = calculate_sentiment_scores()
   return jsonify(result)
    
@app.route("/stock_prices", methods=['POST'])
def get_stock_prices():
    data = request.json
    if not data:
        return {"error": "Missing 'company' in request body"}
    
    company_name = data['company']
    res = collect_stock_prices(company_name, yf)
    return res

@app.route("/get_company_news", methods=['POST'])
def get_company_news():
    return jsonify(finnhub.company_news('AMZN',  _from="2024-06-01", to="2024-06-10"))


if __name__ == "__main__":
    app.run(debug=True, port=8000)
