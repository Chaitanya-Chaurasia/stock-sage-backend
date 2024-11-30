from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

def get_historical_data(symbol, start_date, end_date, yf):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns]
    stock_data['Date_'] = pd.to_datetime(stock_data['Date_'])
    
    return stock_data

def prepare_dataset(sentiment_data, stock_data, symbol):
    sentiment_data['date'] = pd.to_datetime(sentiment_data[['year', 'month']].assign(DAY=1))
    sentiment_data['date'] = sentiment_data['date'].dt.tz_localize(None)
    sentiment_data.reset_index(drop=True, inplace=True)
    stock_data.reset_index(drop=True, inplace=True)
    stock_data['Date_'] = pd.to_datetime(stock_data['Date_'])
    stock_data['Date_'] = stock_data['Date_'].dt.tz_localize(None)
    dataset = pd.merge(sentiment_data, stock_data, left_on='date', right_on='Date_', how='inner')
    dataset['future_date'] = dataset['Date_'] + pd.DateOffset(days=60)
    future_prices = stock_data[['Date_', 'Close_'+symbol]].rename(columns={'Date_': 'future_date', 'Close_'+symbol: 'future_close'})
    dataset = pd.merge(dataset, future_prices, on='future_date', how='left')
    dataset['price_change_pct'] = ((dataset['future_close'] - dataset['Close_'+symbol]) / dataset['Close_'+symbol]) * 100
    dataset['price_up'] = (dataset['price_change_pct'] > 0).astype(int)
    dataset.dropna(inplace=True)
    return dataset

def train_model(dataset):
    X = dataset[['mspr', 'change']]
    y = dataset['price_up']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, report
