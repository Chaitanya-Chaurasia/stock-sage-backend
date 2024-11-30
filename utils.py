import finnhub
import praw
import yfinance as yf

def init_clients():
    finnhub_client = finnhub.Client(api_key="csuma89r01qgo8ni81o0csuma89r01qgo8ni81og")
    return finnhub_client, yf