from datetime import datetime, timedelta
from db import insert_reddit_post, get_reddit_posts 
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import praw
def sanitize_content(content):
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text

def analyze_sentiment(content):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(content)
    return sentiment['compound']

def collect_reddit_posts(keyword):
    reddit = praw.Reddit(
    client_id="wi33GqBzzwVZ0GLmKbQgfQ",
    client_secret="ACR9vv0xaXHMcyadmZa6p9HP51Lecg",
    user_agent="project2"
    )
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1 * 365)
    posts = []
    try:
        for submission in reddit.subreddit("wallstreetbets").search(keyword, limit=100, sort="new"):
            post_date = datetime.fromtimestamp(submission.created_utc)
            if post_date < start_date:
                break
            
            content = submission.title + " " + (submission.selftext if submission.selftext else "")
            sanitized_content = sanitize_content(content)
            sentiment_score = analyze_sentiment(sanitized_content)
            insert_reddit_post(post_date.strftime("%Y-%m-%d"), sanitized_content, sentiment_score)
            posts.append((submission.title, submission.selftext, sentiment_score))
        return posts
    except Exception as e:
        print(f"Error: {e}")

def calculate_sentiment_scores():
    try:
        posts = get_reddit_posts()
        daily_counts = defaultdict(lambda: {'positive': 0, 'negative': 0})
        neutral = 0
        
        for post in posts:
            date = post[0]
            sentiment_score = float(post[-1])
            
            if sentiment_score > 0:
                daily_counts[date]['positive'] += 1
            elif sentiment_score < 0:
                daily_counts[date]['negative'] += 1
            else:
                neutral += 1

        total_non_neutral = sum(
            counts['positive'] + counts['negative'] for counts in daily_counts.values()
        )
        if total_non_neutral > 0 and neutral >= total_non_neutral / 2:
            print("Significant neutrals")
        
        daily_sentiment_scores = []
        total_sentiment = 0
        num_days = 0
        
        for date, counts in sorted(daily_counts.items()):
            N_pos = counts['positive']
            N_neg = counts['negative']
            
            if N_pos + N_neg > 0:
                sentiment = ((N_pos - N_neg) / (N_pos + N_neg)) * 100
                daily_sentiment_scores.append({'date': date, 'score': sentiment, 'sentiment': "positive" if sentiment > 0 else "neutral" if sentiment == 0 else "negative"})
                total_sentiment += sentiment
                num_days += 1
                
        average_sentiment = total_sentiment / num_days if num_days > 0 else 0
        
        return {
            'daily_sentiment_scores': daily_sentiment_scores,
            'average_sentiment': average_sentiment,
            'neutral_sentiment': neutral
        }
    except Exception as e:
        return {"error": str(e)}

def collect_stock_prices(symbol, yf):
    try:
        if not symbol:
            return {"error": "Missing 'symbol' query parameter"}
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=182)  
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if hist.empty:
            return {"error": f"No data found for symbol '{symbol}'"}
        
        daily_prices = []
        for index, row in hist.iterrows():
            daily_prices.append({
                "date": index.strftime('%Y-%m-%d'),
                "open": round(row['Open'], 2),
                "close": round(row['Close'], 2)
            })
        
        total_open = hist['Open'].sum()
        total_close = hist['Close'].sum()
        num_days = len(hist)
        
        average_open = round(total_open / num_days, 2) if num_days > 0 else 0
        average_close = round(total_close / num_days, 2) if num_days > 0 else 0
        
        return {
            "daily_prices": daily_prices,
            "average_open": average_open,
            "average_close": average_close
        }
    
    except Exception as e:
        print(f"Error in /stock_prices endpoint: {e}")
        return {"error": str(e)}