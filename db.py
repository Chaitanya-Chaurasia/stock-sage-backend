import sqlite3

def init_db():
    conn = sqlite3.connect('data/stock_data.db')
    cursor = conn.cursor()
    clear_database()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            date TEXT NOT NULL,
            close_price REAL NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reddit_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            content TEXT NOT NULL,
            sentiment TEXT NOT NULL DEFAULT 'neutral'
        )
    ''')
    conn.commit()
    conn.close()

def clear_database():
    conn = sqlite3.connect('./data/stock_data.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM stock_data')
    cursor.execute('DELETE FROM reddit_posts')
    conn.commit()
    conn.close()
    
def insert_stock_data(company, date, close_price):
    conn = sqlite3.connect('data/stock_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO stock_data (company, date, close_price)
        VALUES (?, ?, ?)
    ''', (company, date, close_price))
    conn.commit()
    conn.close()

def insert_reddit_post(date, content, sentiment='neutral'):
    conn = sqlite3.connect('data/stock_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO reddit_posts (date, content, sentiment)
        VALUES (?, ?, ?)
    ''', (date, content, sentiment))
    conn.commit()
    conn.close()
    
def get_reddit_posts():
    conn = sqlite3.connect('./data/stock_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT date, content, sentiment FROM reddit_posts ORDER BY date DESC')
    posts = cursor.fetchall()
    conn.close()
    return posts

def get_stock_data(company):
    conn = sqlite3.connect('data/stock_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT date, close_price FROM stock_data WHERE company = ? ORDER BY date DESC', (company,))
    stock_data = cursor.fetchall()
    conn.close()
    return stock_data