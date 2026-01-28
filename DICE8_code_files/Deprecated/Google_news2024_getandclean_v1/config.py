# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 18:58:04 2026

@author: 25010
"""

# config.py
"""
配置文件和常量定义
"""

# 资产配置
ASSETS_CONFIG = {
    "S&P 500": {
        "search_terms": ["S&P 500", "SPX", "SP500", "Standard & Poor's 500"],
        "category": "US Stock Index"
    },
    "NASDAQ Composite": {
        "search_terms": ["NASDAQ Composite", "NASDAQ", "IXIC", "NASDAQ Index"],
        "category": "US Stock Index"
    },
    "Dow Jones Industrial Average": {
        "search_terms": ["Dow Jones Industrial Average", "Dow Jones", "DJIA", "Dow 30"],
        "category": "US Stock Index"
    },
    "CAC 40": {
        "search_terms": ["CAC 40", "CAC40", "Paris Stock Exchange", "French stock index"],
        "category": "European Stock Index"
    },
    "FTSE 100": {
        "search_terms": ["FTSE 100", "FTSE100", "London Stock Exchange", "UK stock index"],
        "category": "European Stock Index"
    },
    "EuroStoxx 50": {
        "search_terms": ["EuroStoxx 50", "Euro Stoxx 50", "STOXX50", "EURO STOXX 50"],
        "category": "European Stock Index"
    },
    "Hang Seng Index": {
        "search_terms": ["Hang Seng Index", "Hang Seng", "HSI", "Hong Kong stock index"],
        "category": "Asian Stock Index"
    },
    "Shanghai Composite": {
        "search_terms": ["Shanghai Composite", "SSE Composite", "Shanghai Stock Exchange", "SSEC"],
        "category": "Asian Stock Index"
    },
    "BSE Sensex": {
        "search_terms": ["BSE Sensex", "Sensex", "Bombay Stock Exchange", "Indian stock market"],
        "category": "Asian Stock Index"
    },
    "Nifty 50": {
        "search_terms": ["Nifty 50", "NSE Nifty", "National Stock Exchange India", "Nifty index"],
        "category": "Asian Stock Index"
    },
    "KOSPI": {
        "search_terms": ["KOSPI", "Korea Composite Stock Price Index", "Korean stock index", "South Korea stock market"],
        "category": "Asian Stock Index"
    },
    "Gold": {
        "search_terms": ["Gold price", "Gold market", "XAU", "Gold futures"],
        "category": "Commodity"
    },
    "Silver": {
        "search_terms": ["Silver price", "Silver market", "XAG", "Silver futures"],
        "category": "Commodity"
    },
    "WTI Crude Oil Futures": {
        "search_terms": ["WTI Crude Oil", "WTI oil price", "West Texas Intermediate", "Crude oil futures"],
        "category": "Commodity"
    }
}

# 抓取参数
YEAR = 2023
MONTHS = list(range(1, 13))  # 1-12月
MAX_NEWS_PER_MONTH = 19

# 浏览器设置
CHROME_OPTIONS = {
    "headless": True,
    "window_size": "1920,1080",
    "disable_images": True,
    "disable_css": True
}

# 请求设置
REQUEST_TIMEOUT = 10
SELENIUM_TIMEOUT = 20
DELAY_BETWEEN_REQUESTS = (1, 3)  # 随机延迟范围
DELAY_BETWEEN_MONTHS = (5, 10)
DELAY_BETWEEN_ASSETS = (10, 20)

# 文件输出设置
OUTPUT_BASE_DIR = "multi_asset_news_data"
OUTPUT_FORMATS = ["csv", "json", "txt"]

# 金融关键词（用于内容分析）
FINANCIAL_TERMS = [
    'stock', 'market', 'price', 'index', 'invest', 'trade', 'share',
    'dollar', 'euro', 'yen', 'pound', 'currency', 'exchange',
    'rate', 'yield', 'dividend', 'profit', 'revenue', 'earnings',
    'economic', 'economy', 'growth', 'inflation', 'interest',
    'federal', 'central bank', 'monetary', 'policy', 'crisis',
    'volatility', 'risk', 'return', 'portfolio', 'asset', 'bond',
    'commodity', 'future', 'option', 'derivative', 'hedge',
    'bull', 'bear', 'rally', 'crash', 'correction', 'recession'
]