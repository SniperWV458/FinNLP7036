# config.py - 配置参数文件
# 文本清洗参数配置
CLEANING_CONFIG = {
    # 语言检测阈值
    'language_threshold': 0.7,
    
    # 文本长度过滤
    'min_words': 5,
    'max_words': 500,
    'min_alpha_ratio': 0.7,
    
    # 字符过滤规则
    'allowed_chars': r'[a-zA-Z0-9\s\%\+\.\?\!\$\,\-\:\;\(\)\@\#\&\*]',
    
    # 网页无用信息关键词
    'remove_keywords': [
        'cookie', 'privacy policy', 'terms of use', 'subscribe', 'newsletter',
        'related articles', 'share this', 'comment', 'login', 'sign up',
        'advertisement', 'sponsored content', 'follow us', 'copyright',
        'all rights reserved', 'reprint', 'click here', 'read more',
        'sign in', 'register', 'log in', 'sign out', 'follow'
    ],
    
    # 网页导航栏关键词
    'navigation_keywords': [
        'home', 'news', 'business', 'sports', 'entertainment', 'technology',
        'health', 'science', 'world', 'politics', 'opinion', 'life',
        'travel', 'video', 'photos', 'weather', 'market data',
        'search', 'menu', 'navigation', 'navbar', 'sidebar'
    ]
}

# 爬虫配置
CRAWLER_CONFIG = {
    'max_retries': 3,
    'timeout': 20,
    'delay_between_requests': 2.0,
    'max_workers': 5,  # 并发线程数
    
    # User-Agent列表
    'user_agents': [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    ],
    
    # 网站特定规则
    'site_rules': {
        'marketwatch.com': {'delay': 3.0},
        'wsj.com': {'delay': 5.0},
        'bloomberg.com': {'delay': 4.0},
        'reuters.com': {'delay': 2.0},
        'ft.com': {'delay': 4.0},
        'cnbc.com': {'delay': 3.0},
        'finance.yahoo.com': {'delay': 2.0},
        'investing.com': {'delay': 3.0},
    },
    
    # 跳过域名
    'skip_domains': [
        'stlouisstar.com', 'coloradostar.com', 'batonrougepost.com',
        'milwaukeesun.com', 'orlandoecho.com', 'clevelandstar.com',
        'iranherald.com', 'torontotelegraph.com', 'hongkongherald.com',
        'philippinetimes.com', 'livecharts.co.uk'
    ],
    
    # 已知的URL重定向映射
    'url_redirects': {
        'http://www.marketwatch.com/(S(': 'https://www.marketwatch.com/',
        'http://www.nasdaq.com/article/': 'https://www.nasdaq.com/articles/',
    }
}
