# Data cleaning configuration parameters
CLEANING_CONFIG = {
    # Basic cleaning parameters
    'min_words': 5,                    # Minimum word count
    'max_words': 500,                  # Maximum word count
    'english_threshold': 0.7,          # English content threshold
    'letter_ratio_threshold': 0.7,     # Letter ratio threshold
    'tag_ratio_threshold': 0.75,       # Tag ratio threshold
    
    # Text processing parameters
    'allowed_chars': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 %+..?!$',
    'web_keywords': ['http', 'www', '.com', '.org', 'href=', '<div', '<p', '<br', '&amp;', '&#39;'],
    'tag_pattern': r'\$[^\d\s]+',
    
    # Output format
    'output_prefix': '[STOCKTWITS]',
    'encoding': 'utf-8',
    
    # New: Keyword lists for statistics
    'finance_keywords': [
        # Stock related
        'stock', 'stocks', 'equity', 'equities', 'share', 'shares',
        # Market related
        'market', 'trading', 'trade', 'invest', 'investment', 'portfolio',
        # Price related
        'price', 'prices', 'valuation', 'value', 'premium', 'discount',
        # Financial indicators
        'PE', 'P/E', 'EPS', 'dividend', 'yield', 'ROI', 'return',
        # Asset classes
        'bond', 'bonds', 'commodity', 'commodities', 'future', 'futures',
        # Economic indicators
        'inflation', 'deflation', 'GDP', 'unemployment', 'interest rate',
        # Company related
        'earnings', 'revenue', 'profit', 'loss', 'merger', 'acquisition',
        # Trading related
        'bull', 'bear', 'rally', 'crash', 'volatility', 'liquidity'
    ],
    
    'direction_keywords': {
        'positive': ['+', 'rise', 'rising', 'rose', 'up', 'gain', 'gaining', 'gained', 
                    'higher', 'high', 'peak', 'surge', 'jump', 'increase', 'increasing',
                    'bull', 'bullish', 'buy', 'long', 'outperform', 'beat'],
        'negative': ['-', 'fall', 'falling', 'fell', 'down', 'loss', 'losing', 'lost',
                    'lower', 'low', 'bottom', 'plunge', 'drop', 'decrease', 'decreasing',
                    'bear', 'bearish', 'sell', 'short', 'underperform', 'miss']
    }
}
