# 数据清洗配置参数
CLEANING_CONFIG = {
    # 基本清洗参数
    'min_words': 5,                    # 最小词数
    'max_words': 500,                  # 最大词数
    'english_threshold': 0.7,          # 英语内容阈值
    'letter_ratio_threshold': 0.7,     # 字母占比阈值
    'tag_ratio_threshold': 0.75,       # tag占比阈值
    
    # 文本处理参数
    'allowed_chars': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 %+..?!$',
    'web_keywords': ['http', 'www', '.com', '.org', 'href=', '<div', '<p', '<br', '&amp;', '&#39;'],
    'tag_pattern': r'\$[^\d\s]+',
    
    # 输出格式
    'output_prefix': '[STOCKTWITS]',
    'encoding': 'utf-8',
    
    # 新增：统计相关的关键词列表
    'finance_keywords': [
        # 股票相关
        'stock', 'stocks', 'equity', 'equities', 'share', 'shares',
        # 市场相关
        'market', 'trading', 'trade', 'invest', 'investment', 'portfolio',
        # 价格相关
        'price', 'prices', 'valuation', 'value', 'premium', 'discount',
        # 金融指标
        'PE', 'P/E', 'EPS', 'dividend', 'yield', 'ROI', 'return',
        # 资产类别
        'bond', 'bonds', 'commodity', 'commodities', 'future', 'futures',
        # 经济指标
        'inflation', 'deflation', 'GDP', 'unemployment', 'interest rate',
        # 公司相关
        'earnings', 'revenue', 'profit', 'loss', 'merger', 'acquisition',
        # 交易相关
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
