# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 20:17:17 2026

@author: 25010
"""

# 数据清洗配置参数
CLEANING_CONFIG = {
    # 基本清洗参数
    'min_words': 5,                    # 最小词数
    'max_words': 500,                  # 最大词数
    'english_threshold': 0.7,          # 英语内容阈值
    'letter_ratio_threshold': 0.7,     # 字母占比阈值
    
    # 文本处理参数
    'allowed_chars': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 %+..?!$',  # 允许的字符
    'web_keywords': ['http', 'www', '.com', '.org', 'href=', '<div', '<p', '<br', '&amp;', '&#39;'],  # 网页关键词
    
    # 输出格式
    'output_prefix': '[STOCKTWITS]',   # 输出前缀
    'encoding': 'utf-8'                # 文件编码
}