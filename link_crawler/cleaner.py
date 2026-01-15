# cleaner.py - 文本清洗功能
import re
import pandas as pd
import unicodedata
import hashlib
from config import CLEANING_CONFIG

class TextCleaner:
    def __init__(self):
        self.config = CLEANING_CONFIG
        self.remove_patterns = [
            r'Advertisement', r'Sponsored', r'Click here.*', r'Read more.*',
            r'Sign up.*', r'Subscribe.*', r'Follow us.*', r'Like us.*',
            r'Share this.*', r'Comment.*', r'Related:.*', r'Also read.*',
            r'More from.*', r'Previous article.*', r'Next article.*',
            r'©.*', r'Copyright.*', r'All rights reserved.*',
            r'Terms of Use.*', r'Privacy Policy.*', r'Cookie Policy.*',
            r'var\s+.*?=.*?;',  # JavaScript变量
            r'function\s+.*?\(.*?\)\s*\{.*?\}',  # JavaScript函数
            r'<script.*?>.*?</script>',  # 脚本标签
            r'<style.*?>.*?</style>',  # 样式标签
            r'\[.*?\]',  # 方括号内容
            r'\(.*?\)',  # 括号内容
        ]
    
    def is_english_content(self, text, threshold=0.7):
        """检测文本是否为英文内容"""
        if not text or not isinstance(text, str) or len(text.strip()) < 20:
            return False
        
        try:
            # 统计英文字母、数字、空格和标点
            english_chars = len(re.findall(r'[a-zA-Z0-9\s\.,!?;:\'"\-\(\)]', text))
            total_chars = len(text)
            
            if total_chars == 0:
                return False
            
            english_ratio = english_chars / total_chars
            
            # 额外的检查：常见英语单词比例
            english_words = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any',
                           'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get',
                           'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see',
                           'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say',
                           'she', 'too', 'use']
            
            words = text.lower().split()
            english_word_count = sum(1 for word in words if word in english_words)
            
            if len(words) > 0:
                english_word_ratio = english_word_count / min(len(words), 100)
                return english_ratio >= threshold or english_word_ratio > 0.1
            else:
                return english_ratio >= threshold
                
        except:
            return False
    
    def unicode_normalize(self, text):
        """Unicode标准化"""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # NFKC规范化
            text = unicodedata.normalize('NFKC', text)
            
            # 处理特殊字符
            text = text.replace('\u201c', '"')  # 左双引号
            text = text.replace('\u201d', '"')  # 右双引号
            text = text.replace('\u2018', "'")  # 左单引号
            text = text.replace('\u2019', "'")  # 右单引号
            text = text.replace('\u2013', '-')  # 短横线
            text = text.replace('\u2014', '-')  # 长横线
            text = text.replace('\u2026', '...')  # 省略号
            
            return text
        except:
            return text
    
    def remove_web_junk(self, text):
        """移除网页无用信息"""
        if not text or not isinstance(text, str):
            return ""
        
        text = str(text)
        
        # 移除模式匹配的内容
        for pattern in self.remove_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE | re.DOTALL)
        
        # 移除特定关键词行
        lines = text.split('\n')
        clean_lines = []
        
        remove_keywords = self.config['remove_keywords'] + self.config['navigation_keywords']
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # 过滤空行
            if not line_lower or len(line_lower) < 10:
                continue
            
            # 检查是否包含排除关键词
            contains_junk = any(junk in line_lower for junk in remove_keywords)
            
            # 检查是否看起来像URL
            looks_like_url = re.search(r'https?://|www\.|\.[a-z]{2,3}/', line_lower)
            
            # 检查是否像导航
            looks_like_nav = (len(line_lower.split()) < 5 and 
                            (line_lower.startswith('home') or 
                             line_lower.startswith('news') or
                             line_lower.startswith('about')))
            
            if not contains_junk and not looks_like_url and not looks_like_nav:
                # 移除行内的广告文本
                line_clean = re.sub(r'\b(ad|ads|advertisement|sponsored)\b', '', 
                                  line_lower, flags=re.IGNORECASE)
                if len(line_clean.strip()) > 5:
                    clean_lines.append(line_clean.strip())
        
        return '\n'.join(clean_lines)
    
    def normalize_whitespace(self, text):
        """标准化空白字符"""
        if not text or not isinstance(text, str):
            return ""
        
        # 移除所有空白字符（空格、制表符、换行等）并用单个空格替换
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_special_chars(self, text):
        """移除非常用符号和非英语字符"""
        if not text or not isinstance(text, str):
            return ""
        
        # 保留的字符：字母、数字、空格、常用标点
        allowed = r'[a-zA-Z0-9\s\%\+\-\.,!?\$\'\";:\(\)@#&]'
        text = re.sub(r'[^a-zA-Z0-9\s\%\+\-\.,!?\$\'\";:\(\)@#&]', ' ', text)
        
        # 清理多余空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def calculate_text_stats(self, text):
        """计算文本统计信息"""
        if not text or not isinstance(text, str):
            return 0, 0, 0
        
        words = text.split()
        word_count = len(words)
        
        # 计算字母比例
        alpha_chars = sum(1 for char in text if char.isalpha())
        total_chars = len(text.replace(' ', ''))
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        
        # 计算句子数量
        sentence_count = len(re.findall(r'[.!?]+', text))
        
        return word_count, alpha_ratio, sentence_count
    
    def format_content(self, title, content):
        """格式化内容为[NEWS]标题.内容格式"""
        if not content or not isinstance(content, str):
            return ""
        
        # 清理标题
        if title and isinstance(title, str):
            clean_title = re.sub(r'[^\w\s\-]', '', title)[:150].strip()
            if len(clean_title.split()) > 2:  # 确保标题有内容
                return f"[NEWS]{clean_title}.{content}"
        
        return f"[NEWS]{content}"
    
    def generate_content_hash(self, text):
        """生成内容哈希值用于去重"""
        if not text or not isinstance(text, str):
            return ""
        
        # 使用前100个字符生成哈希
        text_for_hash = text[:1000].lower()
        text_for_hash = re.sub(r'\s+', ' ', text_for_hash).strip()
        
        return hashlib.md5(text_for_hash.encode('utf-8')).hexdigest()
    
    def clean_text(self, text, title=""):
        """完整的文本清洗流程"""
        if not text or not isinstance(text, str) or len(text.strip()) < 20:
            return ""
        
        original_text = text
        
        try:
            # 步骤1: Unicode标准化
            text = self.unicode_normalize(text)
            
            # 步骤2: 移除网页无用信息
            text = self.remove_web_junk(text)
            
            # 步骤3: 移除特殊字符
            text = self.remove_special_chars(text)
            
            # 步骤4: 标准化空白字符
            text = self.normalize_whitespace(text)
            
            # 步骤5: 计算统计信息
            word_count, alpha_ratio, sentence_count = self.calculate_text_stats(text)
            
            # 步骤6: 应用过滤规则
            if (word_count < self.config['min_words'] or 
                word_count > self.config['max_words'] or
                alpha_ratio < self.config['min_alpha_ratio'] or
                sentence_count < 1):
                return ""
            
            # 步骤7: 截断到最大词数
            words = text.split()
            if len(words) > self.config['max_words']:
                text = ' '.join(words[:self.config['max_words']])
            
            # 步骤8: 格式化
            text = self.format_content(title, text)
            
            # 步骤9: 生成哈希
            text_hash = self.generate_content_hash(text)
            
            return text
            
        except Exception as e:
            print(f"文本清洗错误: {e}")
            # 如果清洗失败，返回原始文本的基本清理版本
            try:
                simple_clean = self.normalize_whitespace(
                    self.remove_special_chars(original_text[:1000])
                )
                return self.format_content(title, simple_clean) if len(simple_clean.split()) >= 5 else ""
            except:
                return ""
    
    def clean_dataframe(self, df):
        """清洗整个DataFrame"""
        if df.empty:
            return df
        
        original_count = len(df)
        print(f"开始清洗数据，原始记录数: {original_count}")
        
        # 步骤1: 移除空内容
        df = df[df['raw_content'].notna()]
        df = df[df['raw_content'].apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 20)]
        print(f"移除空内容后: {len(df)}")
        
        # 步骤2: 语言筛选
        df['is_english'] = df['raw_content'].apply(
            lambda x: self.is_english_content(x, self.config['language_threshold'])
        )
        df = df[df['is_english']]
        print(f"英语筛选后: {len(df)}")
        
        # 步骤3: 文本清洗
        print("正在进行文本清洗...")
        cleaned_contents = []
        for idx, row in df.iterrows():
            cleaned = self.clean_text(row['raw_content'], row.get('title', ''))
            cleaned_contents.append(cleaned)
            
            if (idx + 1) % 100 == 0:
                print(f"清洗进度: {idx + 1}/{len(df)}")
        
        df['cleaned_content'] = cleaned_contents
        
        # 步骤4: 移除清洗后为空的内容
        df = df[df['cleaned_content'].notna() & (df['cleaned_content'] != "")]
        print(f"清洗后非空: {len(df)}")
        
        # 步骤5: 去重
        df['content_hash'] = df['cleaned_content'].apply(self.generate_content_hash)
        df = df.drop_duplicates(subset=['content_hash'])
        print(f"去重后: {len(df)}")
        
        # 步骤6: 添加统计信息
        df['word_count'] = df['cleaned_content'].apply(lambda x: len(str(x).split()))
        df['char_count'] = df['cleaned_content'].apply(lambda x: len(str(x)))
        
        # 保留需要的列
        columns_to_keep = [
            'url', 'processed_url', 'seendate', 'year', 'month', 'domain', 
            'asset_name', 'raw_content', 'cleaned_content', 'content_hash',
            'word_count', 'char_count', 'crawl_success'
        ]
        
        # 只保留存在的列
        available_columns = [col for col in columns_to_keep if col in df.columns]
        final_df = df[available_columns]
        
        print(f"清洗完成: {len(final_df)}/{original_count} ({len(final_df)/original_count*100:.1f}%)")
        
        return final_df