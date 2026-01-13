import pandas as pd
import re
import unicodedata
import string
from config import CLEANING_CONFIG

class StockTwitsDataCleaner:
    def __init__(self, config):
        self.config = config
        
    def load_data(self, file_path):
        """加载CSV文件"""
        try:
            df = pd.read_csv(file_path)
            print(f"成功加载文件: {file_path}, 数据量: {len(df)}")
            return df
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return None
    
    def remove_empty_entries(self, df, text_column='Text'):
        """去除无内容entry"""
        initial_count = len(df)
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].str.strip() != '']
        removed_count = initial_count - len(df)
        print(f"去除空内容: {removed_count} 条")
        return df
    
    def unicode_normalization(self, text):
        """Unicode规范化"""
        if pd.isna(text):
            return text
        return unicodedata.normalize('NFKC', str(text))
    
    def remove_web_content(self, text):
        """去除网页无用信息"""
        if pd.isna(text):
            return text
        
        text = str(text)
        # 移除URL
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除特定网页关键词
        for keyword in self.config['web_keywords']:
            text = text.replace(keyword, '')
        
        return text
    
    def normalize_whitespace(self, text):
        """压缩多个空格"""
        if pd.isna(text):
            return text
        return re.sub(r'\s+', ' ', str(text)).strip()
    
    def clean_special_chars(self, text):
        """保留英语字母、数字、常用符号"""
        if pd.isna(text):
            return text
        
        text = str(text)
        # 只保留允许的字符
        allowed_chars_pattern = f"[{re.escape(self.config['allowed_chars'])}]"
        cleaned_text = ''.join(char for char in text if re.match(allowed_chars_pattern, char))
        
        return cleaned_text
    
    def is_english(self, text):
        """检测是否为英语内容（大于70%）"""
        if pd.isna(text) or len(str(text).strip()) < 10:
            return False
        
        try:
            # 计算英语字符比例
            text_str = str(text)
            english_chars = sum(1 for char in text_str if char in string.ascii_letters + string.whitespace)
            total_chars = len(text_str.replace(' ', ''))
            
            if total_chars == 0:
                return False
                
            english_ratio = english_chars / total_chars
            return english_ratio >= self.config['english_threshold']
            
        except Exception:
            return False
    
    def remove_short_texts(self, df, text_column='Text'):
        """去除短于5词的文本"""
        initial_count = len(df)
        df = df[df[text_column].str.split().str.len() >= self.config['min_words']]
        removed_count = initial_count - len(df)
        print(f"去除短文本: {removed_count} 条")
        return df
    
    def check_letter_ratio(self, text):
        """检查字母占比"""
        if pd.isna(text):
            return False
        
        text_str = str(text)
        if len(text_str) == 0:
            return False
        
        # 计算字母占比
        letters = sum(1 for char in text_str if char.isalpha())
        total_chars = len(text_str.replace(' ', ''))
        
        if total_chars == 0:
            return False
            
        letter_ratio = letters / total_chars
        return letter_ratio >= self.config['letter_ratio_threshold']
    
    def calculate_tag_ratio(self, text):
        """计算tag在文本中的占比"""
        if pd.isna(text):
            return 0
        
        text_str = str(text)
        words = text_str.split()
        
        if len(words) == 0:
            return 0
        
        # 统计tag数量（$后接非数字内容的词）
        tag_count = 0
        for word in words:
            if re.match(self.config['tag_pattern'], word):
                tag_count += 1
        
        # 计算tag占比
        tag_ratio = tag_count / len(words)
        return tag_ratio
    
    def remove_high_tag_ratio_texts(self, df, text_column='Text'):
        """剔除tag占比大于90%的数据"""
        initial_count = len(df)
        
        # 计算每条数据的tag占比
        tag_ratios = df[text_column].apply(self.calculate_tag_ratio)
        
        # 保留tag占比小于等于90%的数据
        df = df[tag_ratios <= self.config['tag_ratio_threshold']]
        
        removed_count = initial_count - len(df)
        print(f"去除tag占比过高内容: {removed_count} 条")
        
        return df
    
    def truncate_text(self, text):
        """截断文本到500词以内"""
        if pd.isna(text):
            return text
            
        words = str(text).split()
        if len(words) > self.config['max_words']:
            truncated = ' '.join(words[:self.config['max_words']])
            return truncated
        return text
    
    def format_output(self, text):
        """格式化输出"""
        if pd.isna(text):
            return text
        return f"{self.config['output_prefix']} {text}"
    
    def remove_duplicates(self, df, text_column='Text'):
        """去重"""
        initial_count = len(df)
        df = df.drop_duplicates(subset=[text_column])
        removed_count = initial_count - len(df)
        print(f"去除重复: {removed_count} 条")
        return df
    
    def clean_data(self, df, text_column='Text'):
        """执行完整的数据清洗流程"""
        if df is None:
            return None
            
        print("开始数据清洗流程...")
        
        # 1. 去除无内容entry
        df = self.remove_empty_entries(df, text_column)
        
        # 2. Unicode规范化
        df[text_column] = df[text_column].apply(self.unicode_normalization)
        
        # 3. 去除网页无用信息
        df[text_column] = df[text_column].apply(self.remove_web_content)
        
        # 4. 规范化空白
        df[text_column] = df[text_column].apply(self.normalize_whitespace)
        
        # 5. 清理特殊字符
        df[text_column] = df[text_column].apply(self.clean_special_chars)
        
        # 6. 语言筛选（英语大于70%）
        initial_count = len(df)
        df = df[df[text_column].apply(self.is_english)]
        removed_count = initial_count - len(df)
        print(f"去除非英语内容: {removed_count} 条")
        
        # 7. 去重
        df = self.remove_duplicates(df, text_column)
        
        # 8. 去除短于5词
        df = self.remove_short_texts(df, text_column)
        
        # 9. 去除字母占比小于70%
        initial_count = len(df)
        df = df[df[text_column].apply(self.check_letter_ratio)]
        removed_count = initial_count - len(df)
        print(f"去除字母占比不足内容: {removed_count} 条")
        
        # 10. 剔除tag占比大于90%的数据
        df = self.remove_high_tag_ratio_texts(df, text_column)
        
        # 11. 截断到500词
        df[text_column] = df[text_column].apply(self.truncate_text)
        
        # 12. 格式化输出
        df[text_column] = df[text_column].apply(self.format_output)
        
        print(f"清洗完成，剩余数据: {len(df)} 条")
        return df
    
    def process_file(self, input_file, output_file):
        """处理单个文件"""
        print(f"\n正在处理文件: {input_file}")
        
        # 加载数据
        df = self.load_data(input_file)
        if df is None:
            return False
        
        # 清洗数据
        cleaned_df = self.clean_data(df)
        if cleaned_df is None or len(cleaned_df) == 0:
            print("清洗后无有效数据")
            return False
        
        # 保存结果 - 保留所有原始列
        try:
            cleaned_df.to_csv(output_file, index=False, encoding=self.config['encoding'])
            print(f"结果已保存至: {output_file}")
            return True
        except Exception as e:
            print(f"保存文件失败: {e}")
            return False

def main():
    """主函数"""
    # 文件列表
    files_to_process = [
        'COMPQ_Full_2019_2024.csv',
        'DIA_Full_2019_2024.csv', 
        'GLD_Full_2019_2024.csv',
        'SLV_Full_2019_2024.csv',
        'SPX_Full_2019_2024.csv',
        'USO_Full_2019_2024.csv',
        'COMPQ_Full_2012_2018.csv',
        'DIA_Full_2012_2018.csv', 
        'GLD_Full_2012_2018.csv',
        'SLV_Full_2012_2018.csv',
        'SPX_Full_2012_2018.csv',
        'USO_Full_2012_2018.csv'
    ]
    
    # 创建清洗器实例
    cleaner = StockTwitsDataCleaner(CLEANING_CONFIG)
    
    # 处理每个文件
    success_count = 0
    for input_file in files_to_process:
        output_file = f"cleaned_{input_file}"
        if cleaner.process_file(input_file, output_file):
            success_count += 1
    
    print(f"\n处理完成！成功处理 {success_count}/{len(files_to_process)} 个文件")

if __name__ == "__main__":
    main()
