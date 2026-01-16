import os
import pandas as pd
import re
import unicodedata
import string
import numpy as np
from collections import defaultdict
from config import CLEANING_CONFIG

class StockTwitsDataCleaner:
    def __init__(self, config):
        self.config = config
        self.statistics = {}  # 存储统计信息
        
    def load_data(self, file_path):
        """加载CSV文件"""
        try:
            df = pd.read_csv(file_path)
            print(f"成功加载文件: {file_path}, 数据量: {len(df)}")
            return df
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return None
    
    def calculate_text_length_stats(self, df, text_column='Text'):
        """计算文本长度统计"""
        if df is None or len(df) == 0:
            return {'mean': 0, 'median': 0, 'word_count': []}
        
        # 分别计算字符数和词数
        char_counts = df[text_column].str.len()
        word_counts = df[text_column].str.split().str.len()
        
        return {
            'char_mean': char_counts.mean(),
            'char_median': char_counts.median(),
            'word_mean': word_counts.mean(),
            'word_median': word_counts.median(),
            'char_count': char_counts.tolist(),
            'word_count': word_counts.tolist()
        }
    
    def calculate_char_ratio(self, text):
        """计算英文字符和数字字符占比"""
        if pd.isna(text) or len(str(text)) == 0:
            return {'english_ratio': 0, 'digit_ratio': 0}
        
        text_str = str(text)
        total_chars = len(text_str.replace(' ', ''))
        
        if total_chars == 0:
            return {'english_ratio': 0, 'digit_ratio': 0}
        
        english_chars = sum(1 for char in text_str if char in string.ascii_letters)
        digit_chars = sum(1 for char in text_str if char.isdigit())
        
        return {
            'english_ratio': english_chars / total_chars if total_chars > 0 else 0,
            'digit_ratio': digit_chars / total_chars if total_chars > 0 else 0
        }
    
    def count_keyword_frequency(self, text, keyword_list):
        """统计关键词频率"""
        if pd.isna(text):
            return 0
        
        text_lower = str(text).lower()
        count = 0
        for keyword in keyword_list:
            count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
        return count
    
    def count_direction_keywords(self, text):
        """统计增减方向关键词"""
        if pd.isna(text):
            return {'positive': 0, 'negative': 0}
        
        text_lower = str(text).lower()
        positive_count = 0
        negative_count = 0
        
        for keyword in self.config['direction_keywords']['positive']:
            positive_count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
        
        for keyword in self.config['direction_keywords']['negative']:
            negative_count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
        
        return {'positive': positive_count, 'negative': negative_count}
    
    def count_numbers_in_text(self, text):
        """统计包含数字的条目"""
        if pd.isna(text):
            return False
        return bool(re.search(r'\d', str(text)))
    
    def initialize_asset_stats(self, asset_name):
        """初始化资产统计信息"""
        self.statistics[asset_name] = {
            'original_count': 0,
            'cleaned_count': 0,
            'filter_rate': 0.0,
            # 修正：分别存储清洗前后的字符和词数统计
            'length_before': {'char_mean': 0, 'char_median': 0, 'word_mean': 0, 'word_median': 0},
            'length_after': {'char_mean': 0, 'char_median': 0, 'word_mean': 0, 'word_median': 0},
            'truncated_count': 0,
            'char_ratio_before': {'english': 0, 'digit': 0},
            'char_ratio_after': {'english': 0, 'digit': 0},
            'keyword_freq_before': 0,
            'keyword_freq_after': 0,
            'direction_count_before': {'positive': 0, 'negative': 0},
            'direction_count_after': {'positive': 0, 'negative': 0},
            'number_count_before': 0,
            'number_count_after': 0
        }
    
    def remove_empty_entries(self, df, text_column='Text', asset_name=''):
        """去除无内容entry并记录统计"""
        initial_count = len(df)
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].str.strip() != '']
        removed_count = initial_count - len(df)
        print(f"去除空内容: {removed_count} 条")
        
        if asset_name:
            self.statistics[asset_name]['original_count'] = initial_count
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
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
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
        allowed_chars_pattern = f"[{re.escape(self.config['allowed_chars'])}]"
        cleaned_text = ''.join(char for char in text if re.match(allowed_chars_pattern, char))
        
        return cleaned_text
    
    def is_english(self, text):
        """检测是否为英语内容"""
        if pd.isna(text) or len(str(text).strip()) < 10:
            return False
        
        try:
            text_str = str(text)
            english_chars = sum(1 for char in text_str if char in string.ascii_letters + string.whitespace)
            total_chars = len(text_str.replace(' ', ''))
            
            if total_chars == 0:
                return False
                
            english_ratio = english_chars / total_chars
            return english_ratio >= self.config['english_threshold']
            
        except Exception:
            return False
    
    def remove_short_texts(self, df, text_column='Text', asset_name=''):
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
        
        tag_count = 0
        for word in words:
            if re.match(self.config['tag_pattern'], word):
                tag_count += 1
        
        tag_ratio = tag_count / len(words)
        return tag_ratio
    
    def remove_high_tag_ratio_texts(self, df, text_column='Text', asset_name=''):
        """剔除tag占比大于90%的数据"""
        initial_count = len(df)
        tag_ratios = df[text_column].apply(self.calculate_tag_ratio)
        df = df[tag_ratios <= self.config['tag_ratio_threshold']]
        removed_count = initial_count - len(df)
        print(f"去除tag占比过高内容: {removed_count} 条")
        return df
    
    def truncate_text(self, text, asset_name=''):
        """截断文本到500词以内并记录统计"""
        if pd.isna(text):
            return text
            
        words = str(text).split()
        if len(words) > self.config['max_words']:
            truncated = ' '.join(words[:self.config['max_words']])
            if asset_name:
                self.statistics[asset_name]['truncated_count'] += 1
            return truncated
        return text
    
    def format_output(self, text, output_prefix=None):
        """格式化输出"""
        if pd.isna(text):
            return text
        prefix = output_prefix or self.config['output_prefix']
        return f"{prefix} {text}"
    
    def remove_duplicates(self, df, text_column='Text', asset_name=''):
        """去重"""
        initial_count = len(df)
        df = df.drop_duplicates(subset=[text_column])
        removed_count = initial_count - len(df)
        print(f"去除重复: {removed_count} 条")
        return df
    
    def collect_statistics(self, df, text_column='Text', asset_name='', stage='before'):
        """收集统计信息"""
        if df is None or len(df) == 0:
            return
        
        stats = self.statistics[asset_name]
        
        if stage == 'before':
            # 清洗前统计
            length_stats = self.calculate_text_length_stats(df, text_column)
            stats['length_before'] = {
                'char_mean': length_stats['char_mean'],
                'char_median': length_stats['char_median'],
                'word_mean': length_stats['word_mean'],
                'word_median': length_stats['word_median']
            }
            
            # 字符占比统计
            char_ratios = df[text_column].apply(self.calculate_char_ratio)
            stats['char_ratio_before']['english'] = char_ratios.apply(lambda x: x['english_ratio']).mean()
            stats['char_ratio_before']['digit'] = char_ratios.apply(lambda x: x['digit_ratio']).mean()
            
            # 关键词频率统计
            stats['keyword_freq_before'] = df[text_column].apply(
                lambda x: self.count_keyword_frequency(x, self.config['finance_keywords'])
            ).mean()
            
            # 方向关键词统计
            direction_counts = df[text_column].apply(self.count_direction_keywords)
            stats['direction_count_before']['positive'] = direction_counts.apply(lambda x: x['positive']).sum()
            stats['direction_count_before']['negative'] = direction_counts.apply(lambda x: x['negative']).sum()
            
            # 包含数字的条目数
            stats['number_count_before'] = df[text_column].apply(self.count_numbers_in_text).sum()
            
        elif stage == 'after':
            # 清洗后统计
            length_stats = self.calculate_text_length_stats(df, text_column)
            stats['length_after'] = {
                'char_mean': length_stats['char_mean'],
                'char_median': length_stats['char_median'],
                'word_mean': length_stats['word_mean'],
                'word_median': length_stats['word_median']
            }
            
            stats['cleaned_count'] = len(df)
            stats['filter_rate'] = stats['cleaned_count'] / stats['original_count'] if stats['original_count'] > 0 else 0
            
            # 字符占比统计
            char_ratios = df[text_column].apply(self.calculate_char_ratio)
            stats['char_ratio_after']['english'] = char_ratios.apply(lambda x: x['english_ratio']).mean()
            stats['char_ratio_after']['digit'] = char_ratios.apply(lambda x: x['digit_ratio']).mean()
            
            # 关键词频率统计
            stats['keyword_freq_after'] = df[text_column].apply(
                lambda x: self.count_keyword_frequency(x, self.config['finance_keywords'])
            ).mean()
            
            # 方向关键词统计
            direction_counts = df[text_column].apply(self.count_direction_keywords)
            stats['direction_count_after']['positive'] = direction_counts.apply(lambda x: x['positive']).sum()
            stats['direction_count_after']['negative'] = direction_counts.apply(lambda x: x['negative']).sum()
            
            # 包含数字的条目数
            stats['number_count_after'] = df[text_column].apply(self.count_numbers_in_text).sum()
    
    def clean_data(self, df, text_column='Text', asset_name='', output_prefix=None):
        """执行完整的数据清洗流程"""
        if df is None:
            return None
            
        print("开始数据清洗流程...")
        
        # 初始化资产统计
        self.initialize_asset_stats(asset_name)
        
        # 收集清洗前统计信息
        self.collect_statistics(df, text_column, asset_name, 'before')
        
        # 1. 去除无内容entry
        df = self.remove_empty_entries(df, text_column, asset_name)
        
        # 2. Unicode规范化
        df[text_column] = df[text_column].apply(self.unicode_normalization)
        
        # 3. 去除网页无用信息
        df[text_column] = df[text_column].apply(self.remove_web_content)
        
        # 4. 规范化空白
        df[text_column] = df[text_column].apply(self.normalize_whitespace)
        
        # 5. 清理特殊字符
        df[text_column] = df[text_column].apply(self.clean_special_chars)
        
        # 6. 语言筛选
        initial_count = len(df)
        df = df[df[text_column].apply(self.is_english)]
        removed_count = initial_count - len(df)
        print(f"去除非英语内容: {removed_count} 条")
        
        # 7. 去重
        df = self.remove_duplicates(df, text_column, asset_name)
        
        # 8. 去除短于5词
        df = self.remove_short_texts(df, text_column, asset_name)
        
        # 9. 去除字母占比小于70%
        initial_count = len(df)
        df = df[df[text_column].apply(self.check_letter_ratio)]
        removed_count = initial_count - len(df)
        print(f"去除字母占比不足内容: {removed_count} 条")
        
        # 10. 剔除tag占比大于90%的数据
        df = self.remove_high_tag_ratio_texts(df, text_column, asset_name)
        
        # 11. 截断到500词
        df[text_column] = df[text_column].apply(lambda x: self.truncate_text(x, asset_name))
        
        # 12. 格式化输出
        df[text_column] = df[text_column].apply(
            lambda x: self.format_output(x, output_prefix=output_prefix)
        )
        
        # 收集清洗后统计信息
        self.collect_statistics(df, text_column, asset_name, 'after')
        
        print(f"清洗完成，剩余数据: {len(df)} 条")
        return df
    
    def generate_summary_report(self):
        """生成总体统计报告"""
        if not self.statistics:
            return "无统计信息可用"
        
        report = []
        report.append("=" * 80)
        report.append("数据清洗统计报告")
        report.append("=" * 80)
        
        # 总体统计
        total_original = sum(stats['original_count'] for stats in self.statistics.values())
        total_cleaned = sum(stats['cleaned_count'] for stats in self.statistics.values())
        overall_filter_rate = total_cleaned / total_original if total_original > 0 else 0
        
        report.append(f"总体统计:")
        report.append(f"  总爬取条目数: {total_original:,}")
        report.append(f"  总清洗后条目数: {total_cleaned:,}")
        report.append(f"  总体筛选率: {overall_filter_rate:.2%}")
        report.append("")
        
        # 各资产详细统计
        for asset_name, stats in self.statistics.items():
            report.append(f"资产: {asset_name}")
            report.append(f"  原始条目数: {stats['original_count']:,}")
            report.append(f"  清洗后条目数: {stats['cleaned_count']:,}")
            report.append(f"  筛选率: {stats['filter_rate']:.2%}")
            
            # 修正：分别显示清洗前后的字符和词数统计
            report.append(f"  文本长度统计 - 清洗前:")
            report.append(f"    字符数: 平均{stats['length_before']['char_mean']:.1f}, 中位数{stats['length_before']['char_median']:.1f}")
            report.append(f"    词数: 平均{stats['length_before']['word_mean']:.1f}, 中位数{stats['length_before']['word_median']:.1f}")
            
            report.append(f"  文本长度统计 - 清洗后:")
            report.append(f"    字符数: 平均{stats['length_after']['char_mean']:.1f}, 中位数{stats['length_after']['char_median']:.1f}")
            report.append(f"    词数: 平均{stats['length_after']['word_mean']:.1f}, 中位数{stats['length_after']['word_median']:.1f}")
            
            report.append(f"  被截断条目数: {stats['truncated_count']:,}")
            report.append(f"  字符占比 - 清洗前: 英文{stats['char_ratio_before']['english']:.2%}, 数字{stats['char_ratio_before']['digit']:.2%}")
            report.append(f"  字符占比 - 清洗后: 英文{stats['char_ratio_after']['english']:.2%}, 数字{stats['char_ratio_after']['digit']:.2%}")
            report.append(f"  金融关键词频率 - 清洗前: 平均每条{stats['keyword_freq_before']:.2f}个")
            report.append(f"  金融关键词频率 - 清洗后: 平均每条{stats['keyword_freq_after']:.2f}个")
            report.append(f"  方向关键词 - 清洗前: 正面{stats['direction_count_before']['positive']:,}个, 负面{stats['direction_count_before']['negative']:,}个")
            report.append(f"  方向关键词 - 清洗后: 正面{stats['direction_count_after']['positive']:,}个, 负面{stats['direction_count_after']['negative']:,}个")
            report.append(f"  包含数字条目 - 清洗前: {stats['number_count_before']:,}条")
            report.append(f"  包含数字条目 - 清洗后: {stats['number_count_after']:,}条")
            report.append("")
        
        return "\n".join(report)
    
    def save_statistics_to_csv(self, output_file='cleaning_statistics.csv'):
        """保存统计信息到CSV文件"""
        if not self.statistics:
            print("无统计信息可保存")
            return False
        
        try:
            # 准备数据
            rows = []
            for asset_name, stats in self.statistics.items():
                row = {
                    'asset': asset_name,
                    'original_count': stats['original_count'],
                    'cleaned_count': stats['cleaned_count'],
                    'filter_rate': stats['filter_rate'],
                    # 修正：分别保存清洗前后的字符和词数统计
                    'char_length_before_mean': stats['length_before']['char_mean'],
                    'char_length_before_median': stats['length_before']['char_median'],
                    'word_length_before_mean': stats['length_before']['word_mean'],
                    'word_length_before_median': stats['length_before']['word_median'],
                    'char_length_after_mean': stats['length_after']['char_mean'],
                    'char_length_after_median': stats['length_after']['char_median'],
                    'word_length_after_mean': stats['length_after']['word_mean'],
                    'word_length_after_median': stats['length_after']['word_median'],
                    'truncated_count': stats['truncated_count'],
                    'english_ratio_before': stats['char_ratio_before']['english'],
                    'digit_ratio_before': stats['char_ratio_before']['digit'],
                    'english_ratio_after': stats['char_ratio_after']['english'],
                    'digit_ratio_after': stats['char_ratio_after']['digit'],
                    'keyword_freq_before': stats['keyword_freq_before'],
                    'keyword_freq_after': stats['keyword_freq_after'],
                    'positive_keywords_before': stats['direction_count_before']['positive'],
                    'negative_keywords_before': stats['direction_count_before']['negative'],
                    'positive_keywords_after': stats['direction_count_after']['positive'],
                    'negative_keywords_after': stats['direction_count_after']['negative'],
                    'number_count_before': stats['number_count_before'],
                    'number_count_after': stats['number_count_after']
                }
                rows.append(row)
            
            # 创建DataFrame并保存
            stats_df = pd.DataFrame(rows)
            stats_df.to_csv(output_file, index=False, encoding=self.config['encoding'])
            print(f"统计信息已保存至: {output_file}")
            return True
            
        except Exception as e:
            print(f"保存统计信息失败: {e}")
            return False
    
    def process_file(self, input_file, output_file, text_column='Text'):
        """处理单个文件"""
        print(f"\n正在处理文件: {input_file}")
        
        # 从文件名提取资产名称
        file_name = os.path.basename(input_file)
        asset_name = os.path.splitext(file_name)[0].replace('_merged', '')
        
        # 加载数据
        df = self.load_data(input_file)
        if df is None:
            return False
        if text_column not in df.columns:
            print(f"缺少文本列 '{text_column}'，可用列: {list(df.columns)}")
            return False
        
        # 清洗数据
        normalized_path = os.path.normpath(input_file).lower()
        news_marker = os.path.normpath(os.path.join('data', 'news')).lower()
        news_marker_alt = os.path.normpath(os.path.join('data', 'textual_data', 'news')).lower()
        output_prefix = '[NEWS]' if (news_marker in normalized_path or news_marker_alt in normalized_path) else None
        cleaned_df = self.clean_data(
            df,
            text_column=text_column,
            asset_name=asset_name,
            output_prefix=output_prefix
        )
        if cleaned_df is None or len(cleaned_df) == 0:
            print("清洗后无有效数据")
            return False
        
        # 保存结果
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
    """files_to_process = [
        'COMPQ_merged.csv',
        'DIA_merged.csv', 
        'GLD_merged.csv',
        'SLV_merged.csv',
        'SPX_merged.csv',
        'USO_merged.csv'
    ]"""

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    base_dir = os.path.join(repo_root, 'data', 'textual_data', 'news')
    files_to_process = [
        'DJI_2017-2024_all.csv',
        'GOLD_2017-2024_all.csv',
        'GSPC_2017-2024_all.csv',
        'IXIC_2017-2024_all.csv',
        'OIL_2017-2024_all.csv',
        'SILVER_2017-2024_all.csv',
    ]
    
    # 创建清洗器实例
    cleaner = StockTwitsDataCleaner(CLEANING_CONFIG)
    
    # 处理每个文件
    success_count = 0
    for input_file in files_to_process:
        input_path = os.path.join(base_dir, input_file)
        output_file = f"cleaned_{input_file}"
        output_path = os.path.join(base_dir, output_file)
        if cleaner.process_file(input_path, output_path, text_column='title'):
            success_count += 1
    
    # 生成统计报告
    print("\n" + "="*80)
    print("生成统计报告...")
    print("="*80)
    
    # 打印详细报告
    report = cleaner.generate_summary_report()
    print(report)
    
    # 保存统计信息到CSV
    cleaner.save_statistics_to_csv('cleaning_statistics_summary.csv')
    
    print(f"\n处理完成！成功处理 {success_count}/{len(files_to_process)} 个文件")


if __name__ == "__main__":
    main()
