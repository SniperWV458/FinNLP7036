# main.py - 主程序
import pandas as pd
import os
import sys
from datetime import datetime
from crawler import NewsCrawler
from cleaner import TextCleaner
import argparse
import warnings
import multiprocessing as mp
warnings.filterwarnings('ignore')

def validate_data_file(file_path):
    """验证数据文件"""
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return False
    
    try:
        df = pd.read_csv(file_path, nrows=5)  # 只读取前5行进行验证
        required_columns = ['url']
        for col in required_columns:
            if col not in df.columns:
                print(f"错误: 文件缺少必要列 '{col}' - {file_path}")
                return False
        return True
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path} - {e}")
        return False

def process_single_file(input_file, output_dir='data', sample_size=None, use_threading=True, 
                        use_multiprocessing=None, use_proxy=None):
    """
    处理单个文件
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
        sample_size: 样本大小
        use_threading: 是否使用多线程
        use_multiprocessing: None (使用配置), True (强制多进程), False (使用线程)
        use_proxy: None (使用配置), True (强制使用代理), False (不使用代理)
    """
    print(f"\n{'='*60}")
    print(f"开始处理文件: {input_file}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*60)
    
    # 验证文件
    if not validate_data_file(input_file):
        return None
    
    # 读取原始数据
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"原始数据行数: {len(df)}")
        print(f"列名: {list(df.columns)}")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file, encoding='ISO-8859-1')
            print(f"使用ISO-8859-1编码读取文件")
        except:
            print(f"错误: 无法读取文件编码")
            return None
    except Exception as e:
        print(f"错误: 读取文件失败 - {e}")
        return None
    
    # 确保必要的列存在
    if 'url' not in df.columns:
        print("错误: 数据文件必须包含'url'列")
        return None
    
    # 添加缺失的列
    for col in ['seendate', 'year', 'month', 'domain', 'asset_name']:
        if col not in df.columns:
            df[col] = ''
            print(f"警告: 添加缺失列 '{col}'")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output = os.path.join(output_dir, f"{base_name}_raw_{timestamp}.csv")
    clean_output = os.path.join(output_dir, f"{base_name}_cleaned_{timestamp}.csv")
    stats_output = os.path.join(output_dir, f"{base_name}_stats_{timestamp}.txt")
    
    # 步骤1: 爬取网页内容
    print("\n" + "-"*60)
    print("步骤1: 开始爬取网页内容")
    print("-"*60)
    
    crawler = NewsCrawler(use_proxy=use_proxy)
    raw_df = crawler.process_batch(
        df, 
        output_file=raw_output, 
        sample_size=sample_size,
        use_threading=use_threading,
        use_multiprocessing=use_multiprocessing
    )
    
    if raw_df is None or raw_df.empty:
        print("警告: 没有爬取到任何数据")
        return None
    
    # 步骤2: 文本清洗
    print("\n" + "-"*60)
    print("步骤2: 开始文本清洗")
    print("-"*60)
    
    cleaner = TextCleaner()
    cleaned_df = cleaner.clean_dataframe(raw_df)
    
    if cleaned_df.empty:
        print("警告: 清洗后没有有效数据")
        return None
    
    # 保存清洗后的数据
    cleaned_df.to_csv(clean_output, index=False, encoding='utf-8')
    print(f"清洗后数据已保存至: {clean_output}")
    
    # 步骤3: 生成统计报告
    print("\n" + "-"*60)
    print("步骤3: 生成统计报告")
    print("-"*60)
    
    generate_stats_report(cleaned_df, raw_df, stats_output)
    
    return cleaned_df

def generate_stats_report(cleaned_df, raw_df, stats_file):
    """生成统计报告"""
    stats = []
    
    stats.append("="*60)
    stats.append("数据处理统计报告")
    stats.append("="*60)
    stats.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stats.append("")
    
    # 爬取统计
    if 'crawl_success' in raw_df.columns:
        total_crawls = len(raw_df)
        successful_crawls = raw_df['crawl_success'].sum()
        success_rate = (successful_crawls / total_crawls * 100) if total_crawls > 0 else 0
        
        stats.append("1. 爬取统计:")
        stats.append(f"   总URL数: {total_crawls}")
        stats.append(f"   成功爬取: {successful_crawls}")
        stats.append(f"   爬取成功率: {success_rate:.1f}%")
        stats.append("")
    
    # 清洗统计
    stats.append("2. 清洗统计:")
    stats.append(f"   原始记录数: {len(raw_df)}")
    stats.append(f"   清洗后记录数: {len(cleaned_df)}")
    stats.append(f"   保留比例: {(len(cleaned_df)/len(raw_df)*100):.1f}%")
    stats.append("")
    
    # 内容统计
    if 'word_count' in cleaned_df.columns:
        avg_words = cleaned_df['word_count'].mean()
        max_words = cleaned_df['word_count'].max()
        min_words = cleaned_df['word_count'].min()
        
        stats.append("3. 内容统计:")
        stats.append(f"   平均词数: {avg_words:.0f}")
        stats.append(f"   最多词数: {max_words}")
        stats.append(f"   最少词数: {min_words}")
        stats.append("")
    
    # 域名统计
    if 'domain' in cleaned_df.columns:
        domain_counts = cleaned_df['domain'].value_counts().head(10)
        stats.append("4. 域名分布 (Top 10):")
        for domain, count in domain_counts.items():
            stats.append(f"   {domain}: {count} 条")
        stats.append("")
    
    # 保存统计报告
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats))
    
    print("统计报告:")
    for line in stats:
        print(line)
    
    print(f"\n详细统计已保存至: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description='新闻数据爬取和清洗工具 (支持Luna代理IP轮换)')
    parser.add_argument('--files', nargs='+', help='要处理的CSV文件列表')
    parser.add_argument('--input-dir', default='.', help='输入目录')
    parser.add_argument('--output-dir', default='data', help='输出目录')
    parser.add_argument('--sample', type=int, help='样本大小（测试用）')
    parser.add_argument('--no-threading', action='store_true', help='禁用多线程')
    parser.add_argument('--multiprocessing', action='store_true', help='启用多进程模式（推荐用于代理）')
    parser.add_argument('--no-multiprocessing', action='store_true', help='禁用多进程模式')
    parser.add_argument('--proxy', action='store_true', help='强制启用Luna代理')
    parser.add_argument('--no-proxy', action='store_true', help='强制禁用代理')
    parser.add_argument('--config', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 确定多进程模式
    if args.multiprocessing:
        use_multiprocessing = True
    elif args.no_multiprocessing:
        use_multiprocessing = False
    else:
        use_multiprocessing = None  # 使用配置文件设置
    
    # 确定代理模式
    if args.proxy:
        use_proxy = True
    elif args.no_proxy:
        use_proxy = False
    else:
        use_proxy = None  # 使用配置文件设置
    
    # 确定要处理的文件
    if args.files:
        files_to_process = args.files
    else:
        # 默认处理三个指定文件
        default_files = [
            'DJI_2017-2024_all.csv',
            'IXIC_2017-2024_all.csv', 
            'GSPC_2017-2024_all.csv'
        ]
        files_to_process = []
        for file in default_files:
            file_path = os.path.join(args.input_dir, file)
            if os.path.exists(file_path):
                files_to_process.append(file_path)
            else:
                print(f"警告: 文件不存在，跳过 - {file_path}")
    
    if not files_to_process:
        print("错误: 没有找到要处理的文件")
        return
    
    print(f"找到 {len(files_to_process)} 个文件需要处理")
    print(f"多进程模式: {use_multiprocessing if use_multiprocessing is not None else '使用配置文件设置'}")
    print(f"代理模式: {use_proxy if use_proxy is not None else '使用配置文件设置'}")
    
    # 处理每个文件
    all_results = []
    for file in files_to_process:
        file_path = file if os.path.isabs(file) else os.path.join(args.input_dir, file)
        
        print(f"\n处理文件: {file_path}")
        result = process_single_file(
            file_path, 
            args.output_dir,
            sample_size=args.sample,
            use_threading=not args.no_threading,
            use_multiprocessing=use_multiprocessing,
            use_proxy=use_proxy
        )
        
        if result is not None:
            all_results.append(result)
    
    # 合并所有结果
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_output = os.path.join(args.output_dir, f"combined_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        combined_df.to_csv(combined_output, index=False, encoding='utf-8')
        print(f"\n所有文件合并结果已保存至: {combined_output}")
        
        # 生成总体统计
        total_stats = os.path.join(args.output_dir, f"combined_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(total_stats, 'w', encoding='utf-8') as f:
            f.write(f"处理完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理文件数: {len(files_to_process)}\n")
            f.write(f"总记录数: {len(combined_df)}\n")
            f.write(f"输出目录: {args.output_dir}\n")
        
        print(f"总体统计已保存至: {total_stats}")

if __name__ == "__main__":
    # Windows多进程支持需要这个保护
    mp.freeze_support()
    main()