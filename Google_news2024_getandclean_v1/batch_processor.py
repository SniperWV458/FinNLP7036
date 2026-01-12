# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 19:02:24 2026

@author: 25010
"""

# batch_processor.py
"""
批量处理脚本 - 支持分批处理、断点续传等功能
"""

import os
import json
import time
import argparse
from datetime import datetime
from data_fetcher import DataFetcher
from data_cleaner import DataCleaner
from config import ASSETS_CONFIG, YEAR, MONTHS, MAX_NEWS_PER_MONTH

class BatchProcessor:
    """批量处理器类"""
    
    def __init__(self, base_dir="batch_processed_data"):
        """
        初始化批量处理器
        
        参数:
            base_dir: 基础目录
        """
        self.base_dir = base_dir
        self.state_file = os.path.join(base_dir, "processing_state.json")
        self.log_file = os.path.join(base_dir, "processing_log.txt")
        
        os.makedirs(base_dir, exist_ok=True)
    
    def log_message(self, message):
        """记录日志消息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
    
    def save_state(self, state):
        """保存处理状态"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_state(self):
        """加载处理状态"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def process_batch(self, assets=None, months=None, batch_size=3, resume=False):
        """
        批量处理资产和月份
        
        参数:
            assets: 要处理的资产列表，None表示所有资产
            months: 要处理的月份列表，None表示所有月份
            batch_size: 每批处理的资产数
            resume: 是否从上次中断处恢复
        """
        if assets is None:
            assets = list(ASSETS_CONFIG.keys())
        
        if months is None:
            months = MONTHS
        
        # 加载或初始化状态
        if resume:
            state = self.load_state()
            if state:
                self.log_message(f"从上次状态恢复: {state}")
                processed_assets = state.get('processed_assets', [])
                current_batch = state.get('current_batch', 0)
            else:
                processed_assets = []
                current_batch = 0
                state = {'processed_assets': [], 'current_batch': 0}
        else:
            processed_assets = []
            current_batch = 0
            state = {'processed_assets': [], 'current_batch': 0}
        
        # 过滤已处理的资产
        remaining_assets = [a for a in assets if a not in processed_assets]
        
        if not remaining_assets:
            self.log_message("所有资产都已处理完成")
            return
        
        self.log_message(f"开始批量处理，总共 {len(remaining_assets)} 个资产，每批 {batch_size} 个")
        
        # 分批处理
        total_batches = (len(remaining_assets) + batch_size - 1) // batch_size
        
        for batch_idx in range(current_batch, total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(remaining_assets))
            batch_assets = remaining_assets[batch_start:batch_end]
            
            self.log_message(f"\n处理批次 {batch_idx+1}/{total_batches}: {', '.join(batch_assets)}")
            
            # 更新状态
            state['current_batch'] = batch_idx
            self.save_state(state)
            
            # 处理当前批次
            for asset_idx, asset_name in enumerate(batch_assets, 1):
                self.log_message(f"\n处理资产 ({asset_idx}/{len(batch_assets)}): {asset_name}")
                
                # 创建资产特定的输出目录
                asset_fetched_dir = os.path.join(self.base_dir, "fetched_data", asset_name.replace('/', '_').replace(' ', '_'))
                asset_cleaned_dir = os.path.join(self.base_dir, "cleaned_data", asset_name.replace('/', '_').replace(' ', '_'))
                
                os.makedirs(asset_fetched_dir, exist_ok=True)
                os.makedirs(asset_cleaned_dir, exist_ok=True)
                
                # 处理该资产的所有月份
                asset_config = ASSETS_CONFIG[asset_name]
                
                for month in months:
                    self.log_message(f"  处理 {YEAR}年{month}月")
                    
                    try:
                        # 获取数据
                        fetcher = DataFetcher(output_base_dir=asset_fetched_dir)
                        raw_data = fetcher.fetch_asset_month_data(
                            asset_name, asset_config, YEAR, month, MAX_NEWS_PER_MONTH
                        )
                        
                        if raw_data:
                            fetcher.save_raw_data(raw_data, asset_name, YEAR, month)
                        
                        fetcher.close()
                        
                        # 清洗数据
                        cleaner = DataCleaner(
                            input_base_dir=asset_fetched_dir,
                            output_base_dir=asset_cleaned_dir
                        )
                        
                        cleaned_data = cleaner.clean_asset_month_data(asset_name, YEAR, month)
                        
                        if cleaned_data:
                            cleaner.save_cleaned_data(cleaned_data, asset_name, YEAR, month)
                        
                        # 资产内月份间延迟
                        if month < months[-1]:
                            time.sleep(2)
                            
                    except Exception as e:
                        self.log_message(f"  处理失败: {e}")
                        continue
                
                # 标记资产为已处理
                processed_assets.append(asset_name)
                state['processed_assets'] = processed_assets
                self.save_state(state)
                
                # 资产间延迟
                if asset_idx < len(batch_assets):
                    time.sleep(10)
            
            # 批次间延迟
            if batch_idx < total_batches - 1:
                self.log_message(f"\n批次 {batch_idx+1} 完成，等待30秒后开始下一批...")
                time.sleep(30)
        
        # 处理完成
        self.log_message(f"\n所有批次处理完成!")
        self.log_message(f"总共处理了 {len(processed_assets)} 个资产")
        
        # 生成汇总报告
        self.generate_summary_report(processed_assets)
    
    def generate_summary_report(self, processed_assets):
        """生成汇总报告"""
        report = {
            "processing_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_assets_processed": len(processed_assets),
            "processed_assets": processed_assets,
            "year": YEAR,
            "months": MONTHS,
            "max_news_per_month": MAX_NEWS_PER_MONTH
        }
        
        report_file = os.path.join(self.base_dir, "processing_summary.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.log_message(f"汇总报告已保存到: {report_file}")
        
        # 生成简明的文本报告
        txt_report = f"""批量处理完成报告
====================

处理时间: {report['processing_date']}
处理年份: {YEAR}
处理月份: {len(MONTHS)}个月 ({MONTHS[0]}月-{MONTHS[-1]}月)
处理资产数: {len(processed_assets)}
每月每资产目标新闻数: {MAX_NEWS_PER_MONTH}

已处理的资产:
{chr(10).join(f'  - {asset}' for asset in processed_assets)}

输出目录: {os.path.abspath(self.base_dir)}
"""
        
        txt_report_file = os.path.join(self.base_dir, "processing_summary.txt")
        with open(txt_report_file, 'w', encoding='utf-8') as f:
            f.write(txt_report)
        
        print("\n" + "="*60)
        print("批量处理完成!")
        print(f"输出目录: {os.path.abspath(self.base_dir)}")
        print(f"汇总报告: {txt_report_file}")
        print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量处理金融新闻数据')
    parser.add_argument('--assets', nargs='+', help='要处理的资产列表，用空格分隔')
    parser.add_argument('--months', nargs='+', type=int, help='要处理的月份列表，用空格分隔')
    parser.add_argument('--batch-size', type=int, default=3, help='每批处理的资产数')
    parser.add_argument('--resume', action='store_true', help='从上次中断处恢复')
    parser.add_argument('--output-dir', default='batch_processed_data', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = BatchProcessor(base_dir=args.output_dir)
    
    # 记录开始时间
    start_time = datetime.now()
    processor.log_message(f"批量处理开始于: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 处理参数
    assets = args.assets
    months = args.months
    
    if assets:
        # 验证资产名称
        valid_assets = []
        invalid_assets = []
        for asset in assets:
            if asset in ASSETS_CONFIG:
                valid_assets.append(asset)
            else:
                invalid_assets.append(asset)
        
        if invalid_assets:
            processor.log_message(f"警告: 以下资产名称无效: {invalid_assets}")
        
        assets = valid_assets
    
    if months:
        # 验证月份
        valid_months = [m for m in months if 1 <= m <= 12]
        invalid_months = [m for m in months if m not in valid_months]
        
        if invalid_months:
            processor.log_message(f"警告: 以下月份无效: {invalid_months}")
        
        months = valid_months
    
    # 开始处理
    try:
        processor.process_batch(
            assets=assets,
            months=months,
            batch_size=args.batch_size,
            resume=args.resume
        )
    except KeyboardInterrupt:
        processor.log_message("用户中断了处理")
    except Exception as e:
        processor.log_message(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 记录结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    processor.log_message(f"批量处理结束于: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    processor.log_message(f"总处理时长: {duration}")

if __name__ == "__main__":
    main()