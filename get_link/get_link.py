#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 23:28:40 2026

@author: gjn
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融资产数据按月下载 (2017-2024)
下载表格中的14个资产，每月最多20条
"""

from gdeltdoc import GdeltDoc, Filters
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import random
import json

class FinancialAssetsDownloader:
    """金融资产数据下载器"""
    
    def __init__(self):
        self.gd = GdeltDoc()
        
        # 资产配置 - 来自表格
        self.assets_config = {
            "GSPC": {
                "name": "S&P 500 Index",
                "keywords": ["SP500", "SPX", "Standard Poors 500", "S and P 500"],
                "ticker": "$SPX",  # StockTwits格式的代码
                "type": "E"  # Equity/Index
            },
            "IXIC": {
                "name": "NASDAQ Composite",
                "keywords": ["NASDAQ", "NASDAQ Composite", "IXIC"],
                "ticker": "$QQQ",  # 使用QQQ作为代理
                "type": "E"
            },
            "DJI": {
                "name": "Dow Jones Industrial Average",
                "keywords": ["Dow Jones", "DJIA", "Dow 30"],
                "ticker": "$DIA",
                "type": "E"
            },
            "GOLD": {
                "name": "Gold",
                "keywords": ["gold price", "gold market", "gold bullion"],
                "type": "C"  # Commodity
            },
            "SILVER": {
                "name": "Silver",
                "keywords": ["silver price", "silver market", "silver bullion"],
                "type": "C"
            },
            "OIL": {
                "name": "WTI Crude Oil Futures",
                "keywords": ["crude oil", "oil price", "WTI", "oil futures"],
                "type": "C"
            }
        }
        
        # 下载参数
        self.start_year = 2017
        self.end_year = 2024
        self.max_records_per_month = 20  # 每月最多20条
        
        # 输出目录
        self.base_output_dir = "/Users/gjn/Desktop/financial_assets_data"
        
        # 延迟设置
        self.min_delay = 2.0
        self.max_delay = 4.0
        
    def get_month_date_range(self, year, month):
        """获取一个月的起止日期"""
        start_date = f"{year}-{month:02d}-01"
        
        if month == 12:
            end_date = f"{year}-12-31"
        else:
            end_date = f"{year}-{month:02d}-{(datetime(year, month+1, 1) - timedelta(days=1)).day:02d}"
        
        return start_date, end_date
    
    def download_asset_data(self, asset_id, asset_config):
        """下载单个资产的数据"""
        asset_name = asset_config["name"]
        keywords = asset_config["keywords"]
        
        print(f"\n{'='*60}")
        print(f"开始下载: {asset_id} - {asset_name}")
        print(f"{'='*60}")
        
        # 创建资产输出目录
        asset_dir = os.path.join(self.base_output_dir, asset_id)
        os.makedirs(asset_dir, exist_ok=True)
        
        # 存储所有月份的数据
        all_data = []
        
        for year in range(self.start_year, self.end_year + 1):
            print(f"\n{year}年:")
            
            for month in range(1, 13):
                print(f"  {year}-{month:02d}...", end=" ")
                
                try:
                    # 获取日期范围
                    start_date, end_date = self.get_month_date_range(year, month)
                    
                    # 创建过滤器
                    f = Filters(
                        start_date=start_date,
                        end_date=end_date,
                        num_records=self.max_records_per_month,
                        keyword=keywords,
                        language="English"
                    )
                    
                    # 执行搜索
                    articles_df = self.gd.article_search(f)
                    
                    if not articles_df.empty:
                        # 添加资产信息和时间信息
                        articles_df['asset_id'] = asset_id
                        articles_df['asset_name'] = asset_name
                        articles_df['asset_type'] = asset_config["type"]
                        articles_df['year'] = year
                        articles_df['month'] = month
                        articles_df['download_date'] = datetime.now().strftime("%Y-%m-%d")
                        
                        # 添加到总数据
                        all_data.append(articles_df)
                        
                        # 保存月度文件
                        month_csv = os.path.join(asset_dir, f"{asset_id}_{year}_{month:02d}.csv")
                        articles_df.to_csv(month_csv, index=False, encoding="utf-8-sig")
                        
                        print(f"✓ {len(articles_df)}条")
                        
                    else:
                        print("⚠ 0条")
                        
                except Exception as e:
                    print(f"✗ 错误: {str(e)[:50]}")
                
                # 随机延迟
                time.sleep(random.uniform(self.min_delay, self.max_delay))
        
        # 保存该资产的合并数据
        if all_data:
            # 合并所有月份的数据
            asset_df = pd.concat(all_data, ignore_index=True)
            
            # 按时间排序
            if 'date' in asset_df.columns:
                asset_df = asset_df.sort_values('date')
            elif 'datetime' in asset_df.columns:
                asset_df = asset_df.sort_values('datetime')
            
            # 保存合并的CSV
            asset_csv = os.path.join(asset_dir, f"{asset_id}_2017-2024_all.csv")
            asset_df.to_csv(asset_csv, index=False, encoding="utf-8-sig")
            
            # 统计信息
            total_records = len(asset_df)
            print(f"\n{'='*40}")
            print(f"{asset_name} 下载完成:")
            print(f"  总记录数: {total_records}")
            print(f"  时间范围: {self.start_year}-2024")
            print(f"  文件位置: {asset_csv}")
            print(f"{'='*40}")
            
            return {
                "asset_id": asset_id,
                "asset_name": asset_name,
                "type": asset_config["type"],
                "total_records": total_records,
                "file_path": asset_csv
            }
        else:
            print(f"\n{asset_name} 未获取到任何数据")
            return {
                "asset_id": asset_id,
                "asset_name": asset_name,
                "type": asset_config["type"],
                "total_records": 0,
                "file_path": None
            }
    
    def create_summary_report(self, download_stats):
        """创建下载摘要报告"""
        summary_data = []
        
        for stats in download_stats:
            summary_data.append({
                "Asset_ID": stats["asset_id"],
                "Asset_Name": stats["asset_name"],
                "Type": stats["type"],
                "Total_Records": stats["total_records"],
                "Status": "Success" if stats["total_records"] > 0 else "No Data",
                "File_Path": stats["file_path"] or "N/A"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存摘要
        summary_csv = os.path.join(self.base_output_dir, "download_summary.csv")
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        
        # 显示摘要
        print(f"\n{'='*80}")
        print("📊 下载摘要报告")
        print(f"{'='*80}")
        print(f"{'资产ID':<12} | {'资产名称':<30} | {'类型':<6} | {'记录数':<10} | {'状态':<10}")
        print(f"{'-'*80}")
        
        total_records = 0
        for _, row in summary_df.iterrows():
            print(f"{row['Asset_ID']:<12} | {row['Asset_Name'][:28]:<30} | "
                  f"{row['Type']:<6} | {row['Total_Records']:<10} | {row['Status']:<10}")
            total_records += row['Total_Records']
        
        print(f"{'-'*80}")
        print(f"{'总计':<12} | {'':<30} | {'':<6} | {total_records:<10} |")
        print(f"{'='*80}")
        
        # 保存配置信息
        config_info = {
            "download_parameters": {
                "start_year": self.start_year,
                "end_year": self.end_year,
                "max_records_per_month": self.max_records_per_month,
                "total_assets": len(self.assets_config)
            },
            "assets": self.assets_config,
            "summary": summary_data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        config_json = os.path.join(self.base_output_dir, "download_config.json")
        with open(config_json, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
        
        return summary_df
    
    def run(self):
        """运行下载程序"""
        
        # 创建输出目录
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        print(f"{'='*80}")
        print("📈 金融资产数据下载系统")
        print(f"{'='*80}")
        print(f"资产数量: {len(self.assets_config)} 个")
        print(f"时间范围: {self.start_year}年1月 - {self.end_year}年12月")
        print(f"每月最多: {self.max_records_per_month} 条记录")
        print(f"输出目录: {self.base_output_dir}")
        print(f"{'='*80}")
        
        # 下载统计数据
        download_stats = []
        
        # 下载每个资产
        for i, (asset_id, asset_config) in enumerate(self.assets_config.items(), 1):
            print(f"\n\n[进度 {i}/{len(self.assets_config)}]")
            
            stats = self.download_asset_data(asset_id, asset_config)
            download_stats.append(stats)
            
            # 资产间延迟
            if i < len(self.assets_config):
                delay_time = random.uniform(8, 15)
                print(f"\n⏳ 等待 {delay_time:.1f} 秒后下载下一个资产...")
                time.sleep(delay_time)
        
        # 创建摘要报告
        self.create_summary_report(download_stats)
        
        print(f"\n{'='*80}")
        print("🎉 下载任务完成!")
        print(f"{'='*80}")
        print(f"所有文件已保存到: {self.base_output_dir}")
        print(f"每个资产包含: 月度CSV文件 + 合并的CSV文件")
        print(f"{'='*80}")
        
        # 自动打开文件夹
        try:
            import subprocess
            subprocess.run(["open", self.base_output_dir])
            print("✅ 已自动打开输出目录")
        except:
            print("💡 提示: 请手动打开文件夹查看文件")



# 主程序
if __name__ == "__main__":
    # 使用完整版本
    downloader = FinancialAssetsDownloader()
    downloader.run()
    
    # 或使用简化版本
    # simple_download()
