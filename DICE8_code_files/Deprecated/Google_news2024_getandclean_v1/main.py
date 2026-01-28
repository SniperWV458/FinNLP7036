# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 19:01:30 2026

@author: 25010
"""

# main.py
"""
主控制程序 - 协调数据获取和数据清洗流程
"""

import os
import json
import time
from datetime import datetime
import pandas as pd

from config import (
    ASSETS_CONFIG, YEAR, MONTHS, MAX_NEWS_PER_MONTH,
    OUTPUT_BASE_DIR, CHROME_OPTIONS
)
from data_fetcher import DataFetcher
from data_cleaner import DataCleaner

def main():
    """
    主函数 - 控制整个数据获取和清洗流程
    """
    print("=" * 60)
    print("多资产新闻数据抓取与清洗系统")
    print("=" * 60)
    
    # 显示资产列表
    print("将处理的资产列表:")
    for i, (asset_name, config) in enumerate(ASSETS_CONFIG.items(), 1):
        print(f"{i:2d}. {asset_name:<30} ({config['category']})")
    
    print(f"\n总资产数: {len(ASSETS_CONFIG)}")
    print(f"总月份数: {len(MONTHS)}")
    print(f"每月每资产目标新闻数: {MAX_NEWS_PER_MONTH}")
    print(f"预计总新闻数: {len(ASSETS_CONFIG) * len(MONTHS) * MAX_NEWS_PER_MONTH}")
    
    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"{OUTPUT_BASE_DIR}_{timestamp}"
    
    # 目录结构
    fetched_dir = os.path.join(base_dir, "fetched_data")
    cleaned_dir = os.path.join(base_dir, "cleaned_data")
    logs_dir = os.path.join(base_dir, "logs")
    
    os.makedirs(fetched_dir, exist_ok=True)
    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 保存运行配置
    config_log = {
        "assets_config": ASSETS_CONFIG,
        "year": YEAR,
        "months": MONTHS,
        "max_news_per_month": MAX_NEWS_PER_MONTH,
        "base_directory": base_dir,
        "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    config_file = os.path.join(logs_dir, "run_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_log, f, indent=2, ensure_ascii=False)
    
    print(f"\n输出目录: {base_dir}")
    print(f"开始时间: {config_log['start_time']}")
    
    # 询问用户要执行的操作
    print("\n请选择要执行的操作:")
    print("1. 只获取数据")
    print("2. 只清洗数据")
    print("3. 获取并清洗数据")
    print("4. 测试单个资产")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    if choice == "1":
        # 只获取数据
        run_fetch_only(fetched_dir)
    elif choice == "2":
        # 只清洗数据
        run_clean_only(fetched_dir, cleaned_dir)
    elif choice == "3":
        # 获取并清洗数据
        run_fetch_and_clean(fetched_dir, cleaned_dir)
    elif choice == "4":
        # 测试单个资产
        run_test_mode(fetched_dir, cleaned_dir)
    else:
        print("无效选择，程序退出")
        return
    
    # 保存完成时间
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    completion_log = {
        "end_time": end_time,
        "total_duration": "请查看日志文件"
    }
    
    completion_file = os.path.join(logs_dir, "completion.json")
    with open(completion_file, 'w', encoding='utf-8') as f:
        json.dump(completion_log, f, indent=2, ensure_ascii=False)
    
    print(f"\n程序执行完成!")
    print(f"开始时间: {config_log['start_time']}")
    print(f"结束时间: {end_time}")
    print(f"输出目录: {base_dir}")

def run_fetch_only(fetched_dir):
    """只执行数据获取"""
    print("\n" + "="*60)
    print("开始数据获取...")
    print("="*60)
    
    # 初始化数据获取器
    fetcher = DataFetcher(output_base_dir=fetched_dir)
    
    try:
        # 获取所有数据
        all_raw_data = fetcher.fetch_all_data(
            year=YEAR,
            months=MONTHS,
            max_news_per_month=MAX_NEWS_PER_MONTH
        )
        
        print(f"\n数据获取完成!")
        print(f"总共获取了 {len(all_raw_data) if all_raw_data else 0} 条新闻的原始数据")
        
    except Exception as e:
        print(f"数据获取过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        fetcher.close()

def run_clean_only(fetched_dir, cleaned_dir):
    """只执行数据清洗"""
    print("\n" + "="*60)
    print("开始数据清洗...")
    print("="*60)
    
    # 检查是否有数据可清洗
    if not os.path.exists(fetched_dir):
        print(f"错误: 数据目录不存在: {fetched_dir}")
        print("请先运行数据获取或指定正确的目录")
        return
    
    # 初始化数据清洗器
    cleaner = DataCleaner(
        input_base_dir=fetched_dir,
        output_base_dir=cleaned_dir
    )
    
    try:
        # 清洗所有数据
        all_cleaned_data = cleaner.clean_all_data(
            year=YEAR,
            months=MONTHS
        )
        
        print(f"\n数据清洗完成!")
        print(f"总共清洗了 {len(all_cleaned_data) if all_cleaned_data else 0} 条新闻")
        
    except Exception as e:
        print(f"数据清洗过程中出错: {e}")
        import traceback
        traceback.print_exc()

def run_fetch_and_clean(fetched_dir, cleaned_dir):
    """执行完整的数据获取和清洗流程"""
    print("\n" + "="*60)
    print("开始完整的数据获取和清洗流程...")
    print("="*60)
    
    # 阶段1: 数据获取
    print("\n阶段1: 数据获取")
    print("-"*40)
    
    fetcher = DataFetcher(output_base_dir=fetched_dir)
    
    try:
        all_raw_data = fetcher.fetch_all_data(
            year=YEAR,
            months=MONTHS,
            max_news_per_month=MAX_NEWS_PER_MONTH
        )
        
        print(f"\n数据获取完成!")
        print(f"总共获取了 {len(all_raw_data) if all_raw_data else 0} 条新闻的原始数据")
        
    except Exception as e:
        print(f"数据获取过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    finally:
        fetcher.close()
    
    # 短暂暂停
    time.sleep(2)
    
    # 阶段2: 数据清洗
    print("\n" + "="*60)
    print("阶段2: 数据清洗")
    print("="*60)
    
    cleaner = DataCleaner(
        input_base_dir=fetched_dir,
        output_base_dir=cleaned_dir
    )
    
    try:
        all_cleaned_data = cleaner.clean_all_data(
            year=YEAR,
            months=MONTHS
        )
        
        print(f"\n数据清洗完成!")
        print(f"总共清洗了 {len(all_cleaned_data) if all_cleaned_data else 0} 条新闻")
        
    except Exception as e:
        print(f"数据清洗过程中出错: {e}")
        import traceback
        traceback.print_exc()

def run_test_mode(fetched_dir, cleaned_dir):
    """测试模式 - 处理单个资产"""
    print("\n" + "="*60)
    print("测试模式: 处理单个资产全年数据")
    print("="*60)
    
    # 显示资产列表
    print("可测试的资产:")
    assets_list = list(ASSETS_CONFIG.keys())
    for i, asset in enumerate(assets_list, 1):
        print(f"{i}. {asset}")
    
    try:
        choice = int(input("\n请选择要测试的资产编号: ").strip())
        if 1 <= choice <= len(assets_list):
            asset_name = assets_list[choice-1]
        else:
            print("无效选择，使用默认资产: S&P 500")
            asset_name = "S&P 500"
    except:
        print("无效输入，使用默认资产: S&P 500")
        asset_name = "S&P 500"
    
    asset_config = ASSETS_CONFIG[asset_name]
    
    # 全年配置
    test_months = MONTHS
    test_max_news = MAX_NEWS_PER_MONTH
    
    print(f"\n测试配置:")
    print(f"  资产: {asset_name}")
    print(f"  年份: {YEAR}")
    print(f"  月份: {test_months}")
    print(f"  每月最大新闻数: {test_max_news}")
    
    # 阶段1: 获取全年测试数据
    print("\n" + "-"*40)
    print("获取全年测试数据...")
    
    fetcher = DataFetcher(output_base_dir=fetched_dir)
    total_raw = 0
    all_link_tasks = []
    
    try:
        # 先收集全年链接（存储links.json），避免逐月立即抓取页面
        fetcher.driver = fetcher.setup_driver(headless=CHROME_OPTIONS["headless"])
        for month in test_months:
            print(f"\n获取 {YEAR}年{month}月 链接...")
            month_links = fetcher.collect_asset_month_links(
                asset_name, asset_config, YEAR, month, max_news=test_max_news
            )
            if month_links:
                all_link_tasks.extend(month_links)
                print(f"累计链接: {len(all_link_tasks)}")
            else:
                print(f"{YEAR}年{month}月 未找到链接")
                time.sleep(1)

    finally:
        fetcher.close()
    
    if not all_link_tasks:
        print("全年未获取到任何链接，测试结束")
        return
    
    print(f"\n全年共收集 {len(all_link_tasks)} 条链接，开始多进程抓取页面...")
    raw_data_results = fetcher.scrape_links_in_parallel(all_link_tasks)
    total_raw = len(raw_data_results)

    # 按月分组保存原始数据
    grouped_raw = {}
    for item in raw_data_results:
        key = (item["asset_name"], item["year"], item["month"])
        grouped_raw.setdefault(key, []).append(item)
    for (asset, year_val, month_val), group in grouped_raw.items():
        fetcher.save_raw_data(group, asset, year_val, month_val)

    if total_raw == 0:
        print("多进程抓取未获取到页面，测试结束")
        return

    # 阶段2: 清洗全年测试数据
    print("\n" + "-"*40)
    print("清洗全年测试数据...")
    
    cleaner = DataCleaner(
        input_base_dir=fetched_dir,
        output_base_dir=cleaned_dir
    )
    
    total_cleaned = 0
    try:
        for month in test_months:
            print(f"\n清洗 {YEAR}年{month}月 数据...")
            cleaned_data = cleaner.clean_asset_month_data(asset_name, YEAR, month)
            
            if cleaned_data:
                cleaner.save_cleaned_data(cleaned_data, asset_name, YEAR, month)
                total_cleaned += len(cleaned_data)
                print(f"清洗了 {len(cleaned_data)} 条数据 (累计 {total_cleaned})")
            else:
                print(f"{YEAR}年{month}月 未清洗到数据")
    
    except Exception as e:
        print(f"清洗过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n测试模式完成")
    print(f"全年共获取 {total_raw} 条，清洗 {total_cleaned} 条")

def create_requirements_file():
    """创建requirements.txt文件"""
    requirements = """
selenium==4.15.0
beautifulsoup4==4.12.2
pandas==2.1.4
requests==2.31.0
lxml==4.9.3
"""
    
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements.strip())
    
    print("已创建 requirements.txt 文件")

if __name__ == "__main__":
    # 创建requirements.txt
    create_requirements_file()
    
    # 显示运行说明
    print("=" * 60)
    print("运行说明:")
    print("1. 请确保已安装Chrome浏览器")
    print("2. 请下载对应版本的ChromeDriver，并添加到系统PATH")
    print("3. 已生成requirements.txt，请运行: pip install -r requirements.txt")
    print("=" * 60)
    
    # 运行主程序
    main()