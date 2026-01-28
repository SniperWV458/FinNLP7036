# data_cleaner.py
"""
数据清洗模块 - 负责清洗原始HTML数据，提取结构化信息
"""

import os
import re
import json
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

from config import FINANCIAL_TERMS, ASSETS_CONFIG

class DataCleaner:
    """数据清洗器类"""
    
    def __init__(self, input_base_dir="fetched_data", output_base_dir="cleaned_data"):
        """
        初始化数据清洗器
        
        参数:
            input_base_dir: 输入数据目录
            output_base_dir: 输出数据目录
        """
        self.input_base_dir = input_base_dir
        self.output_base_dir = output_base_dir
        
    def load_raw_data(self, asset_name, year, month):
        """
        加载指定资产和月份的原始数据
        
        参数:
            asset_name: 资产名称
            year: 年份
            month: 月份
        
        返回:
            raw_data_list: 原始数据列表
        """
        asset_safe_name = asset_name.replace('/', '_').replace(' ', '_')
        month_dir = os.path.join(self.input_base_dir, asset_safe_name, f"{year}_{month:02d}")
        
        if not os.path.exists(month_dir):
            print(f"目录不存在: {month_dir}")
            return []
        
        raw_data_list = []
        
        # 加载元数据
        metadata_file = os.path.join(month_dir, "raw_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 加载HTML内容
            for item in metadata:
                html_file = os.path.join(month_dir, f"news_{item['index']:03d}_raw.html")
                if os.path.exists(html_file):
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    raw_data = {
                        'asset_name': asset_name,
                        'year': year,
                        'month': month,
                        'url': item['url'],
                        'html_content': html_content,
                        'timestamp': item['timestamp']
                    }
                    raw_data_list.append(raw_data)
        
        print(f"从 {month_dir} 加载了 {len(raw_data_list)} 条原始数据")
        return raw_data_list
    
    def clean_html_content(self, html_content, url, asset_name):
        """
        清洗HTML内容，提取结构化信息
        
        参数:
            html_content: HTML内容
            url: 网页URL
            asset_name: 资产名称
        
        返回:
            cleaned_data: 清洗后的数据字典
        """
        if not html_content:
            return None
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 移除不需要的标签
            for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', 'iframe', 'noscript', 'form', 'button']):
                tag.decompose()
            
            # 提取标题
            title = ""
            title_selectors = [
                'h1', 'h1[class*="title"]', 'h1[class*="headline"]', 'title',
                'meta[property="og:title"]', 'meta[name="twitter:title"]', 'meta[name="title"]',
                '[data-testid="headline"]', '.headline', '.article-title', '.post-title',
                '.entry-title', '.story-title', '.news-title'
            ]
            
            for selector in title_selectors:
                if selector.startswith('meta'):
                    meta_tag = soup.select_one(selector)
                    if meta_tag and meta_tag.get('content'):
                        title = meta_tag['content'].strip()
                        break
                else:
                    tag = soup.select_one(selector)
                    if tag and tag.get_text(strip=True):
                        title = tag.get_text(strip=True)
                        break
            
            if len(title) > 200:
                title = title[:200] + "..."
            
            # 提取描述/摘要
            description = ""
            desc_selectors = [
                'meta[name="description"]', 'meta[property="og:description"]',
                'meta[name="twitter:description"]', 'meta[name="snippet"]',
                '.article-excerpt', '.post-excerpt', '.summary',
                '.deck', '.subhead', '.article-summary'
            ]
            
            for selector in desc_selectors:
                tag = soup.select_one(selector)
                if tag:
                    if selector.startswith('meta'):
                        desc = tag.get('content', '')
                    else:
                        desc = tag.get_text(strip=True)
                    
                    if desc and len(desc) > 20:
                        description = desc.strip()
                        break
            
            # 提取正文内容
            content_selectors = [
                'article', 'main', '[role="main"]', '.article-content',
                '.post-content', '.entry-content', '.story-body', '.content',
                '.article-body', '.post-body', '[class*="content"]',
                '[class*="article"]', '[class*="post"]', '[class*="story"]',
                'div[itemprop="articleBody"]', 'div.zn-body__paragraph',
                '.article-text', '.story-text', '.text-content',
                '.fin-content', '.market-news-content', '.financial-content',
                '.news-article-body'
            ]
            
            main_content = None
            max_length = 0
            
            for selector in content_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    text_length = len(elem.get_text(strip=True))
                    if text_length > max_length and text_length > 100:
                        main_content = elem
                        max_length = text_length
            
            # 如果没有找到特定内容区域，使用body
            if not main_content:
                main_content = soup.body or soup
            
            # 清理正文文本
            if main_content:
                # 移除不需要的标签
                for tag in main_content(['div', 'span', 'a']):
                    if tag.get_text(strip=True) == '':
                        tag.decompose()
                
                # 获取清理后的文本
                text = main_content.get_text(separator='\n', strip=True)
                
                # 清理多余的空白字符
                text = re.sub(r'\n\s*\n+', '\n\n', text)
                text = re.sub(r'[ \t]{2,}', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                
                # 过滤过短的段落
                lines = text.split('\n')
                filtered_lines = [line.strip() for line in lines if len(line.strip()) > 20]
                text = '\n'.join(filtered_lines)
            else:
                text = ""
            
            # 提取发布日期
            publish_date = ""
            date_selectors = [
                'time[datetime]', 'meta[property="article:published_time"]',
                'meta[name="publish_date"]', 'meta[name="date"]',
                'meta[property="og:published_time"]', '[itemprop="datePublished"]',
                '.date', '.published', '.timestamp', '.article-date',
                '.post-date', '.dateline', '.publish-date'
            ]
            
            for selector in date_selectors:
                tag = soup.select_one(selector)
                if tag:
                    if selector == 'time[datetime]' or selector.startswith('[itemprop'):
                        date_str = tag.get('datetime', tag.get('content', tag.get_text(strip=True)))
                    elif selector.startswith('meta'):
                        date_str = tag.get('content', '')
                    else:
                        date_str = tag.get_text(strip=True)
                    
                    if date_str:
                        publish_date = date_str
                        break
            
            # 提取作者
            author = ""
            author_selectors = [
                'meta[name="author"]', 'meta[property="article:author"]',
                '[itemprop="author"]', '.author', '.byline',
                '.article-author', '.post-author', '[rel="author"]',
                '.writer', '.reporter'
            ]
            
            for selector in author_selectors:
                tag = soup.select_one(selector)
                if tag:
                    if selector.startswith('meta') or selector.startswith('[itemprop'):
                        author_text = tag.get('content', '')
                    else:
                        author_text = tag.get_text(strip=True)
                    
                    if author_text:
                        author = author_text
                        break
            
            # 提取域名
            domain = urlparse(url).netloc
            
            # 提取金融相关关键词
            financial_keywords = []
            if text:
                text_lower = text.lower()
                for term in FINANCIAL_TERMS:
                    if term in text_lower:
                        financial_keywords.append(term)
                
                # 去重
                financial_keywords = list(set(financial_keywords))[:10]
            
            # 计算文本特征
            word_count = len(text.split())
            sentence_count = len(re.findall(r'[.!?]+', text))
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            return {
                'asset_name': asset_name,
                'title': title[:500] if title else "",
                'description': description[:1000] if description else "",
                'content': text[:15000] if text else "",
                'publish_date': publish_date[:100] if publish_date else "",
                'author': author[:200] if author else "",
                'domain': domain,
                'url': url,
                'cleaned_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'content_length': len(text) if text else 0,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': round(avg_sentence_length, 2),
                'financial_keywords': ', '.join(financial_keywords) if financial_keywords else "",
                'has_financial_content': len(financial_keywords) > 0
            }
            
        except Exception as e:
            print(f"清洗HTML时出错 {url}: {e}")
            return None
    
    def clean_asset_month_data(self, asset_name, year, month):
        """
        清洗指定资产和月份的数据
        
        参数:
            asset_name: 资产名称
            year: 年份
            month: 月份
        
        返回:
            cleaned_data_list: 清洗后的数据列表
        """
        print(f"\n清洗 {asset_name} - {year}年{month}月 数据...")
        
        # 加载原始数据
        raw_data_list = self.load_raw_data(asset_name, year, month)
        
        if not raw_data_list:
            print(f"没有找到原始数据")
            return []
        
        # 清洗数据
        cleaned_data_list = []
        success_count = 0
        
        for i, raw_data in enumerate(raw_data_list, 1):
            print(f"\n[{i}/{len(raw_data_list)}] 清洗: {raw_data['url'][:80]}...")
            
            cleaned_data = self.clean_html_content(
                raw_data['html_content'], raw_data['url'], raw_data['asset_name']
            )
            
            if cleaned_data and cleaned_data['content'] and len(cleaned_data['content']) > 2000:
                # 添加额外信息
                cleaned_data['year'] = year
                cleaned_data['month'] = month
                cleaned_data['category'] = ASSETS_CONFIG.get(asset_name, {}).get('category', 'Unknown')
                cleaned_data['original_timestamp'] = raw_data.get('timestamp', '')
                
                cleaned_data_list.append(cleaned_data)
                success_count += 1
                
                print(f"   成功! ({cleaned_data['content_length']} 字符)")
            else:
                print(f"   清洗失败或内容过短")
        
        print(f"\n清洗完成: {success_count}/{len(raw_data_list)} 条成功")
        
        return cleaned_data_list
    
    def save_cleaned_data(self, cleaned_data_list, asset_name, year, month):
        """
        保存清洗后的数据
        
        参数:
            cleaned_data_list: 清洗后的数据列表
            asset_name: 资产名称
            year: 年份
            month: 月份
        """
        if not cleaned_data_list:
            return
        
        # 创建目录
        asset_safe_name = asset_name.replace('/', '_').replace(' ', '_')
        asset_dir = os.path.join(self.output_base_dir, asset_safe_name)
        month_dir = os.path.join(asset_dir, f"{year}_{month:02d}")
        os.makedirs(month_dir, exist_ok=True)
        
        # 保存为CSV
        csv_filename = os.path.join(month_dir, f"{asset_safe_name}_{year}_{month:02d}_cleaned.csv")
        df = pd.DataFrame(cleaned_data_list)
        
        # 列顺序
        column_order = [
            'asset_name', 'category', 'year', 'month',
            'title', 'description', 'content', 'author', 'publish_date',
            'domain', 'url', 'cleaned_timestamp', 'original_timestamp',
            'content_length', 'word_count', 'sentence_count', 'avg_sentence_length',
            'financial_keywords', 'has_financial_content'
        ]
        
        # 确保所有列都存在
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"已保存CSV文件: {csv_filename}")
        
        # 保存为JSON
        json_filename = os.path.join(month_dir, f"{asset_safe_name}_{year}_{month:02d}_cleaned.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data_list, f, indent=2, ensure_ascii=False)
        print(f"已保存JSON文件: {json_filename}")
        
        # 保存文本文件
        for i, data in enumerate(cleaned_data_list, 1):
            txt_filename = os.path.join(month_dir, f"news_{i:03d}_cleaned.txt")
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(f"资产: {data.get('asset_name', '')}\n")
                f.write(f"类别: {data.get('category', '')}\n")
                f.write(f"年份: {year}\n")
                f.write(f"月份: {month}\n")
                f.write(f"标题: {data.get('title', '')}\n")
                f.write(f"来源: {data.get('domain', '')}\n")
                f.write(f"发布日期: {data.get('publish_date', '')}\n")
                f.write(f"作者: {data.get('author', '')}\n")
                f.write(f"描述: {data.get('description', '')}\n")
                f.write(f"URL: {data.get('url', '')}\n")
                f.write(f"清洗时间: {data.get('cleaned_timestamp', '')}\n")
                f.write(f"内容长度: {data.get('content_length', 0)} 字符\n")
                f.write(f"单词数: {data.get('word_count', 0)}\n")
                f.write(f"句子数: {data.get('sentence_count', 0)}\n")
                f.write(f"平均句长: {data.get('avg_sentence_length', 0)}\n")
                f.write(f"金融关键词: {data.get('financial_keywords', '')}\n")
                f.write(f"包含金融内容: {data.get('has_financial_content', False)}\n")
                f.write("="*60 + "\n")
                f.write(data.get('content', '')[:3000])
        
        # 保存统计信息
        stats = {
            "asset_name": asset_name,
            "category": ASSETS_CONFIG.get(asset_name, {}).get('category', 'Unknown'),
            "year": year,
            "month": month,
            "total_articles": len(cleaned_data_list),
            "avg_content_length": df['content_length'].mean() if not df.empty else 0,
            "avg_word_count": df['word_count'].mean() if not df.empty else 0,
            "unique_domains": df['domain'].nunique() if not df.empty else 0,
            "has_financial_content_ratio": (df['has_financial_content'].sum() / len(df) * 100) if not df.empty else 0,
            "cleaned_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        stats_filename = os.path.join(month_dir, f"cleaning_statistics.json")
        with open(stats_filename, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"清洗数据已保存到: {month_dir}")
        
        return csv_filename
    
    def clean_all_data(self, year=2024, months=list(range(1, 13))):
        """
        清洗所有资产的所有月份数据
        
        参数:
            year: 年份
            months: 月份列表
        """
        print("=" * 60)
        print(f"开始清洗 {len(ASSETS_CONFIG)} 种资产的新闻数据")
        print(f"时间范围: {year}年{months[0]}月 - {year}年{months[-1]}月")
        print("=" * 60)
        
        all_cleaned_data = []
        total_assets = len(ASSETS_CONFIG)
        
        # 按资产处理
        for asset_idx, (asset_name, asset_config) in enumerate(ASSETS_CONFIG.items(), 1):
            print(f"\n{'='*60}")
            print(f"清洗资产 {asset_idx}/{total_assets}: {asset_name}")
            print(f"{'='*60}")
            
            asset_cleaned_data = []
            
            # 按月份处理
            for month_idx, month in enumerate(months, 1):
                print(f"\n处理月份 {month_idx}/{len(months)}")
                
                try:
                    # 清洗当前资产和月份的数据
                    month_cleaned_data = self.clean_asset_month_data(asset_name, year, month)
                    
                    if month_cleaned_data:
                        asset_cleaned_data.extend(month_cleaned_data)
                        # 保存清洗后的数据
                        self.save_cleaned_data(month_cleaned_data, asset_name, year, month)
                        
                except Exception as e:
                    print(f"清洗{asset_name} - {year}年{month}月时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if asset_cleaned_data:
                all_cleaned_data.extend(asset_cleaned_data)
                
                # 保存资产汇总
                asset_safe_name = asset_name.replace('/', '_').replace(' ', '_')
                asset_dir = os.path.join(self.output_base_dir, asset_safe_name)
                
                asset_df = pd.DataFrame(asset_cleaned_data)
                asset_csv = os.path.join(asset_dir, f"{asset_safe_name}_{year}_cleaned_summary.csv")
                asset_df.to_csv(asset_csv, index=False, encoding='utf-8-sig')
                
                print(f"\n{asset_name} 清洗统计:")
                print(f"  总新闻数: {len(asset_df)}")
                print(f"  月份覆盖: {asset_df['month'].nunique()}/{len(months)}")
                print(f"  来源网站: {asset_df['domain'].nunique()}")
                print(f"  平均内容长度: {asset_df['content_length'].mean():.0f} 字符")
        
        # 保存所有数据汇总
        if all_cleaned_data:
            all_df = pd.DataFrame(all_cleaned_data)
            summary_csv = os.path.join(self.output_base_dir, f"all_assets_{year}_cleaned_summary.csv")
            all_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
            
            print(f"\n{'='*60}")
            print(f"清洗完成!")
            print(f"{'='*60}")
            print(f"总清洗新闻数: {len(all_df)}")
            print(f"总资产数: {len(ASSETS_CONFIG)}")
            print(f"来源网站总数: {all_df['domain'].nunique()}")
            print(f"平均内容长度: {all_df['content_length'].mean():.0f} 字符")
            
            # 保存总统计信息
            total_stats = {}
            for asset_name in ASSETS_CONFIG.keys():
                asset_df = all_df[all_df['asset_name'] == asset_name]
                if not asset_df.empty:
                    total_stats[asset_name] = {
                        "total_news": len(asset_df),
                        "months_covered": asset_df['month'].nunique(),
                        "unique_domains": asset_df['domain'].nunique(),
                        "avg_content_length": asset_df['content_length'].mean(),
                        "financial_content_ratio": (asset_df['has_financial_content'].sum() / len(asset_df) * 100)
                    }
            
            total_stats_file = os.path.join(self.output_base_dir, f"total_cleaning_statistics.json")
            with open(total_stats_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "overall_stats": {
                        "total_news": len(all_df),
                        "total_assets": len(ASSETS_CONFIG),
                        "total_months": all_df[['asset_name', 'month']].drop_duplicates().shape[0],
                        "unique_domains": all_df['domain'].nunique(),
                        "cleaning_complete_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    },
                    "asset_stats": total_stats
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\n总统计信息已保存到: {total_stats_file}")
            print(f"汇总CSV已保存到: {summary_csv}")
        
        return all_cleaned_data


# 独立运行测试
if __name__ == "__main__":
    # 测试清洗功能
    cleaner = DataCleaner(
        input_base_dir="test_fetched_data",
        output_base_dir="test_cleaned_data"
    )
    
    # 测试清洗S&P 500 2024年1月的数据
    test_asset = "S&P 500"
    
    print("测试数据清洗...")
    cleaned_data = cleaner.clean_asset_month_data(test_asset, 2024, 1)
    
    if cleaned_data:
        cleaner.save_cleaned_data(cleaned_data, test_asset, 2024, 1)
        print(f"测试完成，清洗了 {len(cleaned_data)} 条新闻")
    else:
        print("测试失败，未清洗到数据")