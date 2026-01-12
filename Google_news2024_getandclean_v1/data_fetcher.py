# data_fetcher.py
"""
数据获取模块 - 优化版，解决只能获取10条新闻的问题
"""

import time
import random
import re
import calendar
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import requests
from urllib.parse import urlparse, quote
import warnings
warnings.filterwarnings('ignore')

from config import (
    ASSETS_CONFIG, YEAR, MONTHS, MAX_NEWS_PER_MONTH,
    CHROME_OPTIONS, REQUEST_TIMEOUT, SELENIUM_TIMEOUT,
    DELAY_BETWEEN_REQUESTS, DELAY_BETWEEN_MONTHS, DELAY_BETWEEN_ASSETS
)

class DataFetcher:
    """优化版数据获取器类"""
    
    def __init__(self, output_base_dir="fetched_data"):
        """
        初始化数据获取器
        """
        self.output_base_dir = output_base_dir
        self.driver = None
        
    def setup_driver(self, headless=True):
        """
        设置Chrome浏览器驱动
        
        返回:
            driver: Chrome WebDriver实例
        """
        chrome_options = Options()
        
        if headless:
            chrome_options.add_argument("--headless")
        
        # 防止被检测为自动化程序
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # 添加用户代理
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
        ]
        chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
        
        chrome_options.add_argument("--window-size=1920,1080")
        
        # 禁用图片和CSS加载
        if CHROME_OPTIONS["disable_images"] and CHROME_OPTIONS["disable_css"]:
            prefs = {
                "profile.managed_default_content_settings.images": 2,
                "profile.managed_default_content_settings.stylesheets": 2,
                "profile.default_content_setting_values.notifications": 2
            }
            chrome_options.add_experimental_option("prefs", prefs)
        
        chrome_options.add_argument("--disable-gpu")
        
        # 添加额外的参数
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        
        # 初始化驱动
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver
    
    def get_month_date_range(self, year, month):
      
        
    
    # 当前月份的第一天
        start_date = f"{month}/1/{year}"
    
    # 计算当前月份的最后一天
        if month == 12:
        # 12月的最后一天是12月31日
            end_date = f"12/31/{year}"
        else:
        # 获取当前月份的天数
            month_days = calendar.monthrange(year, month)[1]
            end_date = f"{month}/{month_days}/{year}"
    
        return start_date, end_date
    
    def construct_google_news_url(self, search_term, start_date, end_date, num_results=100):
        """
        构建Google News搜索URL - 优化版
        """
        search_term_encoded = quote(search_term)
        base_url = "https://www.google.com/search"
        
        # 优化参数，增加获取更多结果的可能性
        params = {
            'q': search_term_encoded,
            'tbm': 'nws',
            'tbs': f'cdr:1,cd_min:{start_date},cd_max:{end_date},sbd:1',  # 添加sbd:1按日期排序
            'lr': 'lang_en',
            'num': str(num_results),
            'hl': 'en',  # 语言
            'gl': 'us',  # 地区
            'cr': 'countryUS',  # 国家
            'pws': '0',  # 关闭个性化搜索
        }
        
        query_string = '&'.join([f'{k}={v}' for k, v in params.items()])
        full_url = f"{base_url}?{query_string}"
        
        return full_url
    
    def find_and_click_more_results(self, driver):
        """
        查找并点击"更多结果"按钮
        """
        # 尝试多种可能的"更多结果"按钮选择器
        more_button_selectors = [
            ("input[type='submit'][value*='More']", "css"),
            ("input[value*='更多结果']", "css"),
            ("a[aria-label*='More']", "css"),
            ("//a[contains(text(), 'More results')]", "xpath"),
            ("//a[contains(text(), '更多结果')]", "xpath"),
            ("//a[contains(@class, 'GNJvt ipz2Oe')]", "xpath"),
            ("//div[@role='button' and contains(text(), 'More')]", "xpath"),
            ("//div[@role='button' and contains(text(), '更多')]", "xpath"),
            ("//g-more-link//a", "xpath"),
            ("//a[@id='pnnext']", "xpath"),
            ("//a[@aria-label='Next page']", "xpath"),
            ("//a[span[text()='More']]", "xpath"),
        ]
        
        for selector, selector_type in more_button_selectors:
            try:
                if selector_type == "css":
                    more_button = driver.find_element(By.CSS_SELECTOR, selector)
                else:  # xpath
                    more_button = driver.find_element(By.XPATH, selector)
                
                if more_button.is_displayed():
                    print(f"找到'更多结果'按钮: {selector}")
                    # 滚动到按钮位置
                    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", more_button)
                    time.sleep(1)
                    
                    # 点击按钮
                    driver.execute_script("arguments[0].click();", more_button)
                    print("已点击'更多结果'按钮")
                    time.sleep(random.uniform(3, 5))
                    return True
            except:
                continue
        
        return False
    
    def extract_links_from_current_page(self, driver):
        """
        从当前页面提取新闻链接
        """
        links = []
        
        # 更全面的选择器列表
        selectors = [
            "a[jsname='tljFtd']",
            "a.WlydOe",
            "a[role='link'][href*='http']",
            "a[data-ved][href*='http']",
            "div.SoaBEf a",
            "div.yG4QQe a",
            "div.tNxQIb a",
            "div[class*='Soa'] a",
            "div[class*='yG'] a",
            "div.g:not([class*=' ']) a",
            "div[data-hveid] a",
            "div[class*='card'] a",
            "article a",
            "h3 a",
            "div[class*='news'] a",
            "div[class*='article'] a",
            "a.qtX7Yc",  # 另一种可能的新闻链接类
            "a.lLrAF",   # 另一种可能的新闻链接类
        ]
        
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elements:
                    try:
                        href = elem.get_attribute("href")
                        if href and href.startswith('http') and "google.com" not in href:
                            # 过滤不需要的链接
                            if any(x in href.lower() for x in [
                                'youtube.com', 'video', '/video/', 'watch?v=', 
                                '.pdf', '.jpg', '.png', '.gif', '.jpeg', '.mp4',
                                'accounts.google.com', 'support.google.com',
                                'policies.google.com', 'myaccount.google.com'
                            ]):
                                continue
                            
                            # 清理URL
                            href = href.split('&')[0].split('?')[0]
                            
                            if href not in links and len(href) < 500:
                                links.append(href)
                    except:
                        continue
            except:
                continue
        
        return links
    
    def fetch_news_links(self, driver, url, max_links=20):
        """
        优化的Google News链接获取函数
        """
        print("正在访问Google News...")
        driver.get(url)
        time.sleep(random.uniform(4, 6))
        
        # 检查搜索结果
        page_source = driver.page_source.lower()
        if "did not match any news results" in page_source or "no results found" in page_source:
            print("警告：没有找到搜索结果")
            return []
        
        news_links = []
        attempt = 0
        max_attempts = 5
        last_link_count = 0
        no_new_links_count = 0
        
        while len(news_links) < max_links and attempt < max_attempts:
            attempt += 1
            print(f"\n第 {attempt} 次尝试获取链接...")
            
            # 1. 首先尝试点击"更多结果"按钮
            if attempt > 1:  # 第一次访问时先不点击
                if self.find_and_click_more_results(driver):
                    # 成功点击后等待页面加载
                    time.sleep(random.uniform(3, 5))
            
            # 2. 智能滚动页面
            scroll_height = driver.execute_script("return document.body.scrollHeight")
            scroll_increment = 800
            current_scroll = 0
            
            while current_scroll < scroll_height and len(news_links) < max_links:
                # 执行滚动
                driver.execute_script(f"window.scrollTo(0, {current_scroll});")
                time.sleep(random.uniform(0.5, 1.5))
                
                # 每次滚动后提取链接
                current_links = self.extract_links_from_current_page(driver)
                new_links = [link for link in current_links if link not in news_links]
                
                if new_links:
                    for link in new_links:
                        if len(news_links) < max_links:
                            news_links.append(link)
                            print(f"找到链接 {len(news_links)}: {link[:80]}...")
                
                # 更新滚动位置
                current_scroll += scroll_increment
            
            # 3. 滚动到底部
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(2, 3))
            
            # 4. 再次提取链接
            current_links = self.extract_links_from_current_page(driver)
            new_links = [link for link in current_links if link not in news_links]
            
            if new_links:
                for link in new_links:
                    if len(news_links) < max_links:
                        news_links.append(link)
                        print(f"找到链接 {len(news_links)}: {link[:80]}...")
            
            # 5. 检查是否有新链接
            if len(news_links) == last_link_count:
                no_new_links_count += 1
                if no_new_links_count >= 2:
                    print("连续两次没有找到新链接，停止尝试")
                    break
            else:
                no_new_links_count = 0
            
            last_link_count = len(news_links)
            
            # 6. 尝试模拟人类行为
            if len(news_links) < max_links and attempt < max_attempts:
                # 随机等待
                wait_time = random.uniform(3, 6)
                print(f"等待 {wait_time:.1f} 秒后继续...")
                time.sleep(wait_time)
                
                # 随机上下滚动
                random_scroll = random.randint(200, 800)
                driver.execute_script(f"window.scrollBy(0, {-random_scroll});")
                time.sleep(0.5)
                driver.execute_script(f"window.scrollBy(0, {random_scroll});")
                time.sleep(0.5)
        
        # 7. 尝试从页面源代码中提取链接
        if len(news_links) < max_links:
            print("尝试从页面源代码中提取链接...")
            try:
                html = driver.page_source
                # 改进的正则表达式匹配
                pattern = r'https?://(?!www\.google\.com|accounts\.google\.com|support\.google\.com|policies\.google\.com)[^"\'>]+'
                all_urls = re.findall(pattern, html)
                
                for url in all_urls:
                    # 清理URL
                    url = url.split('&')[0].split('?')[0]
                    url = url.split('"')[0].split("'")[0]
                    
                    if (url.startswith('http') and 
                        "google.com" not in url and
                        not any(x in url.lower() for x in [
                            'youtube.com', '/video/', 'watch?v=', 
                            '.pdf', '.jpg', '.png', '.gif', '.jpeg',
                            'accounts.google', 'support.google', 'policies.google'
                        ])):
                        
                        if url not in news_links and len(news_links) < max_links and len(url) < 300:
                            news_links.append(url)
                            print(f"从源代码找到链接 {len(news_links)}: {url[:80]}...")
            except Exception as e:
                print(f"从源代码提取链接失败: {e}")
        
        # 去重
        news_links = list(dict.fromkeys(news_links))
        
        print(f"\n总共找到 {len(news_links)} 个新闻链接")
        return news_links[:max_links]
    
    def fetch_with_selenium(self, url, timeout=20):
        """
        使用Selenium作为备用方法获取网页内容
        """
        print(f"使用Selenium获取: {url[:80]}...")
        
        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument(f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            prefs = {"profile.managed_default_content_settings.images": 2}
            options.add_experimental_option("prefs", prefs)
            
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(timeout)
            
            driver.get(url)
            time.sleep(random.uniform(1, 2))
            
            html_content = driver.page_source
            driver.quit()
            
            if html_content and len(html_content) > 1000:
                return html_content
            else:
                return None
                
        except Exception as e:
            print(f"Selenium获取失败 {url}: {e}")
            return None
    
    def fetch_webpage_content(self, url, timeout=10):
        """
        获取网页HTML内容
        """
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        ]
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://www.google.com/'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, verify=True)
            
            if response.status_code == 403 or response.status_code == 429:
                print(f"网站返回{response.status_code}错误: {url}")
                return self.fetch_with_selenium(url)
            elif response.status_code == 404:
                return None
            elif response.status_code != 200:
                print(f"HTTP错误 {response.status_code}: {url}")
                return self.fetch_with_selenium(url)
            
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                return None
            
            if response.encoding is None or response.encoding.lower() == 'iso-8859-1':
                response.encoding = response.apparent_encoding or 'utf-8'
            
            html_content = response.text
            
            if len(html_content) < 1000:
                return self.fetch_with_selenium(url)
            
            return html_content
            
        except requests.exceptions.Timeout:
            print(f"请求超时: {url}")
            return self.fetch_with_selenium(url)
        except requests.exceptions.SSLError:
            try:
                response = requests.get(url, headers=headers, timeout=timeout, verify=False)
                return response.text
            except:
                return self.fetch_with_selenium(url)
        except Exception as e:
            print(f"获取网页内容失败 {url}: {e}")
            return self.fetch_with_selenium(url)
    
    def fetch_asset_month_data(self, asset_name, asset_config, year, month, max_news=20):
        """
        优化版：获取指定资产和月份的数据
        """
        print(f"\n{'='*60}")
        print(f"获取 {asset_name} - {year}年{month}月 数据")
        print(f"{'='*60}")
        
        # 初始化driver
        if not self.driver:
            self.driver = self.setup_driver(headless=CHROME_OPTIONS["headless"])
        
        # 获取日期范围
        start_date, end_date = self.get_month_date_range(year, month)
        print(f"日期范围: {start_date} 到 {end_date}")
        
        # 尝试不同的搜索词
        search_terms = asset_config["search_terms"]
        
        # 为某些资产添加额外的搜索词
        additional_terms = {
            "S&P 500": ["S&P500 index", "S&P 500 news", "SPX index", "S&P 500 stock", "Standard & Poor 500"],
            "NASDAQ Composite": ["NASDAQ news", "NASDAQ market", "IXIC index", "NASDAQ stock", "NASDAQ composite index"],
            "Dow Jones Industrial Average": ["Dow Jones news", "DJIA index", "Dow 30", "Dow Jones stock", "Dow Jones average"],
            "CAC 40": ["CAC 40 index", "CAC 40 news", "French stock index", "Paris stock market", "CAC40 index"],
            "FTSE 100": ["FTSE 100 index", "FTSE 100 news", "UK stock index", "London stock market", "FTSE100"],
            "EuroStoxx 50": ["Euro Stoxx 50 index", "Euro Stoxx 50 news", "European stock index", "STOXX 50"],
            "Hang Seng Index": ["Hang Seng index", "HSI news", "Hong Kong stock", "Hong Kong stock index"],
            "Shanghai Composite": ["Shanghai Composite index", "SSE Composite news", "China stock index", "Shanghai stock"],
            "BSE Sensex": ["BSE Sensex index", "Sensex news", "India stock index", "Bombay stock exchange"],
            "Nifty 50": ["Nifty 50 index", "Nifty news", "NSE Nifty index", "India stock market"],
            "KOSPI": ["KOSPI index", "Korea stock index", "South Korea stock", "Korean stock market"],
            "Gold": ["gold price today", "gold market", "XAUUSD", "gold spot price", "gold bullion"],
            "Silver": ["silver price today", "silver market", "XAGUSD", "silver spot price", "silver bullion"],
            "WTI Crude Oil Futures": ["crude oil price", "oil futures", "WTI price", "West Texas Intermediate", "oil market"]
        }
        
        if asset_name in additional_terms:
            search_terms = search_terms + additional_terms[asset_name]
        
        # 去重
        search_terms = list(dict.fromkeys(search_terms))
        
        news_links = []
        used_search_term = ""
        
        for search_term in search_terms:
            if len(news_links) >= max_news:
                break
                
            print(f"\n尝试搜索词: {search_term}")
            used_search_term = search_term
            
            # 构建搜索URL
            search_url = self.construct_google_news_url(search_term, start_date, end_date)
            print(f"搜索URL: {search_url[:100]}...")
            
            # 获取新闻链接
            print(f"尝试获取最多{max_news}条新闻链接...")
            month_links = self.fetch_news_links(self.driver, search_url, max_news)
            
            if month_links:
                # 合并但不重复
                for link in month_links:
                    if link not in news_links and len(news_links) < max_news:
                        news_links.append(link)
                
                print(f"当前找到 {len(news_links)} 个链接")
                
                if len(news_links) >= max_news:
                    break
            else:
                print("未找到链接，尝试下一个搜索词...")
            
            # 搜索词之间延迟
            time.sleep(random.uniform(3, 5))
        
        if not news_links:
            print(f"{asset_name} - {year}年{month}月未找到任何新闻链接")
            return []
        
        print(f"\n找到的链接列表 ({len(news_links)}个):")
        for i, link in enumerate(news_links[:5], 1):
            print(f"{i:2d}. {link[:100]}...")
        if len(news_links) > 5:
            print(f"... 和另外 {len(news_links)-5} 个链接")
        
        # 抓取并保存网页内容
        print(f"\n开始抓取{len(news_links)}个新闻页面...")
        
        raw_data_list = []
        
        for i, link in enumerate(news_links, 1):
            print(f"\n[{i}/{len(news_links)}] 获取: {link[:80]}...")
            
            # 智能延迟
            if i > 1:
                delay = random.uniform(*DELAY_BETWEEN_REQUESTS)
                time.sleep(delay)
            
            # 获取网页内容
            html_content = self.fetch_webpage_content(link)
            
            if html_content:
                # 创建原始数据对象
                raw_data = {
                    'asset_name': asset_name,
                    'category': asset_config["category"],
                    'year': year,
                    'month': month,
                    'search_term': used_search_term,
                    'url': link,
                    'html_content': html_content,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'content_length': len(html_content)
                }
                
                raw_data_list.append(raw_data)
                print(f"   成功! ({len(html_content)} 字符)")
            else:
                print(f"   获取网页内容失败")
        
        return raw_data_list
    
    def save_raw_data(self, raw_data_list, asset_name, year, month):
        """
        保存原始数据
        """
        if not raw_data_list:
            return
        
        import os
        import json
        
        # 创建目录
        asset_safe_name = asset_name.replace('/', '_').replace(' ', '_')
        asset_dir = os.path.join(self.output_base_dir, asset_safe_name)
        month_dir = os.path.join(asset_dir, f"{year}_{month:02d}")
        os.makedirs(month_dir, exist_ok=True)
        
        # 保存原始HTML文件
        for i, raw_data in enumerate(raw_data_list, 1):
            html_filename = os.path.join(month_dir, f"news_{i:03d}_raw.html")
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(raw_data['html_content'])
        
        # 保存元数据
        metadata_filename = os.path.join(month_dir, f"raw_metadata.json")
        metadata = []
        for i, raw_data in enumerate(raw_data_list, 1):
            metadata.append({
                'index': i,
                'asset_name': raw_data['asset_name'],
                'category': raw_data.get('category', ''),
                'year': raw_data['year'],
                'month': raw_data['month'],
                'search_term': raw_data.get('search_term', ''),
                'url': raw_data['url'],
                'timestamp': raw_data['timestamp'],
                'content_length': raw_data['content_length']
            })
        
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"原始数据已保存到: {month_dir}")
    
    def fetch_all_data(self, year=YEAR, months=MONTHS, max_news_per_month=MAX_NEWS_PER_MONTH):
        """
        获取所有资产的所有月份数据
        """
        print("=" * 60)
        print(f"开始获取 {len(ASSETS_CONFIG)} 种资产的新闻数据")
        print(f"时间范围: {year}年{months[0]}月 - {year}年{months[-1]}月")
        print(f"每月每资产目标新闻数: {max_news_per_month}")
        print("=" * 60)
        
        all_raw_data = []
        total_assets = len(ASSETS_CONFIG)
        
        try:
            # 初始化driver
            self.driver = self.setup_driver(headless=CHROME_OPTIONS["headless"])
            
            # 按资产处理
            for asset_idx, (asset_name, asset_config) in enumerate(ASSETS_CONFIG.items(), 1):
                print(f"\n{'='*60}")
                print(f"处理资产 {asset_idx}/{total_assets}: {asset_name}")
                print(f"{'='*60}")
                
                asset_raw_data = []
                
                # 按月份处理
                for month_idx, month in enumerate(months, 1):
                    print(f"\n处理月份 {month_idx}/{len(months)}")
                    
                    try:
                        # 获取当前资产和月份的数据
                        month_raw_data = self.fetch_asset_month_data(
                            asset_name, asset_config, year, month, max_news_per_month
                        )
                        
                        if month_raw_data:
                            asset_raw_data.extend(month_raw_data)
                            # 保存原始数据
                            self.save_raw_data(month_raw_data, asset_name, year, month)
                        
                        # 月份之间的延迟
                        if month < months[-1]:
                            delay = random.uniform(*DELAY_BETWEEN_MONTHS)
                            print(f"\n等待 {delay:.1f} 秒后处理下一个月...")
                            time.sleep(delay)
                            
                    except Exception as e:
                        print(f"处理{asset_name} - {year}年{month}月时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        print("等待5秒后继续...")
                        time.sleep(5)
                        continue
                
                if asset_raw_data:
                    all_raw_data.extend(asset_raw_data)
                
                # 资产之间的延迟
                if asset_idx < total_assets:
                    delay = random.uniform(*DELAY_BETWEEN_ASSETS)
                    print(f"\n等待 {delay:.1f} 秒后处理下一个资产...")
                    time.sleep(delay)
            
            print(f"\n数据获取完成!")
            print(f"总共获取了 {len(all_raw_data)} 条新闻的原始数据")
            
            return all_raw_data
            
        finally:
            # 关闭driver
            if self.driver:
                print("\n关闭浏览器...")
                self.driver.quit()
                self.driver = None
    
    def close(self):
        """关闭资源"""
        if self.driver:
            self.driver.quit()
            self.driver = None


# 独立运行测试
if __name__ == "__main__":
    # 测试单个资产的单个月份
    fetcher = DataFetcher(output_base_dir="test_fetched_data")
    
    # 测试获取S&P 500 2024年1月的数据
    test_asset = "S&P 500"
    test_config = ASSETS_CONFIG[test_asset]
    
    print("测试数据获取...")
    raw_data = fetcher.fetch_asset_month_data(
        test_asset, test_config, 2024, 1, max_news=5
    )
    
    if raw_data:
        fetcher.save_raw_data(raw_data, test_asset, 2024, 1)
        print(f"测试完成，获取了 {len(raw_data)} 条新闻")
    else:
        print("测试失败，未获取到数据")
    
    fetcher.close()