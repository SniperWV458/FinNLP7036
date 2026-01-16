# crawler.py - 增强版网页内容爬取
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import re
import random
from urllib.parse import urlparse, urljoin, urlunparse, urlsplit, urlunsplit
from config import CLEANING_CONFIG, CRAWLER_CONFIG, LUNA_PROXY_CONFIG
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing as mp
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import urllib3
import os
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ==================== 多进程Worker函数 ====================
def crawl_worker(args):
    """
    多进程爬取工作函数
    Args:
        args: (row_dict, use_proxy, retry_count)
    Returns:
        处理后的行数据字典
    """
    row_dict, use_proxy, retry_count = args
    url = row_dict.get('processed_url', row_dict.get('url', ''))
    
    if not url or url == "":
        row_dict['raw_content'] = ""
        row_dict['content_length'] = 0
        row_dict['crawl_success'] = False
        row_dict['error_msg'] = "Empty URL"
        return row_dict
    
    # 每个worker创建自己的session以获得独立的IP
    crawler = NewsCrawler(use_proxy=use_proxy)
    
    # 尝试爬取
    for attempt in range(retry_count):
        try:
            content = crawler.crawl_url(url, retry_count=attempt)
            
            if content and len(content.split()) > 30:
                row_dict['raw_content'] = content
                row_dict['content_length'] = len(content.split())
                row_dict['crawl_success'] = True
                row_dict['error_msg'] = ""
                row_dict['worker_pid'] = os.getpid()
                return row_dict
            
            # 如果内容为空，等待后重试
            if attempt < retry_count - 1:
                time.sleep(random.uniform(2, 5))
                
        except Exception as e:
            error_msg = str(e)[:200]
            print(f"[Worker {os.getpid()}] 爬取失败 (尝试 {attempt+1}/{retry_count}): {url[:80]} - {error_msg}")
            
            if attempt < retry_count - 1:
                time.sleep(random.uniform(3, 6))
            else:
                row_dict['error_msg'] = error_msg
    
    # 所有重试都失败
    row_dict['raw_content'] = ""
    row_dict['content_length'] = 0
    row_dict['crawl_success'] = False
    row_dict['worker_pid'] = os.getpid()
    
    if 'error_msg' not in row_dict:
        row_dict['error_msg'] = "All retries failed"
    
    return row_dict


class NewsCrawler:
    def __init__(self, use_proxy=None):
        """
        初始化爬虫
        Args:
            use_proxy: None (使用配置), True (强制使用), False (强制不使用)
        """
        self.use_proxy = use_proxy if use_proxy is not None else LUNA_PROXY_CONFIG['enabled']
        self.session = self._create_session()
        self.current_ua_index = 0
        self.lock = threading.Lock()
        self.domain_stats = {}
        self.session.headers.update({
            'User-Agent': random.choice(CRAWLER_CONFIG['user_agents']),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'close',  # 使用close避免连接池问题
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        })
    
    def _create_session(self):
        """创建带有重试机制和代理的会话"""
        session = requests.Session()
        
        # 设置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 配置Luna代理
        if self.use_proxy:
            username = LUNA_PROXY_CONFIG['username']
            password = LUNA_PROXY_CONFIG['password']
            server = LUNA_PROXY_CONFIG['server']
            
            proxies = {
                "http": f"http://{username}:{password}@{server}",
                "https": f"http://{username}:{password}@{server}",
            }
            session.proxies.update(proxies)
            print(f"[Proxy] Luna代理已启用 - {server}")
        
        return session
    
    def preprocess_url(self, url):
        """预处理和修复URL"""
        if pd.isna(url) or not isinstance(url, str):
            return ""
        
        url = url.strip()
        
        # 移除URL中的特殊字符和空格
        url = re.sub(r'\s+', '', url)
        
        # 处理已知的重定向
        for old_prefix, new_prefix in CRAWLER_CONFIG['url_redirects'].items():
            if old_prefix in url:
                url = url.replace(old_prefix, new_prefix)
        
        # 确保URL有协议
        if not url.startswith(('http://', 'https://')):
            if url.startswith('www.'):
                url = 'https://' + url
            else:
                url = 'https://' + url if '://' not in url else url
        
        # 修复双斜杠和格式问题
        url = re.sub(r'(?<!:)/{2,}', '/', url)
        url = re.sub(r'https?:///+', 'https://', url)
        
        # 验证URL格式
        try:
            parsed = urlparse(url)
            
            # 检查域名是否需要跳过
            domain = parsed.netloc.lower()
            for skip_domain in CRAWLER_CONFIG['skip_domains']:
                if skip_domain in domain:
                    return ""
            
            # 重建URL
            parsed = parsed._replace(fragment='')  # 移除片段
            url = urlunparse(parsed)
            
            return url
        except Exception as e:
            print(f"URL解析失败 {url}: {e}")
            return ""
    
    def _rotate_user_agent(self):
        """轮换User-Agent"""
        with self.lock:
            self.current_ua_index = (self.current_ua_index + 1) % len(CRAWLER_CONFIG['user_agents'])
            new_ua = CRAWLER_CONFIG['user_agents'][self.current_ua_index]
            self.session.headers.update({'User-Agent': new_ua})
    
    def _get_domain_delay(self, url):
        """根据域名获取延迟时间"""
        try:
            domain = urlparse(url).netloc.lower()
            for site_rule, config in CRAWLER_CONFIG['site_rules'].items():
                if site_rule in domain:
                    return config.get('delay', CRAWLER_CONFIG['delay_between_requests'])
        except:
            pass
        return CRAWLER_CONFIG['delay_between_requests']
    
    def _extract_with_selectors(self, soup):
        """使用多种选择器提取文章内容"""
        selectors = [
            'article',
            '[class*="article"]',
            '[class*="content"]',
            '[class*="story"]',
            '[class*="post-content"]',
            '[class*="article-body"]',
            '[class*="article-content"]',
            '.article__content',
            '.story-body',
            '.post-content',
            '.entry-content',
            '.article-text',
            '.articleBody',
            '.content-article',
            '.story-content',
            '.article-content',
            '[itemprop="articleBody"]',
            '.article__body',
        ]
        
        for selector in selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    if element:
                        text = element.get_text(separator=' ', strip=True)
                        if len(text.split()) > 100:  # 确保有足够内容
                            return text
            except:
                continue
        
        return None
    
    def _extract_from_body(self, soup):
        """从body中智能提取"""
        # 移除不需要的标签
        for tag in ['script', 'style', 'nav', 'footer', 'header', 'aside', 
                    'iframe', 'form', 'button', 'input', 'select', 'textarea',
                    'svg', 'canvas', 'noscript']:
            for element in soup.find_all(tag):
                element.decompose()
        
        # 移除常见导航和广告
        for element in soup.find_all(class_=re.compile(r'nav|menu|sidebar|ad|banner|promo|popup|modal|cookie')):
            element.decompose()
        
        # 提取所有文本段落
        paragraphs = []
        for p in soup.find_all(['p', 'div']):
            if p.find_parent(['nav', 'footer', 'header', 'aside']):
                continue
            
            text = p.get_text(separator=' ', strip=True)
            words = text.split()
            
            # 过滤短段落和可能的广告/导航
            if len(words) >= 10 and len(words) <= 500:
                # 检查是否包含导航关键词
                text_lower = text.lower()
                if not any(keyword in text_lower for keyword in CLEANING_CONFIG['navigation_keywords']):
                    paragraphs.append(text)
        
        # 按长度排序，取最长的几个段落
        paragraphs.sort(key=len, reverse=True)
        combined_text = ' '.join(paragraphs[:20])  # 取前20个最长段落
        
        if len(combined_text.split()) > 50:
            return combined_text
        
        return None
    
    def extract_article_content(self, html_content, url):
        """改进的文章内容提取"""
        if not html_content or len(html_content.strip()) < 1000:
            return ""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 策略1: 使用选择器提取
            content = self._extract_with_selectors(soup)
            if content and len(content.split()) > 100:
                return content
            
            # 策略2: 智能提取
            content = self._extract_from_body(soup)
            if content and len(content.split()) > 50:
                return content
            
            # 策略3: 提取所有文本并过滤
            all_text = soup.get_text(separator=' ', strip=True)
            
            # 移除多余空格
            all_text = re.sub(r'\s+', ' ', all_text)
            
            # 移除明显的导航文本
            lines = []
            for line in all_text.split('. '):
                line = line.strip()
                if len(line.split()) > 5 and len(line.split()) < 100:
                    line_lower = line.lower()
                    if not any(keyword in line_lower for keyword in CLEANING_CONFIG['remove_keywords'] + CLEANING_CONFIG['navigation_keywords']):
                        lines.append(line)
            
            content = '. '.join(lines)
            if len(content.split()) > 30:
                return content
            
        except Exception as e:
            print(f"内容提取错误 {url}: {e}")
        
        return ""
    
    def crawl_url(self, url, retry_count=0):
        """爬取单个URL的内容"""
        if not url or url == "":
            return ""
        
        # 轮换User-Agent
        if retry_count > 0:
            self._rotate_user_agent()
        
        try:
            # 添加随机延迟
            time.sleep(random.uniform(0.5, 2.0))
            
            # 根据域名应用特定延迟
            domain_delay = self._get_domain_delay(url)
            time.sleep(random.uniform(domain_delay * 0.5, domain_delay * 1.5))
            
            response = self.session.get(
                url, 
                timeout=CRAWLER_CONFIG['timeout'],
                verify=False,  # 禁用SSL验证
                allow_redirects=True
            )
            
            # 处理不同的状态码
            if response.status_code == 403:
                # 尝试不同的方法
                if retry_count < 1:
                    self.session.headers.update({
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'User-Agent': random.choice(CRAWLER_CONFIG['user_agents'])
                    })
                    return self.crawl_url(url, retry_count + 1)
                return ""
            
            elif response.status_code == 404:
                print(f"页面不存在: {url}")
                return ""
            
            elif response.status_code == 429:  # 请求过多
                print(f"请求过多，等待后重试: {url}")
                time.sleep(10)
                if retry_count < 2:
                    return self.crawl_url(url, retry_count + 1)
                return ""
            
            elif response.status_code >= 500:
                print(f"服务器错误 {response.status_code}: {url}")
                if retry_count < 1:
                    time.sleep(5)
                    return self.crawl_url(url, retry_count + 1)
                return ""
            
            elif response.status_code == 200:
                # 检查内容类型
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type and 'text/plain' not in content_type:
                    print(f"非HTML内容: {url} - {content_type}")
                    return ""
                
                # 检查内容长度
                if len(response.content) < 1000:
                    print(f"内容过短: {url} - {len(response.content)} bytes")
                    return ""
                
                # 提取内容
                content = self.extract_article_content(response.text, url)
                
                if content and len(content.split()) > 30:
                    return content
                else:
                    print(f"内容提取失败或内容过短: {url}")
                    return ""
            
            else:
                print(f"HTTP错误 {response.status_code}: {url}")
                return ""
                
        except requests.exceptions.Timeout:
            print(f"请求超时: {url}")
            if retry_count < 2:
                time.sleep(3)
                return self.crawl_url(url, retry_count + 1)
            return ""
            
        except requests.exceptions.ConnectionError:
            print(f"连接错误: {url}")
            if retry_count < 2:
                time.sleep(5)
                return self.crawl_url(url, retry_count + 1)
            return ""
            
        except requests.exceptions.TooManyRedirects:
            print(f"重定向过多: {url}")
            return ""
            
        except Exception as e:
            print(f"爬取失败 {url}: {str(e)[:100]}")
            return ""
    
    def crawl_url_wrapper(self, row):
        """包装函数，用于多线程"""
        url = row['url'] if isinstance(row, dict) else row
        content = self.crawl_url(url)
        
        if isinstance(row, dict):
            row['raw_content'] = content
            row['content_length'] = len(content.split()) if content else 0
            row['crawl_success'] = bool(content)
        else:
            row = {
                'url': url,
                'raw_content': content,
                'content_length': len(content.split()) if content else 0,
                'crawl_success': bool(content)
            }
        
        return row
    
    def process_batch(self, df, output_file, sample_size=None, use_threading=True, use_multiprocessing=None):
        """
        批量处理URL
        Args:
            df: 输入数据框
            output_file: 输出文件路径
            sample_size: 样本大小
            use_threading: 是否使用多线程（当use_multiprocessing=False时）
            use_multiprocessing: None (使用配置), True (强制多进程), False (使用线程或单线程)
        """
        print(f"开始批量处理，总URL数: {len(df)}")
        
        # 确定是否使用多进程
        if use_multiprocessing is None:
            use_multiprocessing = LUNA_PROXY_CONFIG.get('use_multiprocessing', False)
        
        # 如果指定了样本大小，则抽样处理
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"抽样处理 {sample_size} 个URL")
        
        # 预处理URL
        print("预处理URL...")
        df['processed_url'] = df['url'].apply(self.preprocess_url)
        
        # 过滤无效URL
        original_count = len(df)
        df = df[df['processed_url'] != ""]
        print(f"有效URL数: {len(df)} / {original_count}")
        
        if len(df) == 0:
            print("没有有效的URL可处理")
            return pd.DataFrame()
        
        results = []
        successful = 0
        
        if use_multiprocessing and len(df) > 1:
            # ========== 多进程处理 ==========
            pool_size = LUNA_PROXY_CONFIG.get('pool_size', 5)
            retry_count = LUNA_PROXY_CONFIG.get('max_retries_per_worker', 3)
            
            print(f"使用多进程处理，进程数: {pool_size}，每个URL重试次数: {retry_count}")
            print(f"代理模式: {'启用' if self.use_proxy else '禁用'}")
            
            # 准备任务参数
            tasks = []
            for idx, row in df.iterrows():
                tasks.append((row.to_dict(), self.use_proxy, retry_count))
            
            # 创建进程池并处理
            with mp.Pool(processes=pool_size) as pool:
                # 使用imap_unordered获取实时进度
                for idx, result in enumerate(pool.imap_unordered(crawl_worker, tasks), 1):
                    results.append(result)
                    
                    if result.get('crawl_success'):
                        successful += 1
                    
                    # 显示进度
                    if idx % 10 == 0 or idx == len(tasks):
                        success_rate = (successful / idx * 100) if idx > 0 else 0
                        print(f"[进度] {idx}/{len(tasks)} | 成功: {successful} ({success_rate:.1f}%)")
            
            print(f"\n多进程爬取完成！")
            
        elif use_threading and len(df) > 1:
            # ========== 多线程处理 ==========
            print(f"使用多线程处理，线程数: {CRAWLER_CONFIG['max_workers']}")
            
            with ThreadPoolExecutor(max_workers=CRAWLER_CONFIG['max_workers']) as executor:
                # 准备任务
                tasks = []
                for idx, row in df.iterrows():
                    task = executor.submit(self.crawl_url_wrapper, row.to_dict())
                    tasks.append(task)
                
                # 处理结果
                for idx, future in enumerate(as_completed(tasks), 1):
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result.get('crawl_success'):
                            successful += 1
                        
                        if idx % 10 == 0 or idx == len(tasks):
                            print(f"已处理 {idx}/{len(tasks)}，成功 {successful}")
                            
                    except Exception as e:
                        print(f"任务处理失败: {e}")
        else:
            # ========== 单线程处理 ==========
            print("使用单线程处理")
            for idx, row in df.iterrows():
                url = row['processed_url']
                print(f"处理 {idx+1}/{len(df)}: {url[:100]}...")
                
                content = self.crawl_url(url)
                
                result = row.to_dict()
                result['raw_content'] = content
                result['content_length'] = len(content.split()) if content else 0
                result['crawl_success'] = bool(content)
                results.append(result)
                
                if content:
                    successful += 1
                
                if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
                    print(f"进度: {idx+1}/{len(df)}，成功: {successful}")
        
        # 转换为DataFrame
        result_df = pd.DataFrame(results)
        
        # 计算成功率
        if len(result_df) > 0:
            success_rate = (successful / len(result_df)) * 100
            print(f"\n{'='*60}")
            print(f"爬取完成: 总共 {len(result_df)}，成功 {successful}，成功率 {success_rate:.1f}%")
            
            # 统计worker进程信息（如果是多进程模式）
            if 'worker_pid' in result_df.columns:
                unique_pids = result_df['worker_pid'].nunique()
                print(f"使用的进程数: {unique_pids}")
            
            # 统计域名成功率
            result_df['domain'] = result_df['processed_url'].apply(
                lambda x: urlparse(x).netloc if isinstance(x, str) else ''
            )
            domain_stats = result_df.groupby('domain')['crawl_success'].agg(['count', 'sum']).round(2)
            domain_stats['rate'] = (domain_stats['sum'] / domain_stats['count'] * 100).round(1)
            print("\n域名成功率统计 (Top 20):")
            print(domain_stats.sort_values('rate', ascending=False).head(20))
            print('='*60)
        
        # 保存结果
        if output_file:
            result_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"原始数据已保存至: {output_file}")
        
        return result_df