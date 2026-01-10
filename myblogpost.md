---
Title: Asset-Level Textual Dataset Construction and Empirical Attribution of Sentiment Signals in Hierarchical Reinforced Learning Portfolio Optimization(by Group "Dice8"）
Date: 2026-01-10 19:15
Category: Reflective Report
Tags: Group
---

##Text Processing

>Our project aims to construct a quality asset level textual dataset for machine
 learning training. News and social media sources are the primary focus to 
retrieve raw textual data. With robust, quality and scalable pipeline that covers the flow from source of text to model ready input features,
 several approaches are made to achieve our target.

*Note: Since the code is too lengthy, please refer to our [GitHub repository](https://github.com/SniperWV458/FinNLP7036/tree/Google_news2024_getandclean_v1/Google_news2024_getandclean_v1)
(Google_news2024_getandclean_v1) for the relevant implementation.*

#News source

We are going to build a sentiment analysis model for financial markets, and 
we need to use data from various news source for 14 major financial assets(from 2003 to 2024). The text data we collected include:

**1.Broad coverage of assets:** including equities like the S&P 500, Nasdaq, Dow Jones, as well as commodities like gold and crude oil. 

**2.Full time span:** from January 2003 to December 2024, covering multiple economic cycles.

**3.Structured data:** each news article needs metadata like headline, content, publication time, and source.

The code we use is as follows:
```python
import nltk
import pandas as pd
myvar = 8
DF = pd.read_csv('XRP-data.csv')
```

# Social media source

Based on our needs for building a financial market sentiment analysis model, social media data,
 in addition to mainstream news, is a crucial supplementary source. We attempted to extract relevant
 discussions from Reddit(especially financial subreddits like r/wallstreetbets and r/stocks) to extract retail investor sentiment.

The core crawling strategy and output summary are as follows:

### 1. Crawling Logic

We traversed multiple target subreddits and used a combination of sorting methods such as "Popular," "Newest," and "Best of the Year,"
 while also using keywords such as "S&P 500" for searching, attempting to maximize data coverage.

### 2. Main Output Results

```Python
Data Statistics:
Total Posts: 2374
Earliest Post: 2025-01-13 05:53:56
Latest Post: 2026-01-10 19:00:15
Daily Posting Statistics:

date
2026-01-01 97
2026-01-02 100
2026-01-03 84
2026-01-04 52
2026-01-05 63
2026-01-06 114
2026-01-07 149
2026-01-08 131
2026-01-09 144
2026-01-10 391
dtype: int64
```

Although the crawler successfully ran and retrieved thousands of data points, the results revealed a fundamental contradiction.
While the code is robust, having crawled 2000+ posts, the data's time span is only about one year, 
and the posts are highly concentrated around the crawl dates. This is unusable for research requiring backtesting with years of historical data (e.g., 2003-2024).

It is found out to be a limitation of the platform design. The Reddit API and page access mechanism default and prioritize returning the latest content creating an invisible barrier to deep access to historical data.
This attempt clearly shows that Reddit cannot directly provide long-term, uniform historical sentiment data. It also modified terms of use of data API to explicitly prohibit usage of Reddit data for commercialized and noncommercialized machine learning training.


**Other alternative channels also had shortcomings:** Twitter's (X) API policy had shifted towards commercialization, making historical data acquisition costly; 
YouTube's comment data, due to its dynamic page loading mechanism and API quota limitations, was difficult to crawl stably and in large quantities; and 
labeled datasets like Sentiment140 lacked accurate timestamps, making alignment with market data difficult.

To address these issues, we experimented with several open-source toolkits on GitHub specifically for data collection and explored other platforms. 
We found that `StockTwits`, which focuses on stock discussions, while requiring an API key application, had relatively clear and user-friendly interface rules, 
potentially providing a viable alternative data source for our research.



## How to Include a Quote

As a famous hedge fund manager once said:
>Fed watching is a great tool to make money. I have been making all my
>gazillions using this technique.



## How to Include an Image

Fed Chair Powell is working hard:

![Picture showing Powell]({static}/images/group-Fintech-Disruption_Powell.jpeg)
