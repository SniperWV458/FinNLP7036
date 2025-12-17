import time
from tqdm import tqdm

from cache_util import MarketDataStore
from constants import Constants


def fetch_all_tickers():
    store = MarketDataStore(root_dir="../market_data", use_duckdb=True)
    for ticker in tqdm(Constants.tickers):
        try:
            store.ensure_ohlcv(ticker, Constants.start_date, Constants.end_date)
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
        time.sleep(2)


if __name__ == "__main__":
    fetch_all_tickers()
