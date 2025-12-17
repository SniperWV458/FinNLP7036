from cache_util import MarketDataStore
from constants import Constants


def preprocess_data():
    store = MarketDataStore(root_dir="../market_data", use_duckdb=True)
    store.clean_then_build_wide_then_normalize_and_log(
        tickers=Constants.tickers,
        wide_out_path="../market_data/wide_prices.parquet",
        normalized_out_path="../market_data/wide_prices_normalized.parquet",
        log_out_path="../market_data/wide_prices_log.parquet",
        price_col="Adj Close",
        clean_first=True,
    )


if __name__ == "__main__":
    preprocess_data()
