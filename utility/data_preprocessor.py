from cache_util import MarketDataStore
from constants import Constants


def preprocess_data(store):
    store.clean_then_build_wide_then_normalize_and_log(
        tickers=Constants.tickers,
        wide_out_path="../market_data/wide_prices.parquet",
        normalized_out_path="../market_data/wide_prices_normalized.parquet",
        log_out_path="../market_data/wide_prices_log.parquet",
        price_col="Adj Close",
        clean_first=True,
    )




if __name__ == "__main__":
    store = MarketDataStore(root_dir="../market_data", use_duckdb=True)
    # preprocess_data(store)
    store.simulate_equal_weight_portfolio_from_wide_parquet_to_csv("../market_data/wide_prices.parquet", "../market_data/equal_weight_portfolio.csv", 1)