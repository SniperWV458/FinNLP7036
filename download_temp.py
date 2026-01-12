from __future__ import annotations

import time
import random
from typing import Iterable, Dict, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

tickers = [
"^GSPC", # S&P 500
"^IXIC", # NASDAQ Composite
"^DJI", # Dow Jones Industrial Average
"^FCHI", # CAC 40 (France)
"^FTSE", # FTSE 100 (UK)
"^STOXX50E",# EuroStoxx 50
"^HSI", # Hang Seng Index (Hong Kong)
"000001.SS",# Shanghai Composite (China)
"^BSESN", # BSE Sensex (India)
"^NSEI", # Nifty 50 (India)
"^KS11", # KOSPI (South Korea)
"GC=F", # Gold
"SI=F", # Silver
"CL=F", # WTI Crude Oil Futures
]

def fetch_close_prices_one_by_one(
    tickers: Iterable[str],
    start: str,
    end: str,
    *,
    base_sleep: float = 2.0,          # sleep between tickers
    max_retries: int = 6,             # per-ticker retries
    backoff_base: float = 5.0,        # initial backoff seconds after an error
    backoff_cap: float = 300.0,       # cap backoff
    jitter: float = 0.35,             # +/- jitter fraction
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Download Close prices per ticker (safer than batch), with:
      - 2s baseline sleep between requests
      - retry + exponential backoff for rate limits / transient failures
      - partial success preserved
    Returns a DataFrame of Close prices (Date index, columns=tickers).
    """
    tickers = [t.strip() for t in tickers if str(t).strip()]
    closes: Dict[str, pd.Series] = {}

    iterator = tqdm(tickers, desc="Downloading", unit="ticker") if show_progress else tickers

    for tkr in iterator:
        success = False
        attempt = 0

        while attempt <= max_retries and not success:
            try:
                df = yf.download(
                    tkr,
                    start=start,
                    end=end,
                    actions=False,
                    auto_adjust=False,
                    progress=False,
                    threads=False,   # avoid parallel spikes; safer for rate limits
                )

                if df is None or df.empty:
                    raise ValueError("Empty response")

                # yfinance returns columns like ["Open","High","Low","Close",...]
                close = df["Close"].copy()
                close.name = tkr

                closes[tkr] = close
                success = True

            except Exception as e:
                msg = str(e)
                is_rate_limited = ("429" in msg) or ("Too Many Requests" in msg) or ("rate limit" in msg.lower())

                if attempt >= max_retries:
                    # Give up on this ticker but continue others
                    if show_progress:
                        tqdm.write(f"[FAIL] {tkr}: {e}")
                    break

                # Exponential backoff (stronger for suspected rate limit)
                mult = 2.0 if is_rate_limited else 1.0
                wait = min(backoff_cap, (backoff_base * (2 ** attempt)) * mult)

                # Add jitter so you don't look like a bot with fixed intervals
                wait *= (1.0 + random.uniform(-jitter, jitter))
                wait = max(0.5, wait)

                if show_progress:
                    tqdm.write(f"[RETRY] {tkr} attempt {attempt+1}/{max_retries} in {wait:.1f}s: {e}")

                time.sleep(wait)
                attempt += 1

        # Baseline pacing between tickers (even after success)
        time.sleep(base_sleep)

    if not closes:
        return pd.DataFrame()

    out = pd.concat(closes.values(), axis=1).sort_index()
    return out


if __name__ == "__main__":
    start_date = "2003-01-01"
    end_date = "2024-12-31"

    # Example: tickers = ["AAPL", "MSFT", "NVDA"]
    data = fetch_close_prices_one_by_one(tickers, start_date, end_date, base_sleep=2.0)

    if not data.empty:
        data.to_csv("stock_closing_prices.csv")
        print("Closing prices saved to stock_closing_prices.csv")
    else:
        print("No data to save.")
