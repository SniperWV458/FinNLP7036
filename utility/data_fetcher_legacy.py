import time
import yfinance as yf
import pandas as pd

tickers = [
    "^GSPC",  # S&P 500
    "^IXIC",  # NASDAQ Composite
    "^DJI",  # Dow Jones Industrial Average
    "^FCHI",  # CAC 40 (France)
    "^FTSE",  # FTSE 100 (UK)
    "^STOXX50E",  # EuroStoxx 50
    "^HSI",  # Hang Seng Index (Hong Kong)
    "000001.SS",  # Shanghai Composite (China)
    "^BSESN",  # BSE Sensex (India)
    "^NSEI",  # Nifty 50 (India)
    "^KS11",  # KOSPI (South Korea)
    "GC=F",  # Gold
    "SI=F",  # Silver
    "CL=F",  # WTI Crude Oil Futures
]


start_date = "2003-01-01"  # Example start date
end_date = "2024-12-31"  # Example end date


# Function to fetch data with rate limit handling
def fetch_data(tickers, start, end):
    retry = True
    delay = 60  # Delay for rate limit errors
    data = None  # Initialize data to None
    while retry:
        try:
            # Batch download, specify actions=False to avoid extra data
            data = yf.download(tickers, start=start, end=end, actions=False)
            # If multi-ticker, extract 'Close' from the multi-level columns
            if len(tickers) > 1:
                data = data['Close']
            else:
                data = data[['Close']]  # Single ticker case
            retry = False
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit hit. Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Error fetching data: {e}")
                retry = False
                data = pd.DataFrame()  # Return empty DataFrame on failure
    return data


# Download closing prices
data = fetch_data(tickers, start_date, end_date)

# Check if data is not empty
if not data.empty:
    # Save to CSV
    data.to_csv("stock_closing_prices.csv")
    print("Closing prices saved to stock_closing_prices.csv")
else:
    print("No data to save.")

# Add delay to avoid rate limits for subsequent requests
time.sleep(2)
