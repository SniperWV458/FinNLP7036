from __future__ import annotations

import os
import numpy as np
import pandas as pd
import yfinance as yf

try:
    import duckdb  # optional
except Exception:
    duckdb = None


class MarketDataStore:
    """
    Local OHLCV cache for yfinance data.

    Storage:
      - Parquet per ticker (simple, robust)
      - Optional DuckDB view/query (fast scans & joins; optional)
    """

    def __init__(self, root_dir: str = "./market_data", use_duckdb: bool = False):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

        self.use_duckdb = bool(use_duckdb and duckdb is not None)
        self._con = None
        if self.use_duckdb:
            db_path = os.path.join(self.root_dir, "market_data.duckdb")
            self._con = duckdb.connect(db_path)

    @staticmethod
    def _safe_ticker(ticker: str) -> str:
        return ticker.replace("^", "").replace("/", "_")

    def _parquet_path(self, ticker: str) -> str:
        return os.path.join(self.root_dir, f"{self._safe_ticker(ticker)}_ohlcv.parquet")

    def _read_parquet(self, ticker: str) -> pd.DataFrame | None:
        path = self._parquet_path(ticker)
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        df = self._normalize_date_column(df)
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        return df

    def _write_parquet(self, ticker: str, df: pd.DataFrame) -> None:
        df = self._normalize_date_column(df)

        if "Adj Close" not in df.columns and "Close" not in df.columns:
            raise ValueError(
                f"Refusing to cache data without Adj Close/Close. columns={list(df.columns)}"
            )

        if "Adj Close" not in df.columns and "Close" in df.columns:
            df = df.copy()
            df["Adj Close"] = df["Close"]

        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        df.to_parquet(self._parquet_path(ticker), index=False)

    def _yf_download_ohlcv(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

        if data is None or len(data) == 0:
            raise ValueError(f"No data returned for {ticker} in [{start}, {end}]")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

        if "Adj Close" not in data.columns and "Close" not in data.columns:
            raise ValueError(f"yfinance returned no Close/Adj Close. columns={list(data.columns)}")

        df = data.reset_index()
        df = self._normalize_date_column(df)

        out_cols = ["date"]
        for c in cols:
            if c in df.columns:
                out_cols.append(c)

        if ("Adj Close" not in out_cols) and ("Close" not in out_cols):
            raise ValueError(f"Output missing Adj Close/Close after normalization. cols={out_cols}")

        out = df[out_cols].copy().sort_values("date").reset_index(drop=True)

        if "Adj Close" not in out.columns and "Close" in out.columns:
            out["Adj Close"] = out["Close"]

        return out

    def _safe_download_ohlcv_best_effort(self, ticker: str, a: pd.Timestamp, b: pd.Timestamp) -> pd.DataFrame | None:
        if a > b:
            return None

        a_str = a.strftime("%Y-%m-%d")
        b_str = (b + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            return self._yf_download_ohlcv(ticker, a_str, b_str)
        except ValueError:
            probe_sizes = [30, 180, 365, 365 * 3, 365 * 10]
            for days in probe_sizes:
                a2 = a + pd.Timedelta(days=days)
                if a2 > b:
                    break
                try:
                    df2 = self._yf_download_ohlcv(ticker, a2.strftime("%Y-%m-%d"), b_str)
                    if df2 is not None and len(df2) > 0:
                        return df2
                except ValueError:
                    continue

            step = 365 * 5
            cur = a
            while cur <= b:
                try:
                    df3 = self._yf_download_ohlcv(ticker, cur.strftime("%Y-%m-%d"), b_str)
                    if df3 is not None and len(df3) > 0:
                        return df3
                except ValueError:
                    pass
                cur = cur + pd.Timedelta(days=step)

            return None

    def ensure_ohlcv(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        local = self._read_parquet(ticker)

        if local is None or local.empty:
            fetched = self._safe_download_ohlcv_best_effort(ticker, start_dt, end_dt)
            if fetched is None or len(fetched) == 0:
                raise ValueError(f"No data available for {ticker} in [{start}, {end}] after download.")
            self._write_parquet(ticker, fetched)
            out = fetched[(fetched["date"] >= start_dt) & (fetched["date"] <= end_dt)].reset_index(drop=True)
            return out

        local_start = local["date"].min()
        local_end = local["date"].max()

        missing = []
        if start_dt < local_start:
            missing.append((start_dt, local_start - pd.Timedelta(days=1)))
        if end_dt > local_end:
            missing.append((local_end + pd.Timedelta(days=1), end_dt))

        merged = local
        for a, b in missing:
            fetched = self._safe_download_ohlcv_best_effort(ticker, a, b)
            if fetched is None or len(fetched) == 0:
                continue
            merged = pd.concat([merged, fetched], axis=0)

        merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if missing:
            self._write_parquet(ticker, merged)

        out = merged[(merged["date"] >= start_dt) & (merged["date"] <= end_dt)].reset_index(drop=True)
        if len(out) == 0:
            raise ValueError(
                f"No data available for {ticker} in [{start}, {end}]. "
                f"Cache covers [{merged['date'].min().date()} -> {merged['date'].max().date()}]."
            )
        return out

    def query(self, sql: str) -> pd.DataFrame:
        if not self.use_duckdb or self._con is None:
            raise RuntimeError("DuckDB is not enabled. Initialize with use_duckdb=True and install duckdb.")
        return self._con.execute(sql).fetchdf()

    @staticmethod
    def adjclose_log_returns(ohlcv: pd.DataFrame) -> pd.Series:
        df = ohlcv.sort_values("date")
        if "Adj Close" in df.columns:
            px = df["Adj Close"].astype(float)
        elif "Close" in df.columns:
            px = df["Close"].astype(float)
        else:
            raise KeyError(f"Need Adj Close or Close. columns={list(df.columns)}")

        r = np.log(px / px.shift(1)).dropna()
        r.index = pd.to_datetime(df["date"].iloc[1:].values)
        r.name = "log_return"
        return r

    @staticmethod
    def _normalize_date_column(df) -> pd.DataFrame:
        if isinstance(df, pd.Series):
            df = df.to_frame()

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"_normalize_date_column expected DataFrame/Series, got {type(df)}")

        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join([str(x) for x in tup if x is not None]).strip() for tup in df.columns.values]

        if "date" not in df.columns and "Date" not in df.columns and "Datetime" not in df.columns:
            df = df.reset_index()

        candidates = ["date", "Date", "Datetime", "index", "timestamp", "time"]

        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break

        if found is None:
            for c in df.columns:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() >= max(5, int(0.5 * len(df))):
                    found = c
                    break

        if found is None:
            raise ValueError(f"Could not identify a date column. Columns={list(df.columns)}")

        if found != "date":
            df = df.rename(columns={found: "date"})

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()].copy()

        return df

def _summarize_parquet_files(root_dir: str) -> pd.DataFrame:
    """Scan root_dir for *_ohlcv.parquet and build a summary table."""
    rows = []
    if not os.path.isdir(root_dir):
        return pd.DataFrame(columns=[
            "file", "ticker", "n_rows", "start", "end", "n_cols", "has_adj_close", "has_close"
        ])

    for fn in sorted(os.listdir(root_dir)):
        if not fn.endswith("_ohlcv.parquet"):
            continue
        path = os.path.join(root_dir, fn)
        ticker = fn.replace("_ohlcv.parquet", "")
        try:
            df = pd.read_parquet(path)
            # Best-effort normalization without needing an instance
            df = MarketDataStore._normalize_date_column(df)
            df = df.sort_values("date")

            n_rows = int(len(df))
            start = df["date"].min() if n_rows else pd.NaT
            end = df["date"].max() if n_rows else pd.NaT
            cols = list(df.columns)
            rows.append({
                "file": fn,
                "ticker": ticker,
                "n_rows": n_rows,
                "start": start,
                "end": end,
                "n_cols": int(len(cols)),
                "has_adj_close": bool("Adj Close" in cols),
                "has_close": bool("Close" in cols),
            })
        except Exception as e:
            rows.append({
                "file": fn,
                "ticker": ticker,
                "n_rows": None,
                "start": pd.NaT,
                "end": pd.NaT,
                "n_cols": None,
                "has_adj_close": None,
                "has_close": None,
                "error": str(e),
            })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(columns=[
            "file", "ticker", "n_rows", "start", "end", "n_cols", "has_adj_close", "has_close"
        ])

    # Stable column order
    cols = ["file", "ticker", "n_rows", "start", "end", "n_cols", "has_adj_close", "has_close"]
    if "error" in out.columns:
        cols.append("error")
    return out[cols]


def _visualize_cache_summary(root_dir: str) -> None:
    """Print and plot basic information about all cached parquet files."""
    import matplotlib.pyplot as plt

    summary = _summarize_parquet_files(root_dir)
    if summary.empty:
        print(f"[cache] No parquet files found in: {root_dir}")
        return

    # Print a concise summary table
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 160)
    print("\n[cache] Parquet cache summary")
    print(summary.to_string(index=False))

    # Plot: row counts by ticker (only those with valid n_rows)
    plot_df = summary.dropna(subset=["n_rows"]).copy()
    if not plot_df.empty:
        plot_df = plot_df.sort_values("n_rows", ascending=False)
        plt.figure(figsize=(10, 4))
        plt.bar(plot_df["ticker"].astype(str), plot_df["n_rows"].astype(int))
        plt.title("Cached OHLCV rows by ticker")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("rows")
        plt.tight_layout()
        plt.show()

    # Plot: coverage length in days
    if "start" in plot_df.columns and "end" in plot_df.columns:
        cov = plot_df.copy()
        cov["coverage_days"] = (pd.to_datetime(cov["end"]) - pd.to_datetime(cov["start"])).dt.days
        plt.figure(figsize=(10, 4))
        plt.bar(cov["ticker"].astype(str), cov["coverage_days"].astype(float))
        plt.title("Cached OHLCV coverage (days) by ticker")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("days")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Visualize basic information stored in all parquet files under root_dir
    store = MarketDataStore(root_dir="./market_data", use_duckdb=True)
    _visualize_cache_summary(store.root_dir)

