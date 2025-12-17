from __future__ import annotations

import os
import numpy as np
import pandas as pd
import yfinance as yf
import duckdb  # optional


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

    def clean_parquet_with_duckdb(self, ticker: str) -> bool:
        """
        Clean a single ticker's cached parquet (LONG format):
        - bfill + linear interpolation on numeric columns (excluding 'date')
        - write back to same parquet (DuckDB if enabled)
        """
        path = self._parquet_path(ticker)
        if not os.path.exists(path):
            return False

        # Read
        if self.use_duckdb and self._con is not None:
            df = self._con.execute("SELECT * FROM read_parquet(?)", [path]).fetchdf()
        else:
            df = pd.read_parquet(path)

        if df.empty:
            print(f"[clean] empty parquet for {ticker}")
            return False

        df = self._normalize_date_column(df)
        df = (
            df.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

        num_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            df[num_cols] = df[num_cols].bfill()
            df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")

            nan_counts = df[num_cols].isna().sum()
            if (nan_counts > 0).any():
                print(f"[clean] Warning: NaNs remain after cleaning {ticker}: {nan_counts[nan_counts > 0].to_dict()}")

        # Write (safe overwrite)
        if self.use_duckdb and self._con is not None:
            tmp_path = path + ".tmp"
            self._con.register("cleaned_df", df)
            self._con.execute("COPY cleaned_df TO ? (FORMAT PARQUET)", [tmp_path])
            self._con.unregister("cleaned_df")
            os.replace(tmp_path, path)
        else:
            df.to_parquet(path, index=False)

        return True

    def _build_wide_price_matrix(
            self,
            tickers: list[str],
            price_col: str = "Adj Close",
    ) -> pd.DataFrame:
        """
        Build a WIDE matrix: columns ['date', <ticker1>, <ticker2>, ...]
        using each ticker's cached LONG parquet.
        Prefers DuckDB; falls back to pandas merges.

        price_col: 'Adj Close' preferred; if absent for a ticker, will fall back to 'Close' if present.
        """
        existing = [t for t in tickers if os.path.exists(self._parquet_path(t))]
        missing = [t for t in tickers if t not in existing]
        if missing:
            print(
                f"[wide] Missing parquet for {len(missing)} tickers: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if not existing:
            raise FileNotFoundError("[wide] No ticker parquets found.")

        if self.use_duckdb and self._con is not None:
            # Build a FULL OUTER JOIN chain in DuckDB. For each ticker, pick price_col else Close.
            # Note: we alias the chosen price column to the ticker symbol.
            pieces = []
            for i, t in enumerate(existing):
                p = self._parquet_path(t)
                # COALESCE handles price_col missing values; but if the column itself doesn't exist,
                # DuckDB would error. We therefore select both if present by reading and checking schema quickly.
                # Lightweight schema check:
                cols = set(
                    self._con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [p]).fetchdf()["column_name"].tolist())
                use_col = price_col if price_col in cols else ("Close" if "Close" in cols else None)
                if use_col is None:
                    print(f"[wide] Warning: {t} has neither {price_col} nor Close; skipped.")
                    continue
                pieces.append((t, p, use_col))

            if not pieces:
                raise ValueError("[wide] No usable tickers (no price columns).")

            # Start with first ticker
            t0, p0, c0 = pieces[0]
            sql = f"""
                WITH t0 AS (
                    SELECT CAST(date AS TIMESTAMP) AS date, "{c0}" AS "{t0}"
                    FROM read_parquet('{p0.replace("'", "''")}')
                )
            """
            from_clause = "SELECT * FROM t0\n"
            for idx, (t, p, c) in enumerate(pieces[1:], start=1):
                alias = f"t{idx}"
                sql += f""",
                {alias} AS (
                    SELECT CAST(date AS TIMESTAMP) AS date, "{c}" AS "{t}"
                    FROM read_parquet('{p.replace("'", "''")}')
                )
                """
                from_clause = f"""
                SELECT
                    COALESCE(w.date, {alias}.date) AS date,
                    w.* EXCLUDE(date),
                    {alias}."{t}"
                FROM ({from_clause.strip()}) w
                FULL OUTER JOIN {alias}
                ON w.date = {alias}.date
                """

            wide = self._con.execute(sql + "\n" + from_clause).fetchdf()
            wide = self._normalize_date_column(wide)
            wide = wide.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
            return wide

        # ---- pandas fallback ----
        wide = None
        for t in existing:
            df = self._read_parquet(t)
            if df is None or df.empty:
                continue
            col = price_col if price_col in df.columns else ("Close" if "Close" in df.columns else None)
            if col is None:
                print(f"[wide] Warning: {t} has neither {price_col} nor Close; skipped.")
                continue
            tmp = df[["date", col]].rename(columns={col: t})
            wide = tmp if wide is None else wide.merge(tmp, on="date", how="outer")

        if wide is None or wide.empty:
            raise ValueError("[wide] Wide matrix could not be constructed.")
        wide = self._normalize_date_column(wide)
        wide = wide.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        return wide

    def _write_df_parquet(self, out_path: str, df: pd.DataFrame) -> None:
        """Utility: write parquet via DuckDB if enabled, else pandas."""
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        if self.use_duckdb and self._con is not None:
            tmp_path = out_path + ".tmp"
            self._con.register("out_df", df)
            self._con.execute("COPY out_df TO ? (FORMAT PARQUET)", [tmp_path])
            self._con.unregister("out_df")
            os.replace(tmp_path, out_path)
        else:
            df.to_parquet(out_path, index=False)

    def _normalize_and_log_from_wide(self, wide_df: pd.DataFrame) -> tuple[
        pd.DataFrame, pd.DataFrame, list[str], list[str]]:
        """
        Given wide df: ['date', tickers...], produce normalized and log-evolution.
        Returns (normalized_df, log_df, all_nan_norm, all_nan_log).
        """
        if wide_df.empty:
            raise ValueError("Wide dataframe is empty.")
        if "date" not in wide_df.columns:
            raise ValueError("Expected 'date' column in wide dataframe.")

        tickers = [c for c in wide_df.columns if c != "date" and pd.api.types.is_numeric_dtype(wide_df[c])]
        if not tickers:
            raise ValueError("No numeric ticker columns in wide dataframe.")

        normalized_df = wide_df[["date"]].copy()
        log_df = wide_df[["date"]].copy()

        all_nan_norm, all_nan_log = [], []

        for t in tickers:
            s = wide_df[t]
            first_valid = s.dropna().iloc[0] if s.dropna().size > 0 else None

            if first_valid is not None and first_valid > 0:
                ratio = s / first_valid
                normalized_df[t] = ratio
                log_df[t] = np.log(ratio)
            else:
                normalized_df[t] = np.nan
                log_df[t] = np.nan

            if normalized_df[t].isna().all():
                all_nan_norm.append(t)
            if log_df[t].isna().all():
                all_nan_log.append(t)

        return normalized_df, log_df, all_nan_norm, all_nan_log

    def clean_then_build_wide_then_normalize_and_log(
            self,
            tickers: list[str],
            wide_out_path: str,
            normalized_out_path: str,
            log_out_path: str,
            price_col: str = "Adj Close",
            clean_first: bool = True,
    ) -> dict:
        """
        Orchestrator:
          1) Clean each ticker parquet (optional)
          2) Build wide price matrix across tickers
          3) Output:
             - wide parquet
             - normalized parquet
             - log-evolution parquet

        Returns a summary dict for logging/QA.
        """
        if not tickers:
            raise ValueError("tickers list is empty.")

        cleaned_ok = []
        cleaned_missing = []
        if clean_first:
            for t in tickers:
                ok = self.clean_parquet_with_duckdb(t)
                (cleaned_ok if ok else cleaned_missing).append(t)
            print(f"[run] Cleaned {len(cleaned_ok)}/{len(tickers)} tickers. Missing/failed: {len(cleaned_missing)}")

        wide = self._build_wide_price_matrix(tickers, price_col=price_col)
        self._write_df_parquet(wide_out_path, wide)

        normalized_df, log_df, all_nan_norm, all_nan_log = self._normalize_and_log_from_wide(wide)
        self._write_df_parquet(normalized_out_path, normalized_df)
        self._write_df_parquet(log_out_path, log_df)

        if all_nan_norm:
            print(
                f"[run] Warning: all-NaN normalized tickers (n={len(all_nan_norm)}): {all_nan_norm[:10]}{'...' if len(all_nan_norm) > 10 else ''}")
        if all_nan_log:
            print(
                f"[run] Warning: all-NaN log tickers (n={len(all_nan_log)}): {all_nan_log[:10]}{'...' if len(all_nan_log) > 10 else ''}")

        summary = {
            "tickers_requested": len(tickers),
            "clean_first": clean_first,
            "cleaned_ok": len(cleaned_ok),
            "cleaned_missing_or_failed": len(cleaned_missing),
            "wide_rows": int(len(wide)),
            "wide_cols": int(len(wide.columns)),
            "all_nan_normalized": all_nan_norm,
            "all_nan_log": all_nan_log,
            "wide_out_path": wide_out_path,
            "normalized_out_path": normalized_out_path,
            "log_out_path": log_out_path,
            "price_col_used_preference": price_col,
        }
        print(f"[run] Wrote wide={wide_out_path}, normalized={normalized_out_path}, log={log_out_path}")
        return summary


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
    store = MarketDataStore(root_dir="../market_data", use_duckdb=True)
    _visualize_cache_summary(store.root_dir)
