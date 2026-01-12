from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import requests


@dataclass(frozen=True)
class FMPNewsItem:
    symbol: str
    publishedDate: str
    publisher: str
    title: str
    image: str
    site: str
    text: str
    url: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FMPNewsItem":
        # Defensive defaults to avoid KeyError if FMP adds/removes fields.
        return FMPNewsItem(
            symbol=str(d.get("symbol", "")).strip(),
            publishedDate=str(d.get("publishedDate", "")).strip(),
            publisher=str(d.get("publisher", "")).strip(),
            title=str(d.get("title", "")).strip(),
            image=str(d.get("image", "")).strip(),
            site=str(d.get("site", "")).strip(),
            text=str(d.get("text", "")).strip(),
            url=str(d.get("url", "")).strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "publishedDate": self.publishedDate,
            "publisher": self.publisher,
            "title": self.title,
            "image": self.image,
            "site": self.site,
            "text": self.text,
            "url": self.url,
        }


class FMPStockNewsFetcher:
    """
    Fetches and locally stores stock news using FinancialModelingPrep's endpoint:
      https://financialmodelingprep.com/stable/news/stock?symbols=AAPL&apikey=...

    Storage format:
      - JSON Lines (jsonl) per symbol: <root_dir>/<SYMBOL>.jsonl
      - A lightweight per-symbol index (url -> last seen) in <root_dir>/<SYMBOL>_index.json

    Deduplication:
      - Uses article "url" as primary key (most stable).
      - If url is missing, falls back to (title, publishedDate, publisher).

    Notes:
      - Handles rate limits / transient errors with retries + backoff.
      - Keeps raw items as returned by FMP but normalizes to the schema you provided.
    """

    BASE_URL = "https://financialmodelingprep.com/stable/news/stock"

    def __init__(
        self,
        api_key: Optional[str] = None,
        root_dir: Union[str, Path] = "./fmp_stock_news",
        session: Optional[requests.Session] = None,
        timeout: int = 30,
        max_retries: int = 5,
        backoff_seconds: float = 2.0,
        user_agent: str = "FMPStockNewsFetcher/1.0",
    ):
        # Prefer an explicitly provided api_key; else fall back to env vars.
        # (You mentioned having a system variable named `secret_key`; this will pick it up if exported.)
        self.api_key = api_key or os.getenv("FMP_API_KEY") or os.getenv("secret_key")
        if not self.api_key:
            raise ValueError(
                "Missing API key. Provide api_key=... or set env var FMP_API_KEY (or secret_key)."
            )

        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    # -------------------------
    # Public API
    # -------------------------
    def fetch(
        self,
        symbols: Union[str, Sequence[str]],
        limit: Optional[int] = None,
        page: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[FMPNewsItem]:
        """
        Fetch latest news for one or multiple symbols.
        Returns a flat list of FMPNewsItem.

        Optional parameters (if supported by your plan / endpoint):
          - limit, page, from_date, to_date
        If the endpoint ignores them, they simply won’t affect the response.
        """
        sym_str = self._normalize_symbols(symbols)
        params: Dict[str, Any] = {"symbols": sym_str, "apikey": self.api_key}
        if limit is not None:
            params["limit"] = int(limit)
        if page is not None:
            params["page"] = int(page)
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        data = self._get_json(self.BASE_URL, params=params)

        # FMP typically returns a JSON array for this endpoint.
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected response type from FMP: {type(data)}; body={data}")

        items = [FMPNewsItem.from_dict(x) for x in data if isinstance(x, dict)]
        # If fetching multiple symbols, some items may have blank symbol; keep them but you can filter.
        return items

    def update_local(
        self,
        symbols: Union[str, Sequence[str]],
        limit: Optional[int] = None,
        page: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        sleep_seconds: float = 0.0,
    ) -> Dict[str, int]:
        """
        Fetches news for given symbols and appends only new items to local storage.
        Returns: dict {symbol: number_of_new_items_written}
        """
        symbol_list = self._symbols_list(symbols)
        results: Dict[str, int] = {}

        for i, sym in enumerate(symbol_list):
            items = self.fetch(
                sym,
                limit=limit,
                page=page,
                from_date=from_date,
                to_date=to_date,
            )
            new_count = self._append_new_items(sym, items)
            results[sym] = new_count

            # Optional politeness delay if you're looping many symbols.
            if sleep_seconds > 0 and i < len(symbol_list) - 1:
                time.sleep(sleep_seconds)

        return results

    def load_local(
        self,
        symbol: str,
        since: Optional[str] = None,
    ) -> List[FMPNewsItem]:
        """
        Load locally saved items for a symbol. If `since` is provided,
        it filters items with publishedDate >= since.

        `since` format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' (assumed UTC naive).
        """
        path = self._symbol_path(symbol)
        if not path.exists():
            return []

        items: List[FMPNewsItem] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if isinstance(d, dict):
                        items.append(FMPNewsItem.from_dict(d))
                except json.JSONDecodeError:
                    continue

        if since:
            cutoff = self._parse_dt(since)
            items = [it for it in items if self._parse_dt(it.publishedDate) >= cutoff]

        # Sort newest first (optional but helpful)
        items.sort(key=lambda x: self._parse_dt(x.publishedDate), reverse=True)
        return items

    # -------------------------
    # Internal helpers
    # -------------------------
    def _get_json(self, url: str, params: Dict[str, Any]) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)

                # Handle common throttling / transient failures.
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(
                        f"HTTP {resp.status_code} (retryable): {resp.text[:300]}",
                        response=resp,
                    )

                resp.raise_for_status()

                # Some APIs return empty string; treat as empty list.
                if not resp.text.strip():
                    return []

                return resp.json()
            except (requests.RequestException, ValueError) as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                # Exponential-ish backoff with cap
                sleep_for = min(self.backoff_seconds * (2 ** (attempt - 1)), 60.0)
                time.sleep(sleep_for)

        raise RuntimeError(f"Failed to fetch JSON from {url}. Last error: {last_err}")

    def _append_new_items(self, symbol: str, items: List[FMPNewsItem]) -> int:
        # Load index for dedup
        index_path = self._symbol_index_path(symbol)
        index = self._load_index(index_path)

        out_path = self._symbol_path(symbol)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        new_items: List[FMPNewsItem] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for it in items:
            # If FMP returns mixed symbols when requesting multiple tickers, keep only relevant ones.
            if it.symbol and it.symbol.upper() != symbol.upper():
                continue

            key = self._dedup_key(it)
            if key in index:
                continue

            # Write and update index
            new_items.append(it)
            index[key] = {"first_seen_utc": now_iso}

        if not new_items:
            return 0

        # Append JSONL
        with out_path.open("a", encoding="utf-8") as f:
            for it in new_items:
                f.write(json.dumps(it.to_dict(), ensure_ascii=False))
                f.write("\n")

        self._save_index(index_path, index)
        return len(new_items)

    @staticmethod
    def _dedup_key(item: FMPNewsItem) -> str:
        if item.url:
            return f"url::{item.url}"
        # Fallback if url is missing
        return f"fallback::{item.title}::{item.publishedDate}::{item.publisher}"

    def _symbol_path(self, symbol: str) -> Path:
        return self.root_dir / f"{symbol.upper()}.jsonl"

    def _symbol_index_path(self, symbol: str) -> Path:
        return self.root_dir / f"{symbol.upper()}_index.json"

    @staticmethod
    def _load_index(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _save_index(path: Path, index: Dict[str, Any]) -> None:
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    @staticmethod
    def _normalize_symbols(symbols: Union[str, Sequence[str]]) -> str:
        if isinstance(symbols, str):
            s = symbols.strip()
            # allow comma-separated input
            parts = [x.strip().upper() for x in s.split(",") if x.strip()]
            return ",".join(parts)
        return ",".join([str(x).strip().upper() for x in symbols if str(x).strip()])

    @staticmethod
    def _symbols_list(symbols: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(symbols, str):
            parts = [x.strip().upper() for x in symbols.split(",") if x.strip()]
            return parts
        return [str(x).strip().upper() for x in symbols if str(x).strip()]

    @staticmethod
    def _parse_dt(s: str) -> datetime:
        """
        FMP publishedDate example: "2025-02-03 21:05:14"
        Treat as naive UTC unless you have evidence it's local exchange time.
        """
        s = (s or "").strip()
        if not s:
            return datetime.min.replace(tzinfo=timezone.utc)

        # Common formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        # ISO fallback
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    # Example usage:
    # 1) Export your key (recommended):
    #    - Windows PowerShell:  setx FMP_API_KEY "YOUR_KEY"
    #    - Linux/macOS:         export FMP_API_KEY="YOUR_KEY"
    #
    # 2) Run this file and it will fetch and save AAPL news.

    fetcher = FMPStockNewsFetcher(root_dir="./fmp_stock_news")

    # Fetch once (in-memory)
    latest = fetcher.fetch("AAPL", limit=50)
    print(f"Fetched {len(latest)} items from API. Example:")
    if latest:
        print(json.dumps(latest[0].to_dict(), indent=2, ensure_ascii=False))

    # Update local storage (deduplicated)
    added = fetcher.update_local(["AAPL", "MSFT"], limit=50, sleep_seconds=0.5)
    print("New items written:", added)

    # Load from disk
    aapl_local = fetcher.load_local("AAPL", since="2025-01-01")
    print(f"Loaded {len(aapl_local)} local AAPL items since 2025-01-01.")
