"""Financial market data tool backend.

Aggregates public market data for the stock market prediction use case.

Free data sources (no API key required)
-----------------------------------------
Yahoo Finance (yfinance)  — price history, fundamentals, earnings calendar
                            pip install yfinance
Alpha Vantage             — technical indicators (free tier: 25 req/day)
                            https://www.alphavantage.co/support/#api-key
FRED (Federal Reserve)    — macro data (free, no key required)
                            https://fred.stlouisfed.org/docs/api/fred/
NewsAPI                   — news sentiment (free 100 req/day)
                            https://newsapi.org/register

Environment variables
---------------------
ALPHA_VANTAGE_API_KEY — Alpha Vantage free API key
NEWSAPI_KEY           — NewsAPI free key
FRED_API_KEY          — optional FRED key (public endpoints work without it)

When no keys are present the backend returns realistic mock data so the
agent pipeline always has something to work with in offline/test mode.

Usage
-----
.. code-block:: python

    from src.mcp.financial_data_backend import FinancialDataBackend

    backend = FinancialDataBackend()
    results = backend.search("NVDA earnings momentum technical")
    for r in results:
        print(r.title, r.snippet[:80])

    # With live data (requires yfinance)
    backend = FinancialDataBackend(use_live_data=True)
    results = backend.search("AAPL price action RSI MACD", ticker="AAPL")
"""

from __future__ import annotations

import os
from typing import Any

from src.models import SearchResult


class FinancialDataBackend:
    """Aggregates price, fundamental, sentiment, and macro signals.

    Parameters
    ----------
    use_live_data:
        When ``True``, tries to fetch live data via yfinance and public APIs.
        Falls back to mock data on any failure. Default: ``False``.
    alpha_vantage_key:
        Alpha Vantage API key for technical indicators.
        Falls back to ``ALPHA_VANTAGE_API_KEY`` env var.
    newsapi_key:
        NewsAPI key for news sentiment.
        Falls back to ``NEWSAPI_KEY`` env var.
    """

    def __init__(
        self,
        use_live_data: bool = False,
        alpha_vantage_key: str | None = None,
        newsapi_key: str | None = None,
    ) -> None:
        self._live = use_live_data
        self._av_key = alpha_vantage_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self._news_key = newsapi_key or os.environ.get("NEWSAPI_KEY", "")

    # ------------------------------------------------------------------
    # ToolBackend protocol
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        num_results: int = 6,
        ticker: str = "",
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Return market data signals as SearchResult snippets.

        Parameters
        ----------
        query:
            Natural-language query, e.g. "NVDA earnings momentum technical".
        num_results:
            Maximum number of signal snippets to return.
        ticker:
            Optional explicit ticker to fetch data for (e.g. ``"NVDA"``).
            When omitted the ticker is extracted from *query* heuristically.
        """
        if not ticker:
            ticker = self._extract_ticker(query)

        if self._live:
            return self._live_signals(ticker, query, num_results)
        return self._mock_signals(ticker, query, num_results)

    # ------------------------------------------------------------------
    # Live data (requires yfinance + optional API keys)
    # ------------------------------------------------------------------

    def _live_signals(
        self, ticker: str, query: str, num_results: int
    ) -> list[SearchResult]:  # pragma: no cover
        """Fetch live signals from yfinance and public APIs."""
        results: list[SearchResult] = []
        try:
            import yfinance as yf  # type: ignore[import]

            t = yf.Ticker(ticker)
            info = t.info or {}
            hist = t.history(period="3mo")

            # Price / technical summary
            price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            fifty_two_wk_high = info.get("fiftyTwoWeekHigh", 0)
            fifty_two_wk_low = info.get("fiftyTwoWeekLow", 0)
            ma50 = info.get("fiftyDayAverage", 0)
            ma200 = info.get("twoHundredDayAverage", 0)

            pct_from_52h = (
                round((price - fifty_two_wk_high) / fifty_two_wk_high * 100, 1)
                if fifty_two_wk_high else 0
            )
            vs_ma50 = (
                "above" if price > ma50 else "below"
            ) if ma50 else "N/A"

            results.append(SearchResult(
                title=f"{ticker} — Price & Technical Summary",
                url=f"https://finance.yahoo.com/quote/{ticker}",
                snippet=(
                    f"Price: ${price:.2f} | 52w High: ${fifty_two_wk_high:.2f} "
                    f"({pct_from_52h:+.1f}%) | 52w Low: ${fifty_two_wk_low:.2f} | "
                    f"50-DMA: ${ma50:.2f} ({vs_ma50}) | 200-DMA: ${ma200:.2f}"
                ),
                relevance_score=1.0,
            ))

            # Fundamental summary
            pe = info.get("trailingPE", "N/A")
            fwd_pe = info.get("forwardPE", "N/A")
            ps = info.get("priceToSalesTrailing12Months", "N/A")
            market_cap = info.get("marketCap", 0)
            cap_str = f"${market_cap/1e9:.1f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.0f}M"
            revenue_growth = info.get("revenueGrowth", "N/A")
            gross_margin = info.get("grossMargins", "N/A")
            results.append(SearchResult(
                title=f"{ticker} — Fundamentals",
                url=f"https://finance.yahoo.com/quote/{ticker}/financials",
                snippet=(
                    f"Market Cap: {cap_str} | P/E: {pe} | Fwd P/E: {fwd_pe} | "
                    f"P/S: {ps} | Revenue Growth: {revenue_growth} | "
                    f"Gross Margin: {gross_margin}"
                ),
                relevance_score=0.95,
            ))

            # Volume / momentum from history
            if not hist.empty and len(hist) >= 20:
                avg_vol = hist["Volume"].tail(20).mean()
                last_vol = hist["Volume"].iloc[-1]
                vol_ratio = last_vol / avg_vol if avg_vol else 1.0
                close_series = hist["Close"]
                ret_1m = (close_series.iloc[-1] / close_series.iloc[-22] - 1) * 100 if len(close_series) >= 22 else 0
                ret_3m = (close_series.iloc[-1] / close_series.iloc[0] - 1) * 100
                results.append(SearchResult(
                    title=f"{ticker} — Momentum & Volume",
                    url=f"https://finance.yahoo.com/quote/{ticker}",
                    snippet=(
                        f"1M Return: {ret_1m:+.1f}% | 3M Return: {ret_3m:+.1f}% | "
                        f"Last Volume: {last_vol:,.0f} ({vol_ratio:.1f}x avg) | "
                        f"Avg 20-day Vol: {avg_vol:,.0f}"
                    ),
                    relevance_score=0.90,
                ))

            # Earnings info
            next_eps = info.get("forwardEps", "N/A")
            earnings_date = str(info.get("earningsTimestamp", "Unknown"))
            results.append(SearchResult(
                title=f"{ticker} — Earnings",
                url=f"https://finance.yahoo.com/quote/{ticker}/analysis",
                snippet=(
                    f"Forward EPS: {next_eps} | "
                    f"Next Earnings: {earnings_date} | "
                    f"Analyst Target: ${info.get('targetMeanPrice', 'N/A')}"
                ),
                relevance_score=0.85,
            ))

        except Exception as exc:
            results.append(SearchResult(
                title=f"{ticker} — Live data unavailable",
                url="",
                snippet=f"yfinance error: {exc}. Using mock signals.",
                relevance_score=0.0,
            ))
            results.extend(self._mock_signals(ticker, query, num_results - len(results)))

        return results[:num_results]

    # ------------------------------------------------------------------
    # Mock signals (always available, no dependencies)
    # ------------------------------------------------------------------

    def _mock_signals(
        self, ticker: str, query: str, num_results: int
    ) -> list[SearchResult]:
        """Return realistic-looking mock market signals for offline testing."""
        t = ticker.upper() or "TICKER"
        return [
            SearchResult(
                title=f"[mock] {t} — Technical Analysis",
                url=f"https://finance.yahoo.com/quote/{t}",
                snippet=(
                    f"{t}: RSI(14)=58 (neutral, not overbought). "
                    f"Price above 50-DMA (+4.2%) and 200-DMA (+12.1%). "
                    f"MACD bullish crossover 3 days ago. "
                    f"Bollinger Band: mid-band, room to run. "
                    f"Volume: 1.3x 20-day avg (accumulation pattern)."
                ),
                relevance_score=1.0,
            ),
            SearchResult(
                title=f"[mock] {t} — Fundamental Snapshot",
                url=f"https://finance.yahoo.com/quote/{t}/financials",
                snippet=(
                    f"{t}: Trailing P/E=28x (sector avg 32x — slight discount). "
                    f"Forward P/E=22x (compression expected). "
                    f"Revenue growth +18% YoY. Gross margin 64% (+200bps YoY). "
                    f"FCF yield 3.8%. Net debt/EBITDA=0.4x (low leverage)."
                ),
                relevance_score=0.95,
            ),
            SearchResult(
                title=f"[mock] {t} — Sentiment & Positioning",
                url=f"https://finance.yahoo.com/quote/{t}",
                snippet=(
                    f"{t}: Short interest 3.2% of float (low, not a squeeze candidate). "
                    f"IV Rank=42 (moderate — options not pricing extreme move). "
                    f"Put/Call ratio=0.72 (mildly bullish skew). "
                    f"Insider buys: 2 Form-4 buys last 30 days ($250K combined). "
                    f"News sentiment: positive (earnings beat expectations last quarter)."
                ),
                relevance_score=0.90,
            ),
            SearchResult(
                title=f"[mock] {t} — Macro & Sector Context",
                url="https://fred.stlouisfed.org",
                snippet=(
                    f"{t} sector: Technology. "
                    f"Sector RS vs S&P500: outperforming (+3.2% last month). "
                    f"10Y yield at 4.35% — growth stock multiple pressure moderate. "
                    f"USD index flat — neutral for international revenue. "
                    f"Fed: 1 cut priced in next 6 months (dovish tilt supports tech)."
                ),
                relevance_score=0.85,
            ),
            SearchResult(
                title=f"[mock] {t} — Upcoming Catalysts",
                url=f"https://finance.yahoo.com/quote/{t}/analysis",
                snippet=(
                    f"{t} catalysts: Q2 earnings in ~6 weeks (consensus EPS +15% YoY). "
                    f"New product launch announcement expected next month. "
                    f"Annual developer conference in 8 weeks (sentiment event). "
                    f"Index rebalancing: potential inclusion in next review cycle."
                ),
                relevance_score=0.80,
            ),
            SearchResult(
                title=f"[mock] {t} — Risk Factors",
                url="",
                snippet=(
                    f"{t} risks: Regulatory scrutiny (antitrust review ongoing). "
                    f"Customer concentration: top 3 customers = 38% revenue. "
                    f"Supply chain: single-source critical component (6-month risk). "
                    f"Valuation: premium to peers — de-rating risk if growth slows. "
                    f"Macro: elevated rates could compress multiple by 15-20%."
                ),
                relevance_score=0.75,
            ),
        ][:num_results]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ticker(query: str) -> str:
        """Heuristically extract a ticker from a query string."""
        import re
        # Look for 1–5 uppercase letters (common ticker pattern)
        tokens = re.findall(r"\b[A-Z]{1,5}\b", query)
        # Filter out common English words
        stopwords = {"A", "I", "AI", "THE", "AND", "OR", "FOR", "RSI", "PE", "EPS", "FCF", "YOY"}
        tickers = [t for t in tokens if t not in stopwords]
        return tickers[0] if tickers else "SPY"
