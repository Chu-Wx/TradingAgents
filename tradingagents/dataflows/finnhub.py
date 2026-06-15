"""Finnhub-based news and analyst recommendation data fetching.

Free tier (60 req/min) covers company news, analyst recommendations,
and quote data. Social sentiment is gated behind a paid plan (403).

API key is read from the ``FINNHUB_API_KEY`` environment variable.
"""

import os
from datetime import datetime
from typing import Optional

import requests

from .config import get_config

_BASE = "https://finnhub.io/api/v1"


def _api_key() -> str:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise ValueError("FINNHUB_API_KEY environment variable is not set")
    return key


def _format_date(d: str) -> str:
    """Normalise a YYYY-MM-DD date string (pass-through; Finnhub uses same format)."""
    return d


def get_news(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Retrieve company news for a ticker via Finnhub.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles
    """
    config = get_config()
    article_limit = config["news_article_limit"]

    try:
        params = {
            "symbol": ticker.upper(),
            "from": start_date,
            "to": end_date,
            "token": _api_key(),
        }
        resp = requests.get(f"{_BASE}/company-news", params=params, timeout=15)
        resp.raise_for_status()
        articles = resp.json()

        if not articles:
            return f"No news found for {ticker} between {start_date} and {end_date}"

        news_str = ""
        for article in articles[:article_limit]:
            headline = article.get("headline", "No title")
            source = article.get("source", "Unknown")
            summary = article.get("summary", "")
            url = article.get("url", "")
            # Finnhub returns epoch seconds
            ts = article.get("datetime")
            pub_date = (
                datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                if ts
                else "unknown date"
            )

            news_str += f"### {headline} (source: {source}, {pub_date})\n"
            if summary:
                news_str += f"{summary}\n"
            if url:
                news_str += f"Link: {url}\n"
            news_str += "\n"

        header = f"## {ticker} News (Finnhub), from {start_date} to {end_date}:\n\n"
        return header + news_str

    except Exception as e:
        return f"Error fetching Finnhub news for {ticker}: {str(e)}"


def get_analyst_recommendations(
    ticker: str,
) -> str:
    """Retrieve analyst recommendation trends for a ticker via Finnhub.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL")

    Returns:
        Formatted string containing analyst consensus data
    """
    try:
        params = {"symbol": ticker.upper(), "token": _api_key()}
        resp = requests.get(
            f"{_BASE}/stock/recommendation", params=params, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return f"No analyst recommendations found for {ticker}"

        # Use the most recent period
        latest = data[0]
        period = latest.get("period", "unknown")
        strong_buy = latest.get("strongBuy", 0)
        buy = latest.get("buy", 0)
        hold = latest.get("hold", 0)
        sell = latest.get("sell", 0)
        strong_sell = latest.get("strongSell", 0)
        total = strong_buy + buy + hold + sell + strong_sell

        lines = [
            f"# Analyst Recommendations for {ticker.upper()} (period: {period})",
            "",
            f"| Rating | Count |",
            f"|--------|-------|",
            f"| Strong Buy | {strong_buy} |",
            f"| Buy | {buy} |",
            f"| Hold | {hold} |",
            f"| Sell | {sell} |",
            f"| Strong Sell | {strong_sell} |",
            f"| **Total Analysts** | **{total}** |",
        ]

        # Price target (separate endpoint)
        try:
            params2 = {"symbol": ticker.upper(), "token": _api_key()}
            resp2 = requests.get(
                f"{_BASE}/stock/price-target", params=params2, timeout=15
            )
            resp2.raise_for_status()
            pt_data = resp2.json()
            if pt_data:
                lines.append("")
                lines.append(
                    f"**Consensus Price Target:** ${pt_data.get('targetMean', 'N/A')} "
                    f"(High: ${pt_data.get('targetHigh', 'N/A')}, "
                    f"Low: ${pt_data.get('targetLow', 'N/A')}, "
                    f"Median: ${pt_data.get('targetMedian', 'N/A')})"
                )
        except Exception:
            pass  # price target is optional enrichment; don't fail on it

        return "\n".join(lines)

    except Exception as e:
        return f"Error fetching analyst recommendations for {ticker}: {str(e)}"
