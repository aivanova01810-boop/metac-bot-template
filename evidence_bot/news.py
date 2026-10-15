"""AskNews retrieval as structured articles, so the ledger keeps real titles, dates and sources."""
from __future__ import annotations

import asyncio
import os

PLACEHOLDERS = {'', 'REPLACE_ME'}


def _env(name: str) -> str | None:
    value = (os.getenv(name) or '').strip()
    return None if value in PLACEHOLDERS else value


def _rate_limited(exc: Exception) -> bool:
    return type(exc).__name__ == 'RateLimitExceededError' or '429' in str(exc)


def asknews_fetcher(n_articles: int = 10, attempts: int = 4, base_delay: float = 10.0, sleep=asyncio.sleep):
    """Return an async fetcher, or None when no AskNews credentials are configured."""
    api_key, client_id, secret = _env('ASKNEWS_API_KEY'), _env('ASKNEWS_CLIENT_ID'), _env('ASKNEWS_SECRET')
    if not api_key and not (client_id and secret):
        return None
    credentials = {'api_key': api_key} if api_key else {'client_id': client_id, 'client_secret': secret}

    async def search(question):
        from asknews_sdk import AsyncAskNewsSDK
        # Latest-news search costs 1 call of the 1k/month allowance; historical search costs 5.
        async with AsyncAskNewsSDK(**credentials) as sdk:
            return await sdk.news.search_news(query=question.question_text, n_articles=n_articles,
                                              return_type='dicts', method='nl', historical=False)

    async def fetch(question) -> list[dict]:
        # The free tier answers 429 to back-to-back calls; back off 10/20/40 s before giving up.
        for attempt in range(attempts):
            try:
                response = await fetch.search(question)
                break
            except Exception as exc:
                if not _rate_limited(exc) or attempt == attempts - 1:
                    raise
                await sleep(base_delay * 2**attempt)
        return [{'title': a.eng_title or a.title, 'summary': a.summary, 'url': str(a.article_url),
                 'published_at': a.pub_date, 'source': a.domain_url or a.source_id}
                for a in response.as_dicts or []]
    fetch.search = search  # exposed so tests can replace the network call
    return fetch
