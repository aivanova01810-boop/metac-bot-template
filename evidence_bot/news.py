"""AskNews retrieval as structured articles, so the ledger keeps real titles, dates and sources."""
from __future__ import annotations

import os

PLACEHOLDERS = {'', 'REPLACE_ME'}


def _env(name: str) -> str | None:
    value = (os.getenv(name) or '').strip()
    return None if value in PLACEHOLDERS else value


def asknews_fetcher(n_articles: int = 10):
    """Return an async fetcher, or None when no AskNews credentials are configured."""
    api_key, client_id, secret = _env('ASKNEWS_API_KEY'), _env('ASKNEWS_CLIENT_ID'), _env('ASKNEWS_SECRET')
    if not api_key and not (client_id and secret):
        return None
    credentials = {'api_key': api_key} if api_key else {'client_id': client_id, 'client_secret': secret}

    async def fetch(question) -> list[dict]:
        from asknews_sdk import AsyncAskNewsSDK
        # Latest-news search costs 1 call of the 1k/month allowance; historical search costs 5.
        async with AsyncAskNewsSDK(**credentials) as sdk:
            response = await sdk.news.search_news(query=question.question_text, n_articles=n_articles,
                                                  return_type='dicts', method='nl', historical=False)
        return [{'title': a.eng_title or a.title, 'summary': a.summary, 'url': str(a.article_url),
                 'published_at': a.pub_date, 'source': a.domain_url or a.source_id}
                for a in response.as_dicts or []]
    return fetch
