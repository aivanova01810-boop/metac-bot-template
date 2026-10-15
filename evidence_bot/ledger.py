"""Evidence ledger: retrieved articles get stable IDs so every claim can be checked.

The ledger is serialized into the research string itself, so it survives the
ForecastBot plumbing (research -> forecast) and can be parsed back exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlsplit

HEADER = 'EVIDENCE LEDGER (cite items only by their [E#] ID)'
_ITEM = re.compile(r'^\[(E\d+)\] published=(\S+) source=(\S*)$')


@dataclass(frozen=True)
class Evidence:
    id: str
    title: str
    summary: str
    url: str
    source: str
    published_at: datetime | None


def parse_time(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else None
    if not isinstance(value, str) or not value or value == 'unknown':
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    # Naive timestamps are ambiguous; they never count as dated evidence.
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else None


def _one_line(text) -> str:
    return ' '.join(str(text or '').split())


def build(articles: list[dict], limit: int = 15) -> list[Evidence]:
    """articles: dicts with title, summary, url, published_at (datetime or ISO) and optional source."""
    items, seen = [], set()
    for article in articles:
        title, url = _one_line(article.get('title')), _one_line(article.get('url'))
        key = url or title.lower()
        if not title or key in seen:
            continue
        seen.add(key)
        source = _one_line(article.get('source')) or (urlsplit(url).hostname or '').removeprefix('www.')
        items.append(Evidence(f'E{len(items)+1}', title, _one_line(article.get('summary')), url,
                              source, parse_time(article.get('published_at'))))
        if len(items) >= limit:
            break
    return items


def render(items: list[Evidence]) -> str:
    if not items:
        return ''
    lines = [HEADER]
    for e in items:
        stamp = e.published_at.isoformat() if e.published_at else 'unknown'
        lines += [f'[{e.id}] published={stamp} source={e.source}', f'title: {e.title}',
                  f'summary: {e.summary}', f'url: {e.url}', '']
    return '\n'.join(lines).rstrip() + '\n'


def parse(research: str) -> list[Evidence]:
    if HEADER not in (research or ''):
        return []
    items, current = [], None
    for line in research.split(HEADER, 1)[1].splitlines():
        match = _ITEM.match(line.strip())
        if match:
            current = {'id': match[1], 'published_at': parse_time(match[2]), 'source': match[3],
                       'title': '', 'summary': '', 'url': ''}
            items.append(current)
        elif current is not None:
            for field in ('title', 'summary', 'url'):
                if line.startswith(field + ': '):
                    current[field] = line[len(field) + 2:]
    return [Evidence(**item) for item in items]
