"""Append-only JSONL journal: every prompt, raw answer and decision, for later scoring."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class Journal:
    def __init__(self, folder: str | Path | None):
        self.folder = Path(folder) if folder else None

    def write(self, kind: str, **record) -> None:
        if self.folder is None:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        line = json.dumps({'at': now.isoformat(), 'kind': kind, **record}, ensure_ascii=False, default=str)
        # One file per UTC day keeps files small enough to upload as CI artifacts.
        with (self.folder / f'{now:%Y-%m-%d}.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
