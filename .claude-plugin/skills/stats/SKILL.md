---
name: stats
description: Show TokenMizer token savings stats — how many tokens saved today, this week, cache hit rate, and which layers are saving the most. Use when user asks about token usage, costs, savings, or "how much have we saved".
---

Show TokenMizer token savings analytics.

## What to do

```bash
curl -s http://localhost:8000/api/stats
```

Then also fetch cache stats:
```bash
curl -s http://localhost:8000/api/cache/stats
```

## Format the output clearly

Show:
- Tokens saved today and this week
- Cost saved in USD
- Cache hit rate
- Which layer saved the most (compression / cache / windowing / file extraction)
- Suggestions if savings are low

## Example output

```
TokenMizer Stats
─────────────────────────
Today:    12,450 tokens saved (34%) — $0.0373
Week:     87,200 tokens saved (31%) — $0.2616

By layer:
  File extraction:  45,000 tokens
  Semantic cache:   28,000 tokens  
  Smart window:     9,200 tokens
  Compression:      3,800 tokens
  Output trim:      1,200 tokens

Cache: 847 entries | 68% hit rate | 8% full
─────────────────────────
Dashboard: http://localhost:8000
```

## If nothing saved yet

Tell user to start using TokenMizer as their proxy:
```python
from openai import OpenAI
client = OpenAI(
    api_key="your-key",
    base_url="http://localhost:8000/v1"
)
```
