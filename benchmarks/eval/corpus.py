"""
Labelled corpus for the extraction eval harness.

CORPUS FORMAT
-------------
One JSON file per session, in `benchmarks/eval/corpus/`:

```json
{
  "id": "fastapi_auth",
  "origin": "synthetic",
  "domain": "backend/python",
  "notes": "free text — what this session is meant to stress",
  "messages": [{"role": "user", "content": "..."}],
  "ground_truth": {
    "completed_tasks": ["..."],
    "pending_tasks":   ["..."],
    "decisions":       ["..."],
    "files":           ["..."],
    "errors":          ["..."]
  }
}
```

`origin` is required and must be `"synthetic"` or `"real"`. The harness
reports the split, because "89% recall" means something different on
hand-written fixtures than on captured transcripts, and a corpus that
does not distinguish them will eventually be quoted as though it did.

RUNNING IT ON YOUR OWN TRANSCRIPTS
----------------------------------
Point the harness at any directory of files in the format above:

    python -m benchmarks.eval --corpus /path/to/my/sessions

The committed corpus is entirely synthetic. Numbers from it are
directional. If you want a number that describes YOUR workload, label a
few of your own sessions and run against those — that is the only way to
get one, and it is why the loader takes a path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"

VALID_ORIGINS = ("synthetic", "real")
CATEGORIES = ("completed_tasks", "pending_tasks", "decisions", "files", "errors")


class CorpusError(ValueError):
    """A corpus file is malformed. Raised loudly rather than skipped —
    a silently-ignored session would quietly change every score."""


@dataclass
class Session:
    id: str
    origin: str
    domain: str
    messages: list[dict]
    ground_truth: dict
    notes: str = ""
    path: Path | None = field(default=None, repr=False)

    @property
    def turns(self) -> int:
        return len(self.messages)

    def expected(self, category: str) -> list[str]:
        return list(self.ground_truth.get(category, []))


def _validate(raw: dict, path: Path) -> Session:
    for key in ("id", "origin", "messages", "ground_truth"):
        if key not in raw:
            raise CorpusError(f"{path.name}: missing required key {key!r}")

    if raw["origin"] not in VALID_ORIGINS:
        raise CorpusError(
            f"{path.name}: origin must be one of {VALID_ORIGINS}, "
            f"got {raw['origin']!r}"
        )

    msgs = raw["messages"]
    if not isinstance(msgs, list) or not msgs:
        raise CorpusError(f"{path.name}: messages must be a non-empty list")
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise CorpusError(f"{path.name}: message {i} needs 'role' and 'content'")

    gt = raw["ground_truth"]
    unknown = set(gt) - set(CATEGORIES)
    if unknown:
        raise CorpusError(
            f"{path.name}: unknown ground_truth categories {sorted(unknown)}; "
            f"valid: {list(CATEGORIES)}"
        )
    for cat, items in gt.items():
        if not isinstance(items, list) or any(not isinstance(x, str) for x in items):
            raise CorpusError(f"{path.name}: ground_truth.{cat} must be a list of strings")

    return Session(
        id=raw["id"],
        origin=raw["origin"],
        domain=raw.get("domain", "unspecified"),
        messages=msgs,
        ground_truth=gt,
        notes=raw.get("notes", ""),
        path=path,
    )


def load(corpus_dir: str | Path | None = None) -> list[Session]:
    """Load every session in `corpus_dir` (default: the committed corpus).

    Sorted by id so a run is reproducible and two runs are diffable.
    """
    d = Path(corpus_dir) if corpus_dir else CORPUS_DIR
    if not d.exists():
        raise CorpusError(f"corpus directory not found: {d}")

    files = sorted(d.glob("*.json"))
    if not files:
        raise CorpusError(f"no .json session files in {d}")

    sessions = []
    for f in files:
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise CorpusError(f"{f.name}: invalid JSON — {e}") from e
        sessions.append(_validate(raw, f))

    ids = [s.id for s in sessions]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise CorpusError(f"duplicate session ids in {d}: {sorted(dupes)}")
    return sessions


def describe(sessions: list[Session]) -> str:
    """One-line provenance summary, printed at the top of every report so
    a number is never quoted without its sample size and origin."""
    real = sum(1 for s in sessions if s.origin == "real")
    synth = len(sessions) - real
    turns = sum(s.turns for s in sessions)
    labels = sum(len(v) for s in sessions for v in s.ground_truth.values())
    domains = sorted({s.domain for s in sessions})
    return (
        f"{len(sessions)} sessions ({synth} synthetic, {real} real) · "
        f"{turns} turns · {labels} labelled items · "
        f"{len(domains)} domains: {', '.join(domains)}"
    )
