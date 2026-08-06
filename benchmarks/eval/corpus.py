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

LABELLING RULE
--------------
Ground truth is mechanical, not editorial. One rule per category, applied
to every session, labelling everything the rule matches:

* **completed_tasks** — work a turn states as finished.
* **pending_tasks** — work a turn states as in progress or planned.
* **decisions** — a choice a turn commits to ("Decided: X", "Going with
  X", "Switching to X"), or a user instruction naming the technology to
  use ("use X", "build it with X", "it should talk X"). Superseded
  choices are still decisions and are still labelled.
* **files** — a path named in a turn.
* **errors** — a failure, exception, status code, vulnerability class or
  stated malfunction named in a turn.

Two constraints make the rule checkable rather than a matter of taste:

1. **Grounded.** Every label must be recoverable from a single message —
   `validate_grounding()` enforces it, and the eval refuses to run on a
   corpus that fails. A label written from hindsight rather than from the
   transcript ("CUDA out of memory" in a session that never mentions it)
   is unfindable by *any* extractor, so it measures nothing and silently
   caps recall.
2. **Exhaustive.** If the rule matches, it gets labelled. Cherry-picking
   which of several stated decisions "counts" turns precision into a
   measure of the annotator's taste, and the extractor is penalised for
   being right.

RUNNING IT ON YOUR OWN TRANSCRIPTS
----------------------------------
Point the harness at any directory of files in the format above:

    python -m benchmarks.eval --corpus /path/to/my/sessions

Numbers from the committed corpus are directional: 14 sessions is a small
sample, and the synthetic half is easier than real transcripts (the report
prints the split so you can see by how much). If you want a number that
describes YOUR workload, label a few of your own sessions and run against
those — that is the only way to get one, and it is why the loader takes a
path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.eval.metrics import covers

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


def ungrounded(session: Session, threshold: float = 0.6) -> list[tuple[str, str]]:
    """Ground-truth labels of `session` that no single message supports.

    Grounding is checked against one message at a time, not the whole
    transcript: tokens scattered across ten turns are not evidence that any
    turn stated the fact, and pooling them would let almost any label pass.

    Uses the same `covers()` relation the scorer uses, so "grounded" means
    exactly "an extractor that copied this message verbatim would be scored
    as having found this label".
    """
    bad = []
    for cat in CATEGORIES:
        for label in session.expected(cat):
            if not any(covers(m.get("content", ""), label, threshold)
                       for m in session.messages):
                bad.append((cat, label))
    return bad


def validate_grounding(sessions: list[Session], threshold: float = 0.6) -> None:
    """Raise if any label in any session is unsupported by its transcript.

    Called by the harness before scoring. An ungrounded label is a corpus
    bug, not an extraction failure — it is unreachable for every extractor,
    heuristic or LLM, and it depresses recall in a way no code change can
    fix. Failing loudly is the only way that stays true as the corpus grows.
    """
    problems = []
    for s in sessions:
        for cat, label in ungrounded(s, threshold):
            problems.append(f"  {s.id}.{cat}: {label!r}")
    if problems:
        raise CorpusError(
            "ground-truth labels not supported by their transcript "
            f"({len(problems)}):\n" + "\n".join(problems) +
            "\n\nEvery label must be recoverable from a single message. "
            "Either quote the transcript or drop the label."
        )


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
