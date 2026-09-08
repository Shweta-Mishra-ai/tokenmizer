"""Deterministic session identifiers.

Every entry point needs a session id and none of them had a default, so the
plugin skills told the assistant to *ask the user for one*. A prompt is not a
default: the wording came out differently every time, and because the
assistant has the operator's name in context it would sometimes address them
by it. The tool sounded improvised because it was.

A session id has an obvious deterministic source. The docs already tell people
to use "the same ID across sessions for the same project" and to prefer a
lowercase slug like `auth-service`; the working directory's name is exactly
that, already stable across sessions on the same project, and already what
people type by hand. So derive it, print what was chosen, and only ask when
the derivation genuinely fails.
"""
from __future__ import annotations

import re
from pathlib import Path

# Session ids travel in URL paths and query strings, land in SQLite keys, and
# are shown back to the reader. Restricting to this alphabet keeps all three
# safe without any escaping being load-bearing.
_UNSAFE = re.compile(r"[^a-z0-9]+")

MAX_LENGTH = 48

# Slugifying a filesystem root or a home directory produces an id that is
# both meaningless and likely to collide between unrelated projects.
_TOO_GENERIC = frozenset({
    "", "tmp", "temp", "home", "users", "user", "root", "src", "code",
    "desktop", "documents", "downloads", "projects", "workspace", "repos",
})


def slugify(name: str) -> str:
    """Lowercase, hyphen-separated, trimmed to MAX_LENGTH.

    Truncation cuts at a hyphen where one is available, so a clipped id stays
    readable rather than ending mid-word.
    """
    slug = _UNSAFE.sub("-", name.strip().lower()).strip("-")
    if len(slug) <= MAX_LENGTH:
        return slug
    cut = slug[:MAX_LENGTH]
    boundary = cut.rfind("-")
    return (cut[:boundary] if boundary >= MAX_LENGTH // 2 else cut).strip("-")


def default_session_id(path: str | Path | None = None) -> str | None:
    """The session id for a working directory, or None if it has no good one.

    None is a real answer, not a failure to handle: at a filesystem root or in
    a generic scratch directory there is no meaningful project name, and
    inventing one would file the session under an id that collides with every
    other project in the same parent. Callers ask the user in that case — the
    one situation where asking is the right behaviour.
    """
    directory = Path(path) if path is not None else Path.cwd()
    try:
        directory = directory.resolve()
    except OSError:
        # An unresolvable cwd (deleted directory, permission denied) is not
        # worth failing a checkpoint over.
        return None

    slug = slugify(directory.name)
    if slug in _TOO_GENERIC:
        return None
    return slug
