"""
The plugin skills must specify behaviour, not hand the assistant a script.

The skills used to say "Ask the user for a session ID if not provided" and
"Tell the user: Loaded [X] tokens...". Those are directions to improvise, so
the wording came out differently on every run — and because the assistant has
the operator's name in its context, it would sometimes address them by it. A
tool that greets different operators differently reads as improvised, because
it is.

The rule these tests hold: a skill asks a question only where the code cannot
produce an answer, and renders a fixed template everywhere else. The code-side
half of this is tokenmizer.core.session.default_session_id, which gives every
entry point a deterministic session ID so there is nothing to ask about.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILLS = sorted((Path(__file__).resolve().parents[2]
                 / ".claude-plugin" / "skills").glob("*/SKILL.md"))


def _body(path: Path) -> str:
    """Skill text with the YAML frontmatter and fenced code blocks removed.

    Frontmatter is metadata and code blocks are literal output templates;
    neither is an instruction to the assistant.
    """
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def test_skills_exist():
    assert SKILLS, "no SKILL.md files found"


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_no_emoji_in_skill_output(path):
    """Emoji carry no information the structure does not already carry, and
    they are what made the output read as unprofessional."""
    found = sorted({c for c in path.read_text(encoding="utf-8") if ord(c) > 0x2500})
    assert not found, (
        f"{path.parent.name}: remove {[hex(ord(c)) for c in found]}"
    )


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_no_open_ended_instruction_to_ask(path):
    """"Ask the user for X if not provided" has no fixed wording, so it has
    no fixed behaviour. A skill may still ask — but only with the exact
    question written down, and only where the code genuinely cannot answer."""
    offending = [
        line.strip()
        for line in _body(path).splitlines()
        # "Never ask the user ..." is a prohibition, which is the behaviour
        # we want, so only unqualified directions to ask are rejected.
        if re.search(r"(?<!never )\bask (?:the )?user\b", line, re.IGNORECASE)
        and "ask exactly" not in line.lower()
    ]
    assert not offending, (
        f"{path.parent.name}: replace with a deterministic default or an "
        f"exact question — {offending}"
    )


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_no_free_form_narration_directive(path):
    """"Tell the user: ..." produces the assistant's voice rather than the
    tool's. Output templates belong in a fenced block the skill renders."""
    offending = [
        line.strip()
        for line in _body(path).splitlines()
        if re.search(r"\btell (?:the )?user\b", line, re.IGNORECASE)
    ]
    assert not offending, (
        f"{path.parent.name}: move the wording into an output template — "
        f"{offending}"
    )
