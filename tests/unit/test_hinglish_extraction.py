"""Hinglish extraction (graph_memory/hinglish.py).

Written alongside the module, so these measure it against the cases its
author thought of — not against real sessions. Each case says what should
be extracted, and the negative cases what must not be.
"""
from __future__ import annotations

import pytest

from tokenmizer.graph_memory.hinglish import extract_hinglish, looks_hinglish


def _labels(d: dict, kind: str) -> list[str]:
    items = d[kind]
    return [i["label"] if isinstance(i, dict) else i for i in items]


def _has(labels: list[str], *words: str) -> bool:
    return any(all(w.lower() in label.lower() for w in words) for label in labels)


# (text, kind, words the label must contain)
POSITIVE = [
    ("Humne decide kiya ki PostgreSQL use karenge orders ke liye.", "decisions", ("PostgreSQL", "orders")),
    ("Session store ke liye Redis use karenge.", "decisions", ("Redis", "session")),
    ("Hum FastAPI use karenge backend ke liye", "decisions", ("FastAPI",)),
    ("Login bug fix kar diya.", "tasks_done", ("Fixed", "login bug")),
    ("Maine payment webhook add kar diya hai.", "tasks_done", ("Added", "payment webhook")),
    ("Migration ho gaya.", "tasks_done", ("Migration",)),
    ("Docker image bana diya", "tasks_done", ("Docker image",)),
    ("Tests add karna baaki hai.", "tasks_todo", ("Add", "tests")),
    ("Refund endpoint update karna hai", "tasks_todo", ("Update", "refund endpoint")),
    ("Rate limiting abhi baaki hai.", "tasks_todo", ("rate limiting",)),
    ("Checkout page nahi chal raha.", "errors", ("checkout page",)),
    ("Payment API fail ho raha hai", "errors", ("payment api",)),
    ("Refund wala bug abhi bhi hai.", "errors", ("refund", "bug")),
    # A postposition carries the target: "staging PE deploy kar diya".
    ("Staging pe deploy kar diya tha.", "tasks_done", ("Deployed", "to staging")),
    ("Ab production pe deploy karna hai.", "tasks_todo", ("Deploy", "to production")),
    ("Config mein timeout update kar diya.", "tasks_done", ("Updated", "timeout")),
    # Hindi verb stems before diya, and a particle before the verb.
    ("Unit tests likh diye.", "tasks_done", ("Wrote", "unit tests")),
    ("Purana endpoint hata diya.", "tasks_done", ("Removed", "endpoint")),
    ("Frontend React pe hi rakhenge.", "decisions", ("React",)),
]


def test_purpose_is_only_the_noun_phrase():
    d = extract_hinglish("Humne decide kiya ki PostgreSQL use karenge orders ke liye.")
    assert [x["label"] for x in d["decisions"]] == ["Use PostgreSQL for orders"]


# Text that must produce nothing of `kind`.
NEGATIVE = [
    ("Redis use nahi karenge.", "decisions"),                 # negated choice
    ("Kya hum MongoDB use karenge?", "decisions"),           # a question
    ("Login bug abhi tak fix nahi kiya.", "tasks_done"),     # negated completion
    ("Kya migration ho gaya?", "tasks_done"),                # a question
    ("Tests add karna hai kya?", "tasks_todo"),              # a question
    ("Deploy abhi mat karo.", "tasks_todo"),                 # a prohibition
    ("Login test pass ho gaya kya?", "tasks_done"),          # a question
    ("Maine coffee li aur lunch kiya.", "tasks_done"),       # not work
]


def test_a_particle_is_never_the_choice():
    labels = [x["label"] for x in extract_hinglish("Frontend React pe hi rakhenge.")["decisions"]]
    assert labels and not any(label.lower().endswith(" hi") for label in labels)


@pytest.mark.parametrize("text,kind,words", POSITIVE)
def test_extracts(text, kind, words):
    labels = _labels(extract_hinglish(text), kind)
    assert _has(labels, *words), f"{kind} from {text!r}: {labels}"


@pytest.mark.parametrize("text,kind", NEGATIVE)
def test_says_nothing(text, kind):
    assert _labels(extract_hinglish(text), kind) == [], text


def test_a_fix_resolves_the_error():
    d = extract_hinglish("Login bug fix kar diya.")
    assert _has(d["resolved_errors"], "login bug")


@pytest.mark.parametrize("text,expected", [
    ("Humne decide kiya ki PostgreSQL use karenge", True),
    ("Tests add karna baaki hai", True),
    ("We decided to use PostgreSQL for orders.", False),
    ("Usamos PostgreSQL para los pedidos.", False),
])
def test_detection(text, expected):
    assert looks_hinglish(text) is expected


def test_reaches_the_graph_through_the_extractor():
    """End to end: a Hinglish session produces nodes, where it produced none."""
    from tokenmizer.graph_memory.hybrid_extractor import get_hybrid_extractor
    d = get_hybrid_extractor(None).heuristic_extract([
        {"role": "user", "content": "Humne decide kiya ki PostgreSQL use karenge orders ke liye."},
        {"role": "assistant", "content": "Theek hai. Login bug fix kar diya, tests add karna baaki hai."},
    ])
    assert _has([x["label"] for x in d.decisions], "PostgreSQL")
    assert _has(d.tasks_done, "login bug")
    assert _has(d.tasks_todo, "tests")


def test_english_sessions_are_untouched():
    from tokenmizer.graph_memory.hybrid_extractor import get_hybrid_extractor
    text = "We fixed the login bug and still need to add tests. Decided: use PostgreSQL."
    d = get_hybrid_extractor(None).heuristic_extract([{"role": "assistant", "content": text}])
    assert not any("done" == t.split()[-1] for t in d.tasks_done)
