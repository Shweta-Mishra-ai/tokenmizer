"""
Hinglish — Hindi written in Latin script, mixed with English — as the
extractor's second language.

WHY A SEPARATE PASS
-------------------
Every prose pattern in patterns.py is English, and patterns.looks_english
sends any message that reads as another language to a structure-only path
(checkboxes, file names, exception names) because English patterns misfire
on it. That made a very common way developers actually write invisible:
"PostgreSQL use karenge orders ke liye", "login bug fix kar diya", "tests
add karna baaki hai" produced no nodes at all.

Hindi is verb-final, so the English shapes ("Decided: X", "Fixed X") do not
exist. The verb phrase closes the clause and the thing it is about comes
BEFORE it: "<object> fix kar diya", "<object> karna hai", "<object> use
karenge". So each rule here finds a closing verb phrase and takes its
object from the words in front of it, back to the start of the clause or
to a word that cannot be part of a noun phrase.

WHAT IT DOES NOT DO
-------------------
Devanagari script, other Indian languages, and anything beyond these four
shapes. It errs towards saying nothing: a negated verb ("nahi kiya", "mat
karo") is never read as done or decided, and a clause with a question word
is skipped. Measured only on the hand-written cases in
tests/unit/test_hinglish_extraction.py, which were written alongside it —
treat its accuracy on real sessions as unknown until a real labelled
Hinglish corpus exists.
"""
from __future__ import annotations

import bisect
import functools
import re

# Function words that mark Latin-script Hindi. Distinct from the shared
# _FOREIGN_FW list in patterns.py: that one decides "not English", this one
# decides "Hinglish specifically", so the pass never runs on Spanish.
_HINDI_FW = frozenset("""hai hain ka ki ke ko mein me karo karna kar nahi nahin aur bhi abhi
kya chahiye hona hua hui hue mujhe humko tumko bahut lekin agar toh kuch sirf jaise wala wali
wale raha rahi rahe liye yeh ye woh hum tum aap kiya kiye karenge karege karna karni hoga hogi
sakta sakti apna apni hota hoti tha thi ab itna kaise kyun gaya gayi gaye diya diye di baaki
baki ho par pe se tak""".split())

# Words that end a noun phrase when walking back from the verb: function
# words, and the subject pronouns that open a clause.
_STOP = _HINDI_FW | frozenset("""humne maine tumne aapne usne unhone hamne we i you ki ke
liye ko ne bhi abhi toh to so then aur and lekin but wo woh ye yeh hi""".split())

# Join two halves of one noun phrase ("refund wala bug") and are dropped
# from the label rather than ending it.
_LINKERS = frozenset({"wala", "wali", "wale", "waala", "waali", "waale"})

_QUESTION = re.compile(r"\b(?:kya|kyun|kyon|kaise|kab|kaun|kahan|kis)\b|\?", re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:nahi|nahin|nhi|na|mat|nai|ni)\b", re.IGNORECASE)
_CLAUSE = re.compile(r"[.!?\n;]|,\s|\s(?:aur|lekin|par|but|and)\s", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9_./#+-]+")

# English verbs that come before "kar diya" / "karna hai": "fix kar diya",
# "add karna hai". Mapped to the past tense for a finished task.
_ENGLISH_VERBS = {
    "fix": "Fixed", "add": "Added", "update": "Updated", "deploy": "Deployed",
    "create": "Created", "remove": "Removed", "delete": "Deleted", "merge": "Merged",
    "implement": "Implemented", "test": "Tested", "refactor": "Refactored",
    "migrate": "Migrated", "setup": "Set up", "set": "Set up", "write": "Wrote",
    "install": "Installed", "configure": "Configured", "change": "Changed",
    "build": "Built", "push": "Pushed", "release": "Released", "rename": "Renamed",
    # Hindi verb stems in the same slot: "unit tests LIKH diye".
    "likh": "Wrote", "hata": "Removed", "badal": "Changed", "jod": "Added",
    "bhej": "Sent", "chala": "Ran", "daal": "Added", "bana": "Created", "laga": "Added",
}

# ── closing verb phrases ─────────────────────────────────────────────────────
_KAR = r"(?:kar|kr)"
_DONE = re.compile(
    r"\b(?:(?P<verb>[a-z]+)\s+)?(?:" + _KAR + r"\s+(?:diya|diye|di|liya|liye|li|chuke|chuka)"
    r"|ho\s+(?:gaya|gayi|gaye|gya|gyi|chuka|chuki)"
    r"|(?:diya|diye|di|liya|liye|li)(?=\W|$)"
    r"|(?:done|complete|ready|finish)\s+(?:hai|ho\s+gaya|ho\s+gayi))\b",
    re.IGNORECASE,
)
_AUXILIARY = re.compile(r"\b(?:kar|kr|ho|done|complete|ready|finish)\b", re.IGNORECASE)

_TODO = re.compile(
    r"\b(?:(?P<verb>[a-z]+)\s+)?(?:" + _KAR + r"(?:na|ni)\s+(?:hai|hain|baaki\s+hai|baki\s+hai|"
    r"padega|padegi|hoga|hogi)|baaki\s+hai|baki\s+hai|pending\s+hai|karna\s+chahiye)\b",
    re.IGNORECASE,
)
_USE = re.compile(
    r"\b(?:use|istemal|istemaal|lagayenge)\s+(?:" + _KAR + r"(?:enge|ege|te\s+hain|na\s+hai|o|eng)"
    r"|hoga|karenge)\b|\b(?:lenge|rakhenge|pe\s+jayenge|par\s+jayenge|final\s+hai)\b",
    re.IGNORECASE,
)
_BROKEN = re.compile(
    r"\b(?:nahi|nahin|nhi)\s+(?:chal|kaam\s+kar)\s+(?:raha|rahi|rahe)(?:\s+hai)?"
    r"|\b(?:fail|crash|break)\s+ho\s+(?:raha|rahi|rahe|gaya|gayi)(?:\s+hai)?"
    r"|\b(?:error|exception)\s+(?:aa|de)\s+(?:raha|rahi)(?:\s+hai)?"
    r"|\btoot\s+(?:gaya|gayi)"
    r"|\b(?:bug|issue|problem)\s+(?:abhi\s+bhi\s+|ab\s+bhi\s+)?(?:hai|hain)\b",
    re.IGNORECASE,
)
_PURPOSE = re.compile(r"\b(?P<what>[A-Za-z0-9_./-]+(?:\s+[A-Za-z0-9_./-]+)?)\s+ke\s+liye\b",
                      re.IGNORECASE)


def looks_hinglish(text: str) -> bool:
    """At least two Hindi function words, and more of them than of any other
    language patterns.looks_english recognises."""
    toks = [w.lower() for w in re.findall(r"[A-Za-z]+", text or "")]
    return sum(w in _HINDI_FW for w in toks) >= 2


@functools.lru_cache(maxsize=16)
def _clause_marks(text: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(starts, ends) of every clause separator, found once per text."""
    found = list(_CLAUSE.finditer(text))
    return tuple(m.start() for m in found), tuple(m.end() for m in found)


def _clause_bounds(text: str, pos: int) -> tuple[int, int]:
    """The clause around `pos`. Separators are found once and
    binary-searched: rescanning from the start for every match made a long
    message quadratic (150 KB took 15 s)."""
    starts, ends = _clause_marks(text)
    i = bisect.bisect_right(ends, pos)
    start = ends[i - 1] if i else 0
    j = bisect.bisect_left(starts, pos)
    return start, starts[j] if j < len(starts) else len(text)


def _object_before(text: str, start: int, end: int, max_words: int = 6) -> str:
    """The noun phrase ending at `end`, walking back to `start` and
    stopping at a word that cannot be part of it."""
    words = _WORD.findall(text[start:end])
    out: list[str] = []
    for w in reversed(words):
        if w.lower() in _LINKERS:
            continue        # "refund WALA bug": joins the phrase, adds nothing
        if w.lower() in _STOP or len(out) >= max_words:
            break
        out.append(w)
    return " ".join(reversed(out)).strip(" .,-")


# Hindi postpositions follow their noun: "staging PE", "config MEIN". When
# one sits right before the verb, the noun in front of it is where the work
# went, not what it was: "staging pe deploy kar diya" is "Deployed to
# staging".
_POSTPOSITIONS = {"pe": "to", "par": "to", "mein": "in", "me": "in", "se": "from",
                  "tak": "to"}


def _target_before(text: str, clause_start: int, end: int) -> str:
    """"to staging" for "...staging pe" ending at `end`, else ""."""
    words = _WORD.findall(text[clause_start:end])
    if len(words) >= 2 and words[-1].lower() in _POSTPOSITIONS:
        cut = text.rfind(words[-1], clause_start, end)
        noun = _object_before(text, clause_start, cut, max_words=3)
        if noun:
            return f"{_POSTPOSITIONS[words[-1].lower()]} {noun}"
    return ""


def _split_verb(text: str, clause_start: int, m: re.Match) -> tuple[str, str]:
    """(English verb or "", object) for a matched closing verb phrase.

    The optional word before "kar diya" / "karna hai" is a verb only when it
    is one of _ENGLISH_VERBS ("fix kar diya"); otherwise it is the last word
    of the object ("migration ho gaya"), and the object runs up to the
    phrase itself.
    """
    word = (m.group("verb") or "").lower()
    if word in _ENGLISH_VERBS:
        obj = _object_before(text, clause_start, m.start("verb"))
        if obj:
            # "config mein timeout update kar diya": the object, and where.
            return word, obj
        return word, _target_before(text, clause_start, m.start("verb"))
    # A function word in that slot ("rate limiting ABHI baaki hai") is not
    # part of the object either.
    if not m.group("verb") or word in _STOP:
        end = m.start("verb") if m.group("verb") else m.start()
    else:
        end = m.end("verb")
    return "", _object_before(text, clause_start, end)


def _capitalise(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def extract_hinglish(text: str) -> dict:
    """{"tasks_done", "tasks_todo", "decisions", "errors", "resolved_errors"}
    read from Hinglish clauses. Labels are English where the verb was, so a
    resume block reads the same whatever language the session was in."""
    out: dict = {"tasks_done": [], "tasks_todo": [], "decisions": [],
                 "errors": [], "resolved_errors": []}
    seen: set = set()

    def add(kind: str, label: str) -> None:
        label = re.sub(r"\s+", " ", label).strip(" .,-")
        key = (kind, label.lower())
        if len(label) >= 4 and key not in seen:
            seen.add(key)
            out[kind].append(label)

    for m in _DONE.finditer(text):
        # A bare "diya"/"li" completes only a known verb stem: "unit tests
        # likh diye" is work, "maine coffee li" is a coffee.
        if (not _AUXILIARY.search(m.group(0))
                and (m.group("verb") or "").lower() not in _ENGLISH_VERBS):
            continue
        c_start, c_end = _clause_bounds(text, m.start())
        clause = text[c_start:c_end]
        if _QUESTION.search(clause) or _NEGATION.search(text[c_start:m.end() + 8]):
            continue
        verb, obj = _split_verb(text, c_start, m)
        if not obj:
            continue
        if verb:
            add("tasks_done", f"{_ENGLISH_VERBS[verb]} {obj}")
            if verb == "fix":
                add("resolved_errors", obj)
        else:
            add("tasks_done", f"{_capitalise(obj)} done")

    for m in _TODO.finditer(text):
        c_start, c_end = _clause_bounds(text, m.start())
        if _QUESTION.search(text[c_start:c_end]):
            continue
        verb, obj = _split_verb(text, c_start, m)
        if not obj:
            continue
        add("tasks_todo", f"{_capitalise(verb)} {obj}" if verb else _capitalise(obj))

    for m in _USE.finditer(text):
        c_start, c_end = _clause_bounds(text, m.start())
        clause = text[c_start:c_end]
        if _QUESTION.search(clause) or _NEGATION.search(text[c_start:m.end()]):
            continue
        # The object is in front of the verb, but a purpose phrase ("orders
        # ke liye") may sit between them or after; neither is the choice.
        before = re.sub(r"\b[A-Za-z0-9_./-]+(?:\s+[A-Za-z0-9_./-]+)?\s+ke\s+liye\b", " ",
                        text[c_start:m.start()], flags=re.IGNORECASE)
        obj = _object_before(before, 0, len(before), max_words=3)
        if not obj:
            # "frontend React PE hi rakhenge": stay ON React.
            trimmed = re.sub(r"(?:\s+(?:pe|par|hi|bhi|mein|me))+\s*$", "", before,
                             flags=re.IGNORECASE)
            obj = _object_before(trimmed, 0, len(trimmed), max_words=3)
        if not obj:
            continue
        purpose = _PURPOSE.search(clause)
        what = (_object_before(clause, 0, purpose.start() + len(purpose.group("what")), 3)
                if purpose else "")
        label = f"Use {obj}" + (f" for {what}" if what else "")
        out["decisions"].append({"label": label, "reason": "", "source_role": "user"})

    for m in _BROKEN.finditer(text):
        c_start, _ = _clause_bounds(text, m.start())
        obj = _object_before(text, c_start, m.start())
        if obj:
            add("errors", f"{obj} not working" if "chal" in m.group(0).lower()
                or "kaam" in m.group(0).lower() else f"{obj} {m.group(0).split()[0].lower()}")

    return out
