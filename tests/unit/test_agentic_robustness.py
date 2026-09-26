"""
Agent sessions, real-text noise, and inputs nobody sends on purpose.

Everything here was found by running the product on something other than a
labelled corpus: a real 700-message Claude Code session (agent tool calls,
code, tables, quoted examples, a Hindi-English mix), a fuzzer throwing
malformed and hostile messages at every extraction entry point, a scan of
every regex in the package against adversarial payloads, and a concurrent
load test of the HTTP endpoint. Each test pins one defect that surfaced
there, and most of them predate the work that found them.
"""
from __future__ import annotations

import json
import time

import pytest

from tokenmizer.graph_memory.graph import GraphMemory, NodeType
from tokenmizer.graph_memory.helpers import scrub_surrogates, tool_signals
from tokenmizer.graph_memory.hybrid_extractor import HybridExtractor
from tokenmizer.graph_memory.patterns import looks_english, mask_mentions


def _x(msgs):
    return HybridExtractor().heuristic_extract(msgs)


def _graph(tmp_path, msgs, incremental=True):
    g = GraphMemory("s", storage_dir=str(tmp_path))
    g.extract_from_messages(msgs, incremental=incremental)
    return g


def _labels(g, node_type):
    return [n.label for n in g._nodes.values() if n.type == node_type and not n._evicted]


# ── Tool calls and tool results ──────────────────────────────────────────────

def test_anthropic_tool_use_edits_become_files_and_reads_do_not():
    files, errors = tool_signals({"role": "assistant", "content": [
        {"type": "text", "text": "Fixing it."},
        {"type": "tool_use", "name": "Edit", "input": {"file_path": "src/api/orders.py"}},
        {"type": "tool_use", "name": "Read", "input": {"file_path": "src/api/users.py"}},
        {"type": "tool_use", "name": "Bash", "input": {"command": "pytest"}},
    ]})
    assert files == ["src/api/orders.py"] and errors == []


def test_openai_tool_calls_and_tool_role_errors():
    files, _ = tool_signals({"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function",
         "function": {"name": "write_file", "arguments": '{"path": "docs/intro.md"}'}}]})
    assert files == ["docs/intro.md"]
    _, errors = tool_signals({"role": "tool", "content": "Error: ENOENT: no such file or directory"})
    assert errors == ["Error: ENOENT: no such file or directory"]
    assert tool_signals({"role": "tool", "content": "3 files written"}) == ([], [])


def test_failed_tool_result_is_labelled_by_its_exception_not_its_header():
    _, errors = tool_signals({"role": "user", "content": [{
        "type": "tool_result", "is_error": True,
        "content": "Exit code 1\nTraceback (most recent call last):\n  File \"a.py\"\nKeyError: 'id'"}]})
    assert errors == ["KeyError: 'id'"]
    _, errors = tool_signals({"role": "user", "content": [{
        "type": "tool_result", "is_error": True,
        "content": [{"type": "text", "text": "Exit code 1\nFAILED tests/test_a.py::test_x - AssertionError"}]}]})
    assert errors == ["FAILED tests/test_a.py::test_x - AssertionError"]


@pytest.mark.parametrize("msg", [
    {"role": "assistant", "tool_calls": "garbage"},
    {"role": "assistant", "tool_calls": [5, None, {"function": "x"}]},
    {"role": "assistant", "content": [1, None, {"type": "tool_use", "input": "x"}]},
    {"role": "assistant", "tool_calls": [{"function": {"name": "w", "arguments": "not json"}}]},
    "not a dict",
])
def test_malformed_tool_calls_never_raise(msg):
    assert tool_signals(msg) == ([], [])


def test_every_tool_turn_reaches_extraction(tmp_path):
    """_msg_hash hashed the TEXT only, so every text-less tool turn hashed
    to sha1("") and all but the first were skipped as already processed."""
    msgs = [{"role": "user", "content": "Refactor the service."}]
    for i in range(5):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "edit_file", "arguments": json.dumps({"path": f"svc/mod{i}.py"})}}]})
    g = GraphMemory("s", storage_dir=str(tmp_path))
    for k in range(1, len(msgs) + 1):
        g.extract_from_messages(msgs[:k])
    assert sorted(_labels(g, NodeType.FILE)) == [f"svc/mod{i}.py" for i in range(5)]


def test_absolute_and_relative_spellings_are_one_file(tmp_path):
    g = GraphMemory("s", storage_dir=str(tmp_path))
    g.extract_from_messages([{"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {
            "name": "edit", "arguments": '{"path": "/home/dev/app/src/api/orders.py"}'}}]}])
    g.extract_from_messages([{"role": "assistant", "content": "Updated src/api/orders.py."}])
    g.extract_from_messages([{"role": "assistant", "content": "And src/api/orders.py again."}])
    assert _labels(g, NodeType.FILE) == ["src/api/orders.py"]


# ── Mentions are not statements ──────────────────────────────────────────────

def test_mask_keeps_offsets_and_log_lines():
    text = ('It reads "retry for failed deliveries" as an error.\n| tasks | 53% |\n'
            '```\nraise ValueError("x")\n```\n```\nValueError: bad date\n```\n'
            'I get "TypeError: x is undefined".')
    masked = mask_mentions(text)
    assert len(masked) == len(text)
    assert "retry for failed" not in masked and "53%" not in masked
    assert "raise ValueError" not in masked
    assert "ValueError: bad date" in masked and "TypeError: x is undefined" in masked


def test_quoted_examples_tables_and_code_are_not_extracted():
    x = _x([{"role": "assistant", "content":
             'The pattern misreads "background retry for failed webhook deliveries".\n'
             "| Category | Before | After |\n| Pending tasks | 2% | 32% |\n"
             "```python\n# TODO: remove this hack\nraise RuntimeError('boom')\n```"}])
    assert x.errors == [] and x.tasks_todo == [] and x.tasks_done == []


def test_a_gerund_phrase_that_is_a_subject_is_not_work_in_progress():
    # Found on the real session: "my edit adding X didn't apply" was stored
    # as the task "X didn't apply (count 0), so that result".
    r = _x([{"role": "assistant", "content":
             "The scan is clean, but my edit adding the new input shapes to "
             "the scanner didn't apply (count 0)."}])
    assert r.tasks_wip == []
    r = _x([{"role": "assistant", "content":
             "Now adding retries that didn't exist before to the webhook client."}])
    assert r.tasks_wip == ["retries that didn't exist before to the webhook client"]


def test_a_passive_agent_is_not_finished_work():
    # "A test set written by someone other than me" became the completed
    # task "Written by someone other than me".
    r = _x([{"role": "assistant", "content":
             "We still want a test set written by someone other than me."}])
    assert r.tasks_done == []
    # "by" + gerund is how the work was done, and is kept.
    r = _x([{"role": "assistant", "content":
             "Fixed by adding a 5 second context timeout in the order client."}])
    assert r.tasks_done, r


def test_a_label_cut_inside_a_code_span_is_closed():
    from tokenmizer.graph_memory.patterns import _clip
    assert _clip("a new error pattern in `graph_memory/patterns.py") == \
        "a new error pattern in `graph_memory/patterns.py`"
    assert _clip("use `uv` for installs") == "use `uv` for installs"


def test_a_pasted_traceback_in_a_code_block_is_still_an_error():
    x = _x([{"role": "user", "content":
             "Getting this:\n```\nTraceback (most recent call last):\n  File \"app.py\", line 3\n"
             "ZeroDivisionError: division by zero\n```"}])
    assert any("ZeroDivisionError" in e for e in x.errors), x.errors


# ── Other languages ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Abhi mujha paisa ni bada key add karogi paid but pending all task and task recall",
    "Mujhe ye feature chahiye, backend ka refactor karna hai aur tests bhi likhne hai",
    "Necesitamos migrar la base de datos a Postgres antes del viernes",
    "Wir müssen die Datenbank bis Freitag auf Postgres migrieren.",
    "Nous devons migrer la base de données vers Postgres avant vendredi.",
    "我们需要把数据库迁移到 Postgres",
])
def test_non_english_is_recognised(text):
    assert not looks_english(text)


@pytest.mark.parametrize("text", [
    "Created: db/session.py. Files: api/orders.py.",
    "Haven't started automated dependency updates yet, soon though.",
    "Use Postgres.",
    "Decided: Postgres for the orders store.",
])
def test_terse_english_is_english(text):
    assert looks_english(text)


def test_non_english_prose_yields_structure_only():
    """English prose patterns misfire on another language: "…karogi paid but
    pending all task and task recall" became a to-do. What still carries
    across is structure — paths, exception names, checkboxes, headers."""
    x = _x([{"role": "user", "content":
             "Abhi backend ka refactor karna hai aur pending sab kaam bhi, "
             "api/orders.py me TypeError aa raha hai.\n- [ ] tests likhne hai"}])
    assert x.tasks_done == [] and x.decisions == []
    assert x.tasks_todo == ["tests likhne hai"]
    assert "api/orders.py" in x.files
    assert any("TypeError" in e for e in x.errors)


# ── Noise found in the real session ──────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "The guard uses fixed-width lookbehinds.",          # `fixed-` is not "fixed"
    "Pending-task recall is 19% and errors are 45%.",   # `pending-` is not a to-do
    "It is a built-in helper.",
])
def test_first_half_of_a_hyphenated_word_is_not_a_keyword(text):
    x = _x([{"role": "assistant", "content": text}])
    assert x.tasks_done == [] and x.tasks_todo == [], (x.tasks_done, x.tasks_todo)


@pytest.mark.parametrize("text", [
    "Verifying the commit attribution I wrote in the report before committing.",
    "The numbers I'd written were from round 2.",
    "Fixing a bug in the pattern I just wrote (a leading anchor was missing).",
    "- **Pending tasks:** 2% -> 32%",
    "I'll start by checking what real text is available.",
    "Dev pending recall dropped 3 items.",
])
def test_narration_is_not_a_task(text):
    x = _x([{"role": "assistant", "content": text}])
    assert x.tasks_done == [] and x.tasks_todo == [] and x.tasks_wip == [], (
        x.tasks_done, x.tasks_todo, x.tasks_wip)


def test_category_word_starting_a_real_todo_is_kept():
    assert _x([{"role": "assistant", "content":
                "Still need to get to error boundaries around each route."}]).tasks_todo


def test_hyphenated_word_is_not_cut_in_an_error_label():
    errors = _x([{"role": "assistant", "content":
                  "- To-dos from early in long sessions being silently dropped."}]).errors
    assert errors and errors[0].startswith("To-dos"), errors


def test_editing_two_common_nouns_is_not_a_decision_change():
    x = _x([{"role": "assistant", "content":
             "Now replacing an error fragment with the longer label."}])
    assert x.superseded == [] and x.decisions == []


def test_adding_a_feature_is_not_a_dependency():
    assert _x([{"role": "assistant", "content": "Now adding agentic tool-call support."}]).dependencies == []
    assert _x([{"role": "assistant", "content": "pip install httpx"}]).dependencies == ["httpx"]


def test_working_on_shows_the_latest_work_first(tmp_path):
    g = GraphMemory("s", storage_dir=str(tmp_path))
    for i in range(6):
        g.extract_from_messages([{"role": "assistant", "content": f"Working on the module number {i} rewrite."}],
                                incremental=True)
    line = next(ln for ln in g.to_context_block(token_budget=400).splitlines()
                if ln.startswith("Working on"))
    assert line.index("number 5") < line.index("number 2"), line


# ── Hostile input ────────────────────────────────────────────────────────────

def test_lone_surrogate_does_not_break_extraction_or_persistence(tmp_path):
    """json.loads accepts "\\ud800"; the first .encode() that met it raised
    and extraction for the whole conversation failed."""
    msgs = [{"role": "assistant", "content": "Decided: Postgres \ud800 for orders."}]
    g = _graph(tmp_path, msgs)
    hashes = set(g._processed_hashes)
    g.extract_from_messages(msgs)
    assert g._processed_hashes == hashes
    reloaded = GraphMemory("s", storage_dir=str(tmp_path))
    assert any("Postgres" in n.label for n in reloaded._nodes.values())
    assert scrub_surrogates({"a": ["x\ud800"]}) == {"a": ["x�"]}


_PAYLOADS = {
    "spaces": " " * 15000,
    "newlines": "\n" * 15000,
    "digits": "1" * 15000,
    "hyphens": "a-" * 7500,
    "dot then long space": (". " + " " * 300) * 50,
    "tab run": "\t" * 15000,
    "blank run": "\n" * 15000,
    "short words then it is": "a " * 5000 + "it is.",
    "quoted runs": ('a "' + "x" * 250 + '" . ') * 55,
    "bullets": ("- " + " " * 200 + "\n") * 70,
}


@pytest.mark.parametrize("name", list(_PAYLOADS))
def test_extraction_stays_fast_on_adversarial_payloads(name):
    """A 5,000-space message took 16.5 s in _TASK_DONE_PASSIVE; blank lines
    1.3 s in _DECISION_HEADER; a digit run 1.3 s in _EVIDENCE_NUMBER; the
    header patterns were quadratic over the space runs masking leaves. The
    ceiling is generous for slow CI machines and still an order of magnitude
    below any of those."""
    # Wrapped in real characters: an all-blank message is skipped before any
    # pattern runs, and the original 16-second case was "x" + spaces + "x".
    text = "x" + _PAYLOADS[name] + "x"
    started = time.monotonic()
    _x([{"role": "user", "content": text}, {"role": "assistant", "content": text}])
    assert time.monotonic() - started < 2.0


def test_the_graph_layer_accepts_what_the_extractor_accepts(tmp_path):
    msgs = [{"role": "user", "content": None},
            {"role": "assistant", "content": [None, 3, {"type": "image"}]},
            {"role": "tool", "content": {"text": "Error: boom"}},
            {"role": None, "content": "Done: survived."}]
    g = _graph(tmp_path, msgs)
    g.to_context_block(token_budget=0)
    g.to_context_block(token_budget=400)


# ── The HTTP boundary ────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    from tokenmizer.api import app as app_module
    from tokenmizer.api.app import app
    from tokenmizer.providers.providers import LLMResponse

    fake = AsyncMock()
    fake.chat = AsyncMock(return_value=LLMResponse(
        text="ok", input_tokens=1, output_tokens=1, model="m", provider="fake", latency_ms=1.0))
    monkeypatch.setattr(app_module, "_get_provider", lambda: fake)
    monkeypatch.setattr(app_module.settings.cache, "enabled", False)
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.mark.parametrize("field,value", [
    ("max_tokens", -5), ("max_tokens", 0), ("max_completion_tokens", 0),
    ("temperature", -1), ("temperature", 3), ("top_p", 1.5),
])
def test_out_of_range_sampling_values_are_rejected_up_front(client, field, value):
    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "hi"}], field: value})
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("name,content", [
    ("a.csv", "id,name\n1,x\ud800y\n"),
    ("a.json", '{"k": "x\ud800"}'),
    ("a.txt", "hello \ud800 world\n"),
])
def test_file_intelligence_accepts_a_lone_surrogate(name, content):
    from tokenmizer.filters.file_intelligence import FileIntelligence
    result = FileIntelligence().process(content, name)
    assert result.original_size_bytes > 0


@pytest.mark.parametrize("session_id", ["सत्र-एक", "%F0%9F%9A%80run", "a%22%3B b"])
def test_obsidian_export_accepts_any_session_id(client, session_id):
    # The raw id went into the Content-Disposition header, which must be
    # Latin-1: a non-English id was a 500.
    r = client.get(f"/api/graph/{session_id}/obsidian")
    assert r.status_code == 200, r.text
    disposition = r.headers["content-disposition"]
    assert disposition.isascii() and '"' not in disposition and disposition.count(";") == 1


def test_a_session_id_that_is_not_text_is_a_422_not_a_503(client):
    # It failed inside the ownership store and was reported as the
    # server's outage ("ownership state unavailable").
    r = client.post("/v1/chat/completions", content=(
        '{"model":"m","session_id":"s\\ud800x",'
        '"messages":[{"role":"user","content":"hi"}]}'),
        headers={"Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    assert "session_id" in r.text


@pytest.mark.parametrize("body", [
    '{"model":"m","messages":[{"role":"user","content":"\\ud800 hi"}]}',
    '{"model":"m","messages":[{"role":"user","content":"a\\u0000b"}]}',
    '{"model":"m","messages":[{"role":"user","content":"' + " " * 50000 + '"}]}',
    '{"model":"m","messages":[{"role":"assistant","content":null,"tool_calls":'
    '[{"id":"1","type":"function","function":{"name":"edit","arguments":"{\\"path\\": 5}"}}]},'
    '{"role":"tool","tool_call_id":"1","content":"Error: boom"}]}',
    '{"model":"m","session_id":"s\\ud800x","messages":[{"role":"user","content":"hi"}]}',
# Explicit ids: pytest copies the test id into the PYTEST_CURRENT_TEST
# environment variable, and Windows rejects one over 32,767 characters.
], ids=["lone-surrogate", "nul-byte", "50k-whitespace", "bad-tool-args",
        "surrogate-session-id"])
def test_hostile_bodies_never_produce_a_server_error(client, body):
    r = client.post("/v1/chat/completions",
                    content=body.encode("utf-8", "surrogatepass"),
                    headers={"content-type": "application/json"})
    assert r.status_code < 500, r.text


@pytest.mark.parametrize("text", [
    "A test that passes immediately proves nothing unless it would fail on the old code.",
    "This could break under load.",
    "The job is expected to fail on Windows.",
])
def test_a_possible_failure_is_not_an_error(text):
    assert _x([{"role": "assistant", "content": text}]).errors == []


def test_a_real_failure_still_is():
    assert _x([{"role": "assistant", "content": "The build fails with exit code 1 on the linter step."}]).errors


def test_pytest_error_line_stays_an_error(tmp_path):
    """The validator retyped any two-word label ending in a path as a FILE,
    which filed "ERROR tests/unit/test_api.py" as a file being worked on."""
    g = _graph(tmp_path, [{"role": "user", "content": [{
        "type": "tool_result", "is_error": True,
        "content": "ERROR tests/unit/test_api.py"}]}])
    assert _labels(g, NodeType.ERROR) == ["ERROR tests/unit/test_api.py"]
    assert _labels(g, NodeType.FILE) == []


def test_pronoun_object_is_not_a_task():
    assert _x([{"role": "assistant", "content": "Adding them properly and rescanning."}]).tasks_wip == []


def test_label_does_not_end_on_a_contraction_apostrophe():
    labels = _x([{"role": "assistant", "content":
                  "Now adding a retry wrapper so that the result doesn't get lost."}]).tasks_wip
    assert labels and not labels[0].endswith("doesn"), labels


def test_a_time_adverbial_before_the_subject_is_not_part_of_it():
    todo = _x([{"role": "assistant", "content":
                "At some point documenting the model card has to happen too."}]).tasks_todo
    assert todo == ["documenting the model card"], todo


# ── Round 5: general constructions, measured on held-out v3 ──────────────────

@pytest.mark.parametrize("text,kind,needle", [
    ("Payment webhooks: done.", "done", "payment webhooks"),
    ("[done] the nightly export", "done", "nightly export"),
    ("Managed to finish the CSV importer.", "done", "csv importer"),
    ("The retention job went live on Monday.", "done", "retention job"),
    ("Upcoming: rotating the signing keys.", "todo", "rotating the signing keys"),
    ("The SSO migration keeps getting pushed.", "todo", "sso migration"),
    ("The quota dashboard is waiting on review.", "todo", "quota dashboard"),
    ("The audit trail needs to get done before the release.", "todo", "audit trail"),
    ("Somebody should look into the cold-start budget.", "todo", "cold-start budget"),
    ("We landed on Temporal for the workflows.", "decision", "temporal"),
    ("After benchmarking, Valkey came out ahead.", "decision", "valkey"),
    ("On-call got paged over a disk-full condition on the primary.", "error", "disk-full condition"),
    ("The culprit was an unbounded retry loop.", "error", "unbounded retry loop"),
    ("Error — checksum mismatch on the uploaded archive.", "error", "checksum mismatch"),
    ("Customers reported double-charged renewals.", "error", "double-charged renewals"),
])
def test_general_constructions(text, kind, needle):
    x = _x([{"role": "assistant", "content": text}])
    got = {"done": x.tasks_done, "todo": x.tasks_todo + x.tasks_wip,
           "decision": [d["label"] for d in x.decisions], "error": x.errors}[kind]
    assert any(needle in g.lower() for g in got), got


@pytest.mark.parametrize("text", [
    "Nothing failed on the last run.",
    "Shipped the deploy. Nothing failed overnight.",
    "No jobs crashed this week.",
    "Fixed by adding a 5 second context timeout in client.go.",
])
def test_no_failure_and_fix_descriptions_are_not_errors(text):
    assert _x([{"role": "assistant", "content": text}]).errors == []


def test_went_into_a_file_is_not_a_completion():
    assert _x([{"role": "assistant", "content": "Most of the afternoon went into api/app.py."}]).tasks_done == []


def test_object_window_stops_at_a_full_stop():
    errors = _x([{"role": "assistant", "content":
                  "The job failed on the last run. Most of the afternoon went elsewhere."}]).errors
    assert all("afternoon" not in e for e in errors), errors


def test_filename_at_the_end_of_an_error_is_kept_whole():
    errors = _x([{"role": "assistant", "content":
                  "The build fails with exit code 1 on the linter step in ci/lint.yml."}]).errors
    assert any(e.endswith("ci/lint.yml") for e in errors), errors


def test_planned_work_is_not_part_of_an_error_label():
    errors = _x([{"role": "assistant", "content": "We need to fix the memory leak in the parser."}]).errors
    assert errors == ["memory leak in the parser"], errors


@pytest.mark.parametrize("text,kind", [
    ("Pendiente: migrar la base de datos.", "todo"),
    ("Fehler: Verbindung abgelehnt.", "error"),
    ("Erreur: délai dépassé sur l'export.", "error"),
    ("Decisión: Postgres para pedidos.", "decision"),
])
def test_headers_in_other_languages(text, kind):
    x = _x([{"role": "user", "content": text}])
    got = {"todo": x.tasks_todo, "error": x.errors, "decision": x.decisions}[kind]
    assert got, (text, x)


def test_keyword_prefilter_never_changes_output():
    """_PREFILTER skips a pattern when none of its keywords is in the
    message. That is only safe if each keyword list is a superset; this
    compares output with and without it."""
    import dataclasses
    import glob

    import tokenmizer.graph_memory.hybrid_extractor as H

    sessions = [json.load(open(f))["messages"]
                for f in sorted(glob.glob("benchmarks/eval/corpus*/*.json"))]
    sessions += [[{"role": "assistant", "content": t}] for t in (
        "Deploy failed, rolled back.", "The cache returns stale data after a deploy.",
        "Query takes 12 seconds, it used to take 200ms.", "stats reported healthy over an empty db",
        "The fallback is unreachable.", "Rate limiting is implemented.", "The worker keeps restarting.",
        "Docker build broke after the base image bump.", "one bad read deletes the whole file",
    )]

    def run():
        return [dataclasses.asdict(H.HybridExtractor().heuristic_extract(s)) for s in sessions]

    with_filter = run()
    saved = dict(H._PREFILTER)
    H._PREFILTER.clear()
    try:
        without = run()
    finally:
        H._PREFILTER.update(saved)
    assert with_filter == without


def test_llm_prompt_sees_content_blocks_tool_calls_and_language_rules():
    import asyncio

    seen = {}

    async def fake_model(messages, system, max_tokens):
        seen["prompt"], seen["system"] = messages[0]["content"], system
        return {"text": '{"files": ["api/orders.py"]}'}

    msgs = [
        {"role": "user", "content": "Mujhe backend ka refactor karna hai"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Editing now."},
            {"type": "tool_use", "name": "Edit", "input": {"file_path": "api/orders.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "is_error": True,
                                      "content": "Traceback (most recent call last):\nKeyError: id"}]},
    ]
    result = asyncio.run(HybridExtractor().llm_extract(msgs, fake_model))
    assert "Editing now. [edited api/orders.py]" in seen["prompt"]
    assert "[tool error: KeyError: id]" in seen["prompt"]
    assert "any language" in seen["system"]
    assert result.files == ["api/orders.py"]


def test_a_match_across_a_blanked_quote_is_skipped():
    assert _x([{"role": "assistant", "content":
                'The one I found earlier: "retry for failed deliveries" read as an error.'}]).errors == []


def test_postfix_status_must_end_its_clause():
    assert _x([{"role": "assistant", "content":
                "Now the per-message passes: completed and pending."}]).tasks_done == []


def test_observation_verb_is_not_part_of_the_error_label():
    assert _x([{"role": "assistant", "content": "We found a race in the fixture teardown."}]).errors == [
        "race in the fixture teardown"]


def test_no_source_string_holds_a_lone_surrogate():
    # A non-raw docstring containing the text "\ud800" is a real lone
    # surrogate; on Python 3.13 importing the app then failed for every API
    # test. Write such examples as raw strings.
    import ast
    import pathlib

    import tokenmizer
    bad = []
    for path in pathlib.Path(tokenmizer.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and any(
                    0xD800 <= ord(c) <= 0xDFFF for c in node.value):
                bad.append(f"{path.name}:{node.lineno}")
    assert not bad, bad


def test_startup_warms_the_extractor_and_a_failed_warm_up_still_starts(monkeypatch):
    from fastapi.testclient import TestClient

    import tokenmizer.api.app as app_module
    from tokenmizer.graph_memory import hybrid_extractor

    calls = []
    real = hybrid_extractor.get_hybrid_extractor
    monkeypatch.setattr(hybrid_extractor, "get_hybrid_extractor",
                        lambda d=None: (calls.append(d), real(d))[1])
    with TestClient(app_module.app) as c:
        assert c.get("/health").status_code == 200
    assert calls, "the extractor was not warmed at startup"

    def boom(d=None):
        raise RuntimeError("no extractor")
    monkeypatch.setattr(hybrid_extractor, "get_hybrid_extractor", boom)
    with TestClient(app_module.app) as c:
        assert c.get("/health").status_code == 200


# ── Lists under headings, and files named as examples ────────────────────────

def test_errors_listed_under_a_fixed_heading_are_resolved():
    # "**Bugs fixed along the way:**" then bullets: each was a real failure
    # and each is over. Stored as open errors, they told the next session to
    # fix what was already fixed.
    r = _x([{"role": "assistant", "content":
             "**Bugs fixed along the way:**\n"
             "- A race condition in the cache warmer.\n"
             "- To-dos from early in long sessions being silently dropped.\n"}])
    assert len(r.errors) == 2 and sorted(r.resolved_errors) == sorted(r.errors)


def test_items_under_a_problem_heading_are_errors():
    r = _x([{"role": "assistant", "content":
             "Known issues:\n- Pagination returns duplicate rows on page 2\n"
             "- None\n- Should we shard the orders table?\n"}])
    assert r.errors == ["Pagination returns duplicate rows on page 2"]
    assert r.resolved_errors == []


def test_a_generic_problems_heading_needs_a_defect_in_the_item():
    # "Problems I found:" also lists concerns that are not failures.
    r = _x([{"role": "assistant", "content":
             "**Problems I found:**\n"
             "1. **New bug in the product.** It stores adjectives as errors.\n"
             "2. **Your changelog's figure can't be compared with this.**\n"}])
    assert r.errors == ["New bug in the product"], r.errors


def test_non_problem_lists_are_not_errors():
    r = _x([{"role": "assistant", "content":
             "Next steps:\n- Add pagination to the orders endpoint\n- Write the runbook\n"}])
    assert r.errors == []


@pytest.mark.parametrize("text", [
    "Multi-dot names (`vite.config.ts`) and build files with a directory (`fastlane/Fastfile`).",
    "Config files such as `tsconfig.json` are skipped.",
    "Paths like docs/index.md are ignored by the linter.",
])
def test_files_named_as_examples_are_not_files(text):
    assert _x([{"role": "assistant", "content": text}]).files == []


@pytest.mark.parametrize("text,files", [
    ("Updated two files (`src/a.py`, `src/b.py`).", ["src/a.py", "src/b.py"]),
    ("I edited the settings module (config/settings.py).", ["config/settings.py"]),
    ("The fix lives in src/api/orders.py.", ["src/api/orders.py"]),
])
def test_files_that_were_worked_on_are_still_files(text, files):
    assert _x([{"role": "assistant", "content": text}]).files == files


@pytest.mark.parametrize("payload", ["\t" * 8000, "\n" * 8000, " " * 8000, ". " * 4000,
                                     "  without" + " " * 400 + "x" * 300 + " " * 400 + "x" * 300
                                     + ' it is./w/src/  File "x.py", line 1',
                                     ("a.py\t" + " " * 300) * 8 + "!"],
                         ids=["tabs", "blank-lines", "spaces", "dot-space",
                              "names-then-failure", "tabbed-names-then-failure"])
def test_every_extraction_regex_is_linear_on_whitespace(payload):
    """Every compiled pattern in the extraction modules, run directly. The
    full-extraction payload test cannot see a pattern that only ever gets
    short input there (_TRAILING_AUX took 25 s on 20 KB of spaces, and a
    decision pattern 199 s on 17 KB of tabs, before each was fixed)."""
    import re as _re

    from tokenmizer.graph_memory import hybrid_extractor, patterns
    for module in (patterns, hybrid_extractor):
        for name, value in vars(module).items():
            if isinstance(value, _re.Pattern):
                started = time.monotonic()
                list(value.finditer(payload))
                assert time.monotonic() - started < 1.0, name


# ── Found on real agent sessions (SWE-bench trajectories, dev split) ─────────

def test_different_errors_of_one_class_are_both_kept():
    # One label per exception class used to be kept for the whole call, so a
    # later, unrelated ValueError deleted an earlier one.
    errors = HybridExtractor()._drop_restated_errors([
        "ValueError: Imaginary coordinates are not permitted",
        "ValueError: Error from parse_expr with transformed code",
    ])
    assert len(errors) == 2
    # A thin restatement of the same failure is still collapsed.
    assert HybridExtractor()._drop_restated_errors([
        "Login keeps returning 422",
        "Fixed: 422 error — missing email validation in the LoginRequest model",
    ]) == ["Fixed: 422 error — missing email validation in the LoginRequest model"]


@pytest.mark.parametrize("text", [
    "Remember, YOU CAN ONLY ENTER ONE COMMAND AT A TIME. Wait for feedback.",
    "Your changes have NOT been applied.\nDO NOT re-run the same failed edit command.",
    "Let's run it again without encountering any configuration errors.",
    "Let's modify it to raise a ValueError when the left operand is not a matrix.",
    "[File: /r/v.py (900 lines total)]\n587:def f(X):\n588:    \"\"\"Raise a ValueError if X has 64bit indices\n589:    \"\"\"\n",
])
def test_agent_framework_text_is_not_an_error(text):
    assert _x([{"role": "user", "content": text}]).errors == []


@pytest.mark.parametrize("text,label", [
    ("Traceback (most recent call last):\n  File \"x.py\", line 1\n"
     "RuntimeError: Model class __main__.B doesn't declare an explicit app_label",
     "RuntimeError: Model class __main__.B doesn't declare an explicit app_label"),
    ("Traceback (most recent call last):\n  File \"x.py\", line 1\n"
     "django.db.utils.OperationalError: no such table: tests_parent",
     "django.db.utils.OperationalError: no such table: tests_parent"),
    ("Traceback (most recent call last):\n  File \"x.py\", line 1\n"
     "django.core.exceptions.ImproperlyConfigured: Requested setting INSTALLED_APPS, but not set",
     "django.core.exceptions.ImproperlyConfigured: Requested setting INSTALLED_APPS"),
    ("connect ECONNREFUSED 127.0.0.1:5432", "ECONNREFUSED"),
    ("Running it raised a ValueError: invalid literal for int().", "ValueError: invalid literal"),
    ("Please note the build failed twice yesterday.", "build failed twice"),
])
def test_real_failures_keep_their_whole_message(text, label):
    errors = _x([{"role": "user", "content": text}]).errors
    assert any(label in e for e in errors), errors


def test_a_directory_listing_is_not_a_list_of_worked_on_files():
    listing = "lib/matplotlib:\n__init__.py\n_afm.py\n_cm.py\naxes\nfigure.py\ncolors.py\n"
    r = _x([{"role": "user", "content": listing},
            {"role": "assistant", "content": "The bug is in lib/matplotlib/figure.py."}])
    assert r.files == ["lib/matplotlib/figure.py"], r.files


@pytest.mark.parametrize("text,files", [
    ("from django.db import models", []),
    # The name runs on past `.db`: "from" does not make it a file.
    ("from django.db.models import Q", []),
    ("import django.conf", []),
    ("django.db.utils.OperationalError: no such table", []),
    ("Removed the dependency from go.mod.", ["go.mod"]),
    ("Copied data from test.db to prod.db.", ["test.db", "prod.db"]),
])
def test_module_paths_are_not_files(text, files):
    assert _x([{"role": "assistant", "content": text}]).files == files


@pytest.mark.parametrize("text", [
    # Attributes and methods whose last part is a file extension, as they
    # appear in the code SWE-agent and OpenHands sessions display.
    "obj.save(force_insert=True, using=self.db)",
    "connection = connections[self.db]",
    "        self.sql = sql",
    "if not np.allclose(np.abs(1. - np.sum(weights)), 0.):",
    "if (any(np.less(weights, 0.)) or",
    "            if instance._state.db is None:",
    "        db = queryset.db",
    "rows = cache[conn.db]",
])
def test_code_attributes_are_not_files(text):
    assert _x([{"role": "tool", "content": text}]).files == []


@pytest.mark.parametrize("text,files", [
    ("Copied data from test.db to prod.db.", ["test.db", "prod.db"]),
    ("I updated go.sum and pom.xml, and the schema in schema.sql.",
     ["go.sum", "pom.xml", "schema.sql"]),
    # Brackets and shell assignments name files, not attributes.
    ("See [notes.md] for details.", ["notes.md"]),
    ("Linked it from [[design.md]].", ["design.md"]),
    ("Set SCRIPT=build.sh and DB=app.db in the env.", ["build.sh", "app.db"]),
])
def test_files_with_attribute_like_extensions_are_still_files(text, files):
    assert _x([{"role": "assistant", "content": text}]).files == files


def test_a_diff_header_names_the_file_once_without_its_side_prefix():
    diff = ("diff --git a/sympy/solvers/polysys.py b/sympy/solvers/polysys.py\n"
            "index b9809fd4e9..674322d4eb 100644\n"
            "--- a/sympy/solvers/polysys.py\n+++ b/sympy/solvers/polysys.py\n"
            "@@ -240,7 +240,7 @@ def _solve_reduced_system(system, gens, entry=False):\n")
    assert _x([{"role": "assistant", "content": diff}]).files == ["sympy/solvers/polysys.py"]


def test_a_path_that_starts_with_a_or_b_outside_a_diff_keeps_it():
    assert _x([{"role": "assistant", "content": "The fix is in b/handlers.py."}]).files == ["b/handlers.py"]


@pytest.mark.parametrize("text,files", [
    ("Added the key to .env and .env.local, and ignored both in .gitignore.",
     [".env", ".env.local", ".gitignore"]),
    ("See config/.env.", ["config/.env"]),
    ("pyobj_role = self.env.get_domain('py')", []),
])
def test_bare_dotfiles_are_files(text, files):
    assert _x([{"role": "assistant", "content": text}]).files == files


@pytest.mark.parametrize("name", ["Pipfile", "Podfile", "Containerfile", "Tiltfile", ".gitignore"])
def test_every_file_the_extractor_emits_survives_validation(tmp_path, name):
    # The validator kept its own copy of the extensionless names; it lacked
    # these, and dropped them after the extractor had found them.
    g = GraphMemory("s", storage_dir=str(tmp_path))
    g.extract_from_messages([{"role": "user", "content": "x"},
                             {"role": "assistant", "content": f"I updated the {name}."}],
                            incremental=False)
    assert name in [n.label for n in g._nodes.values() if n.type == NodeType.FILE]


def test_resume_lists_the_most_recent_open_errors(tmp_path):
    g = GraphMemory("s", storage_dir=str(tmp_path))
    for i in range(6):
        g.extract_from_messages([{"role": "user", "content":
            f"Traceback (most recent call last):\n  File \"x.py\", line 1\n"
            f"KeyError: 'field_{i}'"}], incremental=False)
    block = g.to_context_block(token_budget=400)
    assert "field_5" in block and "field_0" not in block, block


def test_a_columnar_listing_with_crlf_is_not_a_list_of_files():
    # Plain `ls` prints names in columns; tool output often ends lines with
    # \r\n. One such `ls -R` in an OpenHands session gave 906 FILE nodes.
    listing = ("ls -R /workspace/repo\r\n/workspace/repo:\r\n"
               "CHANGES.rst\t    appveyor.yml      licenses\r\n"
               "CITATION\t    astropy\t      pip-requirements\r\n"
               "CODE_OF_CONDUCT.md  astropy.egg-info  setup.py\r\n"
               "CONTRIBUTING.md     conftest.py       tox.ini\r\n"
               "LICENSE.rst\t    docs\t      README.rst\r\n")
    assert _x([{"role": "tool", "content": listing}]).files == []


def test_a_file_changed_by_a_tool_call_leads_the_resume(tmp_path):
    g = GraphMemory("s", storage_dir=str(tmp_path))
    msgs = [
        {"role": "user", "content": "Context: a.py b.py c.py d.py e.py f.py g.py h.py i.py j.py k.py."},
        {"role": "assistant", "content": "Editing.", "tool_calls": [{
            "id": "1", "type": "function", "function": {
                "name": "str_replace_editor",
                "arguments": json.dumps({"command": "str_replace", "path": "/w/src/core.py",
                                         "old_str": "a", "new_str": "b"})}}]},
    ]
    for k in (1, 2):
        g.extract_from_messages(msgs[:k])
    files_line = next(ln for ln in g.to_context_block(400).splitlines() if ln.startswith("Files:"))
    assert files_line.startswith("Files: /w/src/core.py"), files_line
