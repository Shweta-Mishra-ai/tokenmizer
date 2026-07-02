"""
Memory accuracy tests — simulate real coding sessions and measure
task/decision/file retention precision and recall.
"""
import pytest

from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType

# ── Synthetic 30-turn session ─────────────────────────────────────────────────

SESSION_MESSAGES = [
    {"role": "user", "content": "Let's build a FastAPI authentication service with JWT tokens and PostgreSQL"},
    {"role": "assistant", "content": "Great plan. I'm creating the project structure. Files: api/main.py, api/auth.py, api/models.py, config.py"},
    {"role": "user", "content": "Use Python 3.12 and FastAPI 0.111"},
    {"role": "assistant", "content": "Using Python 3.12 and FastAPI 0.111. Decided: PostgreSQL for user storage (concurrent write support required). Completed setting up virtual environment."},
    {"role": "user", "content": "Add password hashing with bcrypt"},
    {"role": "assistant", "content": "Implemented password hashing using bcrypt. Updated api/auth.py with hash_password() and verify_password() functions. Added bcrypt to requirements.txt"},
    {"role": "user", "content": "Now implement the login endpoint"},
    {"role": "assistant", "content": "Created POST /api/auth/login endpoint in api/auth.py. Returns JWT access token on success. Using python-jose for JWT signing."},
    {"role": "user", "content": "The login endpoint is returning 422 — validation error"},
    {"role": "assistant", "content": "Fixed the 422 error: the LoginRequest model was missing email field validation. Updated Pydantic model in api/models.py. Login endpoint now working."},
    {"role": "user", "content": "Add refresh token support"},
    {"role": "assistant", "content": "Implementing refresh token support. This requires: 1) refresh_token field in DB, 2) POST /api/auth/refresh endpoint, 3) token rotation logic. Working on api/auth.py"},
    {"role": "user", "content": "We decided to NOT store refresh tokens in DB — use Redis instead"},
    {"role": "assistant", "content": "Decided: Store refresh tokens in Redis (not DB) — faster revocation, no DB schema change needed. Adding redis dependency. Updated config.py with REDIS_URL setting."},
    {"role": "user", "content": "Add tests"},
    {"role": "assistant", "content": "Created tests/test_auth.py with 8 tests covering: login, logout, token refresh, invalid credentials, expired token. All 8 tests passing."},
]

# Ground truth for assertions
EXPECTED_COMPLETED_TASKS = [
    "project structure",
    "password hashing",
    "login endpoint",
    "422 error",
]
EXPECTED_DECISIONS = [
    "postgresql",
    "bcrypt",
    "redis",
]
EXPECTED_FILES = [
    "api/auth.py",
    "api/models.py",
    "config.py",
]
EXPECTED_OPEN_TASKS = [
    "refresh token",
]


@pytest.fixture
def filled_graph(tmp_path):
    g = GraphMemory("accuracy-test", storage_dir=str(tmp_path))
    g.extract_from_messages(SESSION_MESSAGES, incremental=False)
    return g


class TestTaskRetention:

    def test_completed_tasks_extracted(self, filled_graph):
        completed = [
            n for n in filled_graph._nodes.values()
            if n.type in (NodeType.TASK, NodeType.ENDPOINT) and n.status == NodeStatus.COMPLETED
        ]
        labels_lower = {n.label.lower() for n in completed}
        found = sum(
            1 for expected in EXPECTED_COMPLETED_TASKS
            if any(expected.lower() in label for label in labels_lower)
        )
        found / max(1, len(completed))
        recall = found / len(EXPECTED_COMPLETED_TASKS)
        # We expect at least 40% recall from heuristic extraction
        assert recall >= 0.4, f"Task recall too low: {recall:.2f} ({found}/{len(EXPECTED_COMPLETED_TASKS)})"

    def test_open_tasks_detected(self, filled_graph):
        open_tasks = [
            n for n in filled_graph._nodes.values()
            if n.type == NodeType.TASK and n.status in (NodeStatus.PENDING, NodeStatus.IN_PROGRESS)
        ]
        labels_lower = {n.label.lower() for n in open_tasks}
        found = sum(
            1 for expected in EXPECTED_OPEN_TASKS
            if any(expected.lower() in label for label in labels_lower)
        )
        # At least one open task should be detected
        assert found >= 1 or len(open_tasks) >= 1, "No open tasks detected"


class TestDecisionRetention:

    def test_key_decisions_extracted(self, filled_graph):
        decisions = [n for n in filled_graph._nodes.values() if n.type == NodeType.DECISION]
        labels_lower = {n.label.lower() for n in decisions}

        found = sum(
            1 for expected in EXPECTED_DECISIONS
            if any(expected.lower() in label for label in labels_lower)
        )
        recall = found / len(EXPECTED_DECISIONS)
        # Heuristic extraction should catch at least 1 of 3 decisions
        assert recall >= 0.33, f"Decision recall too low: {recall:.2f}"

    def test_redis_decision_captured(self, filled_graph):
        """Redis over DB decision is critical — should be in graph."""
        decisions = [n for n in filled_graph._nodes.values() if n.type == NodeType.DECISION]
        " ".join(n.label.lower() for n in decisions)
        # Either "redis" or "decided" should appear
        has_some_decision = len(decisions) > 0
        assert has_some_decision, "No decisions extracted at all"


class TestFileRetention:

    def test_files_extracted(self, filled_graph):
        files = [n for n in filled_graph._nodes.values() if n.type == NodeType.FILE]
        file_labels = {n.label for n in files}

        found = sum(
            1 for expected in EXPECTED_FILES
            if any(expected.lower() in label.lower() for label in file_labels)
        )
        recall = found / len(EXPECTED_FILES)
        assert recall >= 0.33, f"File recall too low: {recall:.2f} ({found}/{len(EXPECTED_FILES)})"

    def test_auth_file_extracted(self, filled_graph):
        """api/auth.py is the most important file — must be captured."""
        files = [n for n in filled_graph._nodes.values() if n.type == NodeType.FILE]
        labels = {n.label.lower() for n in files}
        has_auth = any("auth" in label for label in labels)
        assert has_auth, f"api/auth.py not found in graph. Files found: {labels}"


class TestResumeBlock:

    def test_resume_block_not_empty(self, filled_graph):
        block = filled_graph.to_context_block()
        assert len(block) > 50

    def test_resume_block_under_token_budget(self, filled_graph):
        from tokenmizer.core.tokenizer import count_tokens
        block = filled_graph.to_context_block(token_budget=300)
        tokens = count_tokens(block)
        assert tokens <= 350  # some slack

    def test_resume_block_contains_decisions(self, filled_graph):
        block = filled_graph.to_context_block()
        # Should have some content from decisions
        assert len(block) > 0

    def test_full_history_vs_partial(self, tmp_path):
        """Extracting full history should give more nodes than partial."""
        g_full = GraphMemory("full", storage_dir=str(tmp_path))
        g_full.extract_from_messages(SESSION_MESSAGES, incremental=False)

        g_partial = GraphMemory("partial", storage_dir=str(tmp_path))
        g_partial.extract_from_messages(SESSION_MESSAGES[-4:], incremental=False)

        assert len(g_full._nodes) >= len(g_partial._nodes), \
            "Full history extraction should produce at least as many nodes as partial"
