from tokenmizer.dashboard.page import DASHBOARD_HTML


def test_dashboard_retries_stats_requests_with_session_api_key():
    assert "sessionStorage.getItem(API_KEY_STORAGE_KEY)" in DASHBOARD_HTML
    assert "sessionStorage.setItem(API_KEY_STORAGE_KEY, key)" in DASHBOARD_HTML
    assert "Authorization: `Bearer ${apiKey}`" in DASHBOARD_HTML
    assert "response.status !== 401" in DASHBOARD_HTML

    assert "dashboardFetch('/api/stats')" in DASHBOARD_HTML
    assert "dashboardFetch('/api/cache/stats')" in DASHBOARD_HTML
    assert "fetch('/api/stats')" not in DASHBOARD_HTML
    assert "fetch('/api/cache/stats')" not in DASHBOARD_HTML


def test_dashboard_shares_a_single_api_key_prompt_between_requests():
    assert "let apiKeyPrompt = null;" in DASHBOARD_HTML
    assert "if (!apiKeyPrompt)" in DASHBOARD_HTML
    assert "window.prompt(" in DASHBOARD_HTML
    assert "apiKeyPrompt = null;" in DASHBOARD_HTML


def test_graph_link_uses_bearer_when_key_is_stored():
    """The session list linked to /api/graph/{id}/html with a plain
    anchor. Every other dashboard request goes through dashboardFetch
    with the stored Bearer key, but a link cannot carry a header, so
    with api_key set the graph page answered 401 straight from the
    dashboard. The link stays a real href (dev mode, middle-click,
    copy-link) and a click handler takes over only when a key is
    stored: open the tab first, then fetch with the key and hand the
    tab the page by writing the document (no blob: navigation, which
    embedded browsers refuse)."""
    assert 'href="/api/graph/' in DASHBOARD_HTML
    assert "openGraph" in DASHBOARD_HTML
    assert "doc.write(html)" in DASHBOARD_HTML
    # The tab must be opened synchronously inside the click, before any
    # await, or the user gesture is gone and popup blocking eats it.
    assert "window.open('', '_blank')" in DASHBOARD_HTML


class TestDashboardShowsRealState:
    """The dashboard's two largest cards used to be hard-coded: an example
    resume block and an example session legend. They said exactly the same
    thing on a fresh install and on a deployment with a thousand sessions,
    which is worse than showing nothing — a reader takes them for data.
    """

    def test_no_hardcoded_example_resume(self):
        for literal in ("Build FastAPI auth service with JWT + PostgreSQL",
                        "Refresh token rotation | Rate limiting",
                        "~280 tokens"):
            assert literal not in DASHBOARD_HTML, literal

    def test_resume_block_is_fetched_per_session(self):
        assert "dashboardFetch('/api/resume/' + encodeURIComponent(id))" in DASHBOARD_HTML
        assert "d.resume_context" in DASHBOARD_HTML
        # The source matters: a block built live from the graph and one
        # replayed from an old checkpoint are different claims.
        assert "d.source === 'live_graph'" in DASHBOARD_HTML

    def test_health_is_polled_and_rendered(self):
        assert "fetch('/health')" in DASHBOARD_HTML
        assert "renderHealth" in DASHBOARD_HTML
        assert "persist_failures" in DASHBOARD_HTML
        assert "sessions_with_unreadable_graph" in DASHBOARD_HTML
        # /health takes no API key, so it must not go through dashboardFetch
        # (which would prompt for one on a 401 that cannot happen).
        assert "dashboardFetch('/health')" not in DASHBOARD_HTML

    def test_graph_preview_uses_srcdoc_not_a_second_request(self):
        """The graph page is self-contained; srcdoc means the iframe makes
        no request of its own, so it needs no credential inside the frame."""
        assert "frame.srcdoc" in DASHBOARD_HTML
        assert "loadPreview" in DASHBOARD_HTML

    def test_empty_state_tells_the_reader_what_to_do(self):
        assert "getting-started" in DASHBOARD_HTML
        assert "POST /api/checkpoint?session_id=" in DASHBOARD_HTML
        assert "use_llm_extraction" in DASHBOARD_HTML

    def test_selecting_a_session_drives_both_panels(self):
        assert "function selectSession" in DASHBOARD_HTML
        assert "loadResume(id)" in DASHBOARD_HTML and "loadPreview(id)" in DASHBOARD_HTML
