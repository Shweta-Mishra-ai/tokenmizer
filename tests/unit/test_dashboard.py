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
