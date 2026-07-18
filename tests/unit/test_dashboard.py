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
