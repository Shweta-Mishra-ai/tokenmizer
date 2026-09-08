"""Deterministic session IDs — see tokenmizer/core/session.py."""
from __future__ import annotations

import pytest

from tokenmizer.core.session import MAX_LENGTH, default_session_id, slugify


@pytest.mark.parametrize("name,expected", [
    ("auth-service", "auth-service"),
    ("Auth Service", "auth-service"),
    ("my_project", "my-project"),
    ("  Spaced  Name  ", "spaced-name"),
    ("weird!!chars@@here", "weird-chars-here"),
    ("--leading-and-trailing--", "leading-and-trailing"),
    ("CamelCase", "camelcase"),
    ("v2.1-api", "v2-1-api"),
])
def test_slugify(name, expected):
    assert slugify(name) == expected


def test_slugify_bounds_length_at_a_word_boundary():
    slug = slugify("a-very-long-project-name-that-keeps-going-and-going-forever")
    assert len(slug) <= MAX_LENGTH
    assert not slug.endswith("-")
    assert "-" in slug and slug in "a-very-long-project-name-that-keeps-going-and-going-forever"


def test_slugify_truncates_a_single_long_word_without_a_boundary():
    slug = slugify("x" * 200)
    assert slug == "x" * MAX_LENGTH


def test_default_is_the_directory_name(tmp_path):
    project = tmp_path / "Auth Service"
    project.mkdir()
    assert default_session_id(project) == "auth-service"


@pytest.mark.parametrize("name", ["tmp", "temp", "src", "code", "Downloads"])
def test_generic_directories_have_no_default(tmp_path, name):
    """A generic folder name is not a project identity — every unrelated
    project under it would collide on one session. Callers ask instead, which
    is the one case where asking is right."""
    d = tmp_path / name
    d.mkdir()
    assert default_session_id(d) is None


def test_filesystem_root_has_no_default():
    import pathlib
    root = pathlib.Path(pathlib.Path.cwd().anchor)
    assert default_session_id(root) is None


def test_default_is_stable_across_calls(tmp_path):
    """The whole point: the same project yields the same ID every session,
    so a later resume finds the earlier checkpoint."""
    project = tmp_path / "order-pipeline"
    project.mkdir()
    assert default_session_id(project) == default_session_id(project)


def test_derived_id_is_safe_in_a_url_and_a_db_key(tmp_path):
    project = tmp_path / "My Project (v2)!"
    project.mkdir()
    derived = default_session_id(project)
    assert derived is not None
    assert all(c.isalnum() or c == "-" for c in derived), derived
