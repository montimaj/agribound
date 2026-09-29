"""The Release and Python badges in README.md and docs/index.md match the package.

The Release badge must name ``agribound.__version__`` and the Python badge the
floor of ``requires-python`` in ``pyproject.toml``. These tests read the source
checkout and are skipped when it is not available, e.g. when the suite runs
from an installed wheel.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

import pytest

import agribound

ROOT = Path(__file__).resolve().parents[2]
PAGES = ("README.md", "docs/index.md")

#: A linked static shields.io badge: ``[![alt](https://img.shields.io/badge/...)](href)``.
_BADGE = re.compile(
    r"\[!\[(?P<alt>[^\]]*)\]\((?P<src>https://img\.shields\.io/badge/[^)\s]+)\)\]\([^)\s]+\)"
)
#: A single dash separates the fields of a static badge; "--" is a literal dash.
_FIELD_SEPARATOR = re.compile(r"(?<!-)-(?!-)")
#: The lower bound of a ``requires-python`` specifier, e.g. ``>=3.12``.
_PYTHON_FLOOR = re.compile(r">=\s*(\d+(?:\.\d+)*)")


def _unescape(field: str) -> str:
    """Undo the shields.io escapes: ``--`` is ``-``, ``__`` is ``_``, ``_`` is a space."""
    field = field.replace("--", "\0").replace("__", "\1").replace("_", " ")
    return unquote(field.replace("\0", "-").replace("\1", "_"))


def _badge_fields(src: str) -> list[str]:
    """Label, message and colour of a static shields.io badge URL, unescaped."""
    path = src.removeprefix("https://img.shields.io/badge/").split("?", 1)[0]
    return [_unescape(field) for field in _FIELD_SEPARATOR.split(path.removesuffix(".svg"))]


def _badge(page: str, label: str) -> tuple[str, str]:
    """Alt text and message of the one badge on ``page`` whose label is ``label``."""
    path = ROOT / page
    if not path.is_file():
        pytest.skip(f"{page} is not available")
    found = []
    for match in _BADGE.finditer(path.read_text(encoding="utf-8")):
        fields = _badge_fields(match.group("src"))
        if len(fields) == 3 and fields[0].lower() == label:
            found.append((match.group("alt"), fields[1]))
    assert len(found) == 1, f"{page}: expected one {label!r} badge, found {len(found)}"
    return found[0]


def _requires_python_floor() -> str:
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml is not available")
    spec = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["requires-python"]
    floors = _PYTHON_FLOOR.findall(spec)
    assert len(floors) == 1, f"requires-python {spec!r} should have one '>=' lower bound"
    return floors[0]


@pytest.mark.parametrize(
    ("src", "fields"),
    [
        (
            "https://img.shields.io/badge/release-v1.0.0-green.svg",
            ["release", "v1.0.0", "green"],
        ),
        ("https://img.shields.io/badge/python-3.12%2B-blue", ["python", "3.12+", "blue"]),
        (
            "https://img.shields.io/badge/license-Apache--2.0-green",
            ["license", "Apache-2.0", "green"],
        ),
        (
            "https://img.shields.io/badge/docs-GitHub%20Pages-blue",
            ["docs", "GitHub Pages", "blue"],
        ),
        ("https://img.shields.io/badge/a_b__c-x-red", ["a b_c", "x", "red"]),
    ],
)
def test_badge_fields_are_parsed(src, fields):
    assert _badge_fields(src) == fields


@pytest.mark.parametrize("page", PAGES)
def test_release_badge_matches_package_version(page):
    alt, message = _badge(page, "release")
    assert alt == "Release"
    assert message == f"v{agribound.__version__}", (
        f"{page}: the Release badge says {message!r}; agribound.__version__ is "
        f"{agribound.__version__!r}"
    )


@pytest.mark.parametrize("page", PAGES)
def test_python_badge_matches_requires_python(page):
    floor = _requires_python_floor()
    alt, message = _badge(page, "python")
    assert message == f"{floor}+", (
        f"{page}: the Python badge says {message!r}; requires-python starts at {floor}"
    )
    assert alt == f"Python {floor}+"
