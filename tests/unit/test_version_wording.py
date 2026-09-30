"""Docstrings, help texts and docs name the release lines as 0.1.x and 1.0.

The release before 1.0.0 is the 0.1.x line (0.1.0-0.1.3). Calling it 1.x ("the
agribound 1.x vote rule", "1.x had no tolerance") or v1 ("the v1 rule") was
wrong; "1.x" on its own ("the 1.x line") is the current line and passes. Any
name for a release line that does not exist yet is also wrong (pre-release
drafts numbered this release 2.0.0): "agribound 2.0.0", "agribound v2",
"agribound==2.0.0", "Changed in (version) 2.0", "In 2.0, ...", "since 2.0",
"the 2.0 release", "the v2.0 release notes", "the 2.x line", and a bare "v2" or
"v2.0" such as "not measured for v2", "In v2, ..." or "(v2 layout)". The
current release line is written "1.0" (this release "1.0.1", the first "1.0.0").

Other projects' versions are legitimate. A bare "v2" or "v2.0" passes when a
product name stands just before it (TESSERA, Delineate Anything, DA, FTW, HLS,
Sentinel-2, SAM, Prithvi, or an internal schema, index, format or recipe
revision), when "v1" or "v1.1" is on the same or an adjacent line (a list of
dataset or checkpoint versions), when "beta", "embeddings", "models" or "Zarr"
follows it (TESSERA v2), and when it is a quoted literal such as ``"v2"``. A
"2.0" or "2.x" without a "v" counts only in a release context (after "in",
"since", "Changed in", "upgrading to" and the like, or after "the"/"a"/"to"
and before "release", "line", "composites" and the like), so numbers ("default
2.0", "set to 2.0", "in 2.0 m steps") and other packages' lines ("ftw-tools
2.x", "SAM 2.0", "Apache License, Version 2.0", "shapely 2.0.x") pass. A
version glued to other text is never matched: ``large_v2``, ``FTW_v2_*``,
``v2-2B-L~beta1``, ``lychee-action@v2``, ``/v1/messages``, Prithvi-EO-2.0,
ftw-tools 2.0.0b5 and ``>=2.0.0b5``. In Python sources only comments and string
literals are checked, so a variable named ``v2`` is not wording. Notebooks are
checked in cell sources and text outputs. Internal format constants such as
``CACHE_SCHEMA_VERSION = "2"`` are schema revisions, not the release.
"""

from __future__ import annotations

import io
import json
import re
import tokenize
from pathlib import Path

import pytest
from click.testing import CliRunner

import agribound
from agribound.cli import main

PACKAGE_DIR = Path(agribound.__file__).parent
ROOT = PACKAGE_DIR.parent
THIS_FILE = Path(__file__).resolve()

#: "1.x" not preceded by a digit or dot (so "0.1.x" passes) nor by "anthropic ".
_ONE_X_TOKEN = r"(?<![\d.])(?<!anthropic )1\.x\b"
#: "1.x" used for the 0.1.x behaviour: before a rule/behaviour noun ("the 1.x vote
#: rule") or a past-tense verb ("1.x had"), or after "as in"/"from"/"unlike". "1.x"
#: on its own is the current line ("the 1.x line") and passes.
_ONE_X = re.compile(
    _ONE_X_TOKEN
    + r"(?=\s+(?:[\w-]+\s+)?(?:rules?|behaviou?r|definitions?|keys?|grid|conventions?)\b"
    r"|\s+(?:had|was|were|did|used|returned|rounded|kept|ignored|wrote|applied|matched"
    r"|required|dropped|treated|stored|computed)\b)"
    r"|\b(?:as in|from|unlike|than in) (?:agribound )?" + _ONE_X_TOKEN,
    re.IGNORECASE,
)
#: Phrases that used "v1" for the previous agribound release.
_V1_RELEASE = re.compile(
    r"\bv1 (?:behaviou?r|rules?|definitions?|keys?|grid|conventions?)\b"
    r"|\b(?:as in|from|in) v1\b(?![./])",
    re.IGNORECASE,
)
#: TESSERA dataset versions ("in v1 (and v1.1 ...") are legitimate.
_TESSERA = re.compile(r"tessera|v1\.1", re.IGNORECASE)

#: Numbered names of an agribound 2 release: "2.0.0" on its own (not "2.0.0b5",
#: not another package's specifier such as "geedim>=2.0.0"), "agribound 2",
#: "agribound v2", "agribound 2.x" and "agribound-v2", an agribound requirement
#: on 2 ("agribound==2.0.0", "agribound[gfm]>=2.0"; an upper bound "<2" passes),
#: "Changed/Added/New in (version) 2.0" and the old migration page name.
_V2_NUMBER = re.compile(
    r"(?<![\w.=<>~^])v?2\.0\.0(?!\w|\.\d)"
    r"|\bagribound[ -](?:version |release )?v?2(?:\.[\dx]+)*\b"
    r"|\bagribound(?:\[[^\]]*\])?\s*(?:===?|>=?|~=)\s*v?2(?:\.[\dx]+)*\b"
    r"|\b(?:added|changed|new|deprecated|removed|introduced) in (?:version |v)?2(?:\.(?:0|x))*\b"
    r"(?![.\-]\w)"
    r"|\bmigration-v2\b",
    re.IGNORECASE,
)
#: A bare "v2", "v2.0" or "version 2" token: not glued to a word, path, "@" or
#: "-" before it, nor followed by ".0", "-2B", "_x" and the like.
_BARE_V2 = re.compile(r"(?<![\w.\-/@~])(?:v2(?:\.0)?|version 2)\b(?![.\-_]\w)", re.IGNORECASE)
#: Text just before a bare "v2" that names whose version it is.
_V2_OWNER = re.compile(
    r"(?:tessera|delineate[\s-]*anything|\bda|\bftw|\bhls|sentinel-2|\bsam|prithvi(?:-eo)?"
    r"|schema|index|format|recipe)[\s-]+$",
    re.IGNORECASE,
)
#: A "2.0", "2.x", "v2.0" or "version 2.0" token (not "2.0.0b5", "2.05",
#: "Prithvi-EO-2.0", "sam2.1" or ">=2.0"). Whether it names a release depends on
#: the text around it (``_v2_line_in_context``).
_V2_LINE = re.compile(r"(?<![\w.\-/@~=<>^])(?:version |v)?2\.(?:0|x)(?![.\-]?\w)", re.IGNORECASE)
#: Just before the token: a release cue ("In 2.0, ...", "since 2.0", "Changed in
#: version 2.0", "upgrading to 2.0").
_V2_LINE_CUE = re.compile(
    r"\b(?:in|since|before|after|until|as of|prior to|starting (?:with|in)"
    r"|(?:upgrad|migrat)\w*\s+(?:to|from)(?:\s+the)?)\s+$",
    re.IGNORECASE,
)
#: Just after the token: a quantity, so "in 2.0 m steps" or "after 2.0 s" is a number.
_V2_LINE_QUANTITY = re.compile(
    r"\s*(?:%|°|×|(?:to|and|or|-|–)\s*\d"
    r"|(?:m|km|cm|mm|ha|px|s|ms|min|h|x|deg|pixels?|metres?|meters?|seconds?|minutes?"
    r"|hours?|degrees?|times|gb|mb|gib|mib)\b)",
    re.IGNORECASE,
)
#: "for" counts only with the token at the end of a clause: "(not measured for 2.0)".
_V2_LINE_FOR = re.compile(r"\bfor\s+$", re.IGNORECASE)
_CLAUSE_END = re.compile(r"\s*(?:[),;:!?]|\.(?!\d)|$)")
#: Just before the token: a function word or the start of a sentence, not a
#: product name ("the 2.0 release" but not "SAM 2.0 checkpoints", "ftw-tools 2.x").
_V2_LINE_LEAD = re.compile(
    r"(?:^|[.!?:;(\[\"'`*]|\b(?:the|a|an|any|every|each|our|this|that|next|its|upcoming"
    r"|future|to|from|in|on|of|for|with|by|since|before|after|until|under))\s*$",
    re.IGNORECASE,
)
#: Just after the token: a noun that makes it a release name ("the 2.0 release",
#: "the 2.x line", "2.0 composites", "the 2.0 harmonisation").
_V2_LINE_NOUN = re.compile(
    r"\s+(?:releases?|release notes|notes|line|series|branch|migration|upgrade|api|changes"
    r"|changelog|defaults|behaviou?r|composites?|outputs?|caches?|layout|conventions?"
    r"|contract|harmoni[sz]ation|helpers?|users?|docs|documentation|wording|rules?)\b",
    re.IGNORECASE,
)
#: An earlier version of the same series nearby: "v1 | v1.1 | v2", "for v1, ... for v2".
_V1_SERIES = re.compile(r"(?<![\w/@.\-])v1(?:\.1)?\b(?![/\-]\w)", re.IGNORECASE)
#: Text just after a bare "v2" that makes it TESSERA's ("v2 is a sparse beta").
_V2_TAIL = re.compile(
    r"^\W{0,3}(?:is an?\s+)?(?:sparse\s+)?beta\b|^\s+(?:embeddings?|models?|zarr)\b",
    re.IGNORECASE,
)

#: Source-checkout text files outside the package that could name the release
#: (docs, examples, notebooks, tests, tools, metadata). Missing files are
#: skipped (e.g. an installed wheel).
_DOC_GLOBS = (
    "README.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "DISCLAIMER.md",
    "MANIFEST.in",
    "mkdocs.yml",
    "pyproject.toml",
    "environment*.yml",
    ".lycheeignore",
    "docs/**/*.md",
    "docs/**/*.yml",
    "examples/**/*.py",
    "examples/**/*.ipynb",
    "examples/**/*.md",
    "examples/**/*.sh",
    "examples/**/*.sbatch",
    "examples/**/*.env",
    "examples/**/*.yaml",
    "examples/**/*.yml",
    "tests/**/*.py",
    "tools/*.py",
    ".github/**/*.yml",
)


def _offenders(paths, check) -> list[str]:
    found = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if check(line):
                found.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    return found


def _package_files(package_dir: Path = PACKAGE_DIR) -> list[Path]:
    return sorted(package_dir.rglob("*.py"))


def _doc_files(root: Path = ROOT) -> list[Path]:
    files = {p for pattern in _DOC_GLOBS for p in root.glob(pattern) if p.is_file()}
    return sorted(p for p in files if p.resolve() != THIS_FILE)


def _previous_line_wrong(line: str) -> bool:
    return bool(_ONE_X.search(line) or (_V1_RELEASE.search(line) and not _TESSERA.search(line)))


def _other_v2(line: str, match: re.Match, prev: str = "", nxt: str = "") -> bool:
    """Whether the bare "v2" *match* in *line* is another project's version."""
    start, end = match.span()
    before, after = line[:start], line[end:]
    if before and before[-1] in "\"'`" and after[:1] == before[-1]:
        return True  # a quoted literal: "v2", 'v2', ``v2``
    if _V2_OWNER.search(f"{prev} {before}") or _V2_TAIL.search(after):
        return True
    return any(_V1_SERIES.search(text) for text in (prev, line, nxt))


def _v2_line_in_context(line: str, match: re.Match, prev: str = "") -> bool:
    """Whether the "2.0"/"2.x" *match* in *line* names a release line."""
    before, after = line[: match.start()], line[match.end() :]
    context = f"{prev} {before}"  # a token at the start of a line continues *prev*
    if _V2_LINE_CUE.search(context) and not _V2_LINE_QUANTITY.match(after):
        return True
    if _V2_LINE_FOR.search(context) and _CLAUSE_END.match(after):
        return True
    if _V2_LINE_LEAD.search(context):
        return match.group().lower().endswith("x") or bool(_V2_LINE_NOUN.match(after))
    return False


def _v2_release_wording(line: str, prev: str = "", nxt: str = "") -> bool:
    """Whether *line* names an agribound 2 release (neighbouring lines give context)."""
    if _V2_NUMBER.search(line):
        return True
    if any(_v2_line_in_context(line, m, prev) for m in _V2_LINE.finditer(line)):
        return True
    return any(not _other_v2(line, m, prev, nxt) for m in _BARE_V2.finditer(line))


def _python_prose(source: str) -> dict[int, str] | None:
    """Comments and string literals of Python *source* by line (None if it does not tokenize)."""
    kinds = {tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)}
    lines: dict[int, list[str]] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in kinds:
                for offset, part in enumerate(tok.string.split("\n")):
                    lines.setdefault(tok.start[0] + offset, []).append(part)
    except (tokenize.TokenError, SyntaxError):
        return None
    return {number: " ".join(parts) for number, parts in lines.items()}


def _lines(text: str, python: bool = False) -> dict[int, str]:
    """{line number: checked text}: prose only for Python that tokenizes, else every line."""
    prose = _python_prose(text) if python else None
    return prose if prose is not None else dict(enumerate(text.splitlines(), 1))


def _joined(text) -> str:
    """A notebook text field, which may be a list of lines."""
    return "".join(text) if isinstance(text, list) else (text or "")


#: Notebook output fields that hold text (stream output and plain/Markdown results).
_OUTPUT_MIME = ("text/plain", "text/markdown")


def _text_blocks(path: Path) -> list[tuple[str, dict[int, str]]]:
    """The checked text of *path*: (location prefix, {line number: text}) blocks."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        blocks = []
        for index, cell in enumerate(json.loads(text).get("cells", []), 1):
            source = _joined(cell.get("source", ""))
            blocks.append((f"cell {index}, line ", _lines(source, cell.get("cell_type") == "code")))
            for number, output in enumerate(cell.get("outputs") or [], 1):
                data = output.get("data") or {}
                parts = [output.get("text")] + [data.get(mime) for mime in _OUTPUT_MIME]
                shown = "\n".join(_joined(part) for part in parts if part)
                if shown:
                    blocks.append((f"cell {index}, output {number}, line ", _lines(shown)))
        return blocks
    return [("", _lines(text, path.suffix == ".py"))]


def _v2_release_offenders(paths, root: Path = ROOT) -> list[str]:
    found = []
    for path in paths:
        for prefix, lines in _text_blocks(path):
            for number, line in sorted(lines.items()):
                prev, nxt = lines.get(number - 1, ""), lines.get(number + 1, "")
                if _v2_release_wording(line, prev, nxt):
                    found.append(f"{path.relative_to(root)}:{prefix}{number}: {line.strip()}")
    return found


def test_no_agribound_1x_or_v1_release_wording():
    offenders = _offenders(_package_files(), _previous_line_wrong)
    assert not offenders, (
        "use '0.1.x' for the previous release line ('1.0' for this one):\n" + "\n".join(offenders)
    )


def test_no_agribound_2_release_wording_in_package():
    offenders = _v2_release_offenders(_package_files())
    assert not offenders, "this release is 1.0 / 1.0.1:\n" + "\n".join(offenders)


def test_no_agribound_2_release_wording_in_docs():
    files = _doc_files()
    if not files:
        pytest.skip("the source checkout (docs/, examples/) is not available")
    offenders = _v2_release_offenders(files)
    assert not offenders, "this release is 1.0 / 1.0.1:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "relative",
    [
        "pyproject.toml",
        "environment.yml",
        "examples/notebooks/01_new_mexico_landsat_timeseries.ipynb",
        "examples/regions/run_namoi_catchment_au.sh",
        "examples/hpc/profiles/generic.env",
        "tests/unit/test_repro.py",
        "tools/make_workflow_diagram.py",
    ],
)
def test_doc_globs_cover_notebooks_tests_tools_and_metadata(relative):
    path = ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} is not in this checkout")
    assert path in _doc_files()


def test_doc_globs_skip_this_module():
    # It quotes the forbidden wording as test data.
    assert THIS_FILE not in {p.resolve() for p in _doc_files()}


def test_patterns_catch_the_old_wording():
    assert _ONE_X.search("the agribound 1.x vote rule")
    assert _ONE_X.search("    1.x had no tolerance")
    assert not _ONE_X.search("the agribound 0.1.x vote rule")
    assert not _ONE_X.search("anthropic 1.x removed them")
    assert not _ONE_X.search("agribound 1.0.0 fixes it; in 1.0 the default is one_to_one")
    assert _ONE_X.search("many_to_one (the 1.x rule; one prediction")
    assert _ONE_X.search("keys kept from agribound 1.x")
    # 1.x on its own is the current line.
    assert not _ONE_X.search("deprecated names are kept for the 1.x line")
    assert not _ONE_X.search("stable across agribound 1.x releases")
    assert _V1_RELEASE.search("many_to_one (the v1 rule; one prediction")
    assert _V1_RELEASE.search("Keys kept from v1 (")
    assert not _V1_RELEASE.search("for v1 models (ftw-tools default)")


@pytest.mark.parametrize(
    "text",
    [
        "agribound 2.0.0 is a major release",
        '__version__ = "2.0.0"',
        "[![Release](https://img.shields.io/badge/release-v2.0.0-green.svg)]",
        '!!! note "Changed in 2.0.0"',
        "Changed in 2.0: the default is one_to_one",
        "Added in v2:",
        "Backwards-compatible private name used before v2.",
        "composites after the v2 harmonisation",
        "converting nodata to the v2 convention",
        "the upgrade from agribound 2.x",
        "agribound v2 changes the defaults",
        "the agribound 2 line",
        "agribound version 2 is not released",
        "conda activate agribound-v2",
        "- Migrating to v2: migration-v2.md",
        # Wordings the 2.0.0 -> 1.0.0 renumbering removed.
        "Estimated runtime (not measured for v2): ~15-30 minutes with a GPU (FTW,",
        "Estimated runtime (not measured for v2): ~20-40 minutes (downloads plus SAM 2;",
        "example scripts have since been updated for v2, and the years they use",
        "Composites written by 0.1.x should not be reused with v2.",
        'v2. v2 cache names differ (new keys, `CACHE_SCHEMA_VERSION = "2"`), and a',
        "0.1.x output has no provenance record, so v2 refuses to reuse it",
        "ignored the named arguments. In v2, keyword arguments and named arguments that",
        "smoothing, so v2 outputs can have fewer polygons. See",
        "`embedding_clusters_{source}.tif`. v2 names every intermediate with a content",
        "# Engines (v2):",
        "# by-admin-conf layout (v2): determination:datetime, confidence, partitions",
        "# float products just above an integer up; v2 agrees with it everywhere else.",
        "float32 reflectance x 10000, NaN nodata (v2 layout).",
        "new in version 2 of agribound",
        # Requirements, "2.0"/"2.x" in a release context and "v2.0".
        "pip install agribound==2.0.0",
        '"agribound>=2.0.0"',
        "pip install 'agribound[gfm]>=2.0'",
        "In 2.0, keyword arguments and named arguments that",
        "In 2.0 the floor is configuration",
        "since 2.0, the default is one_to_one",
        "Changed in version 2.0",
        '!!! note "New in 2.x"',
        "the 2.0 release",
        "see the v2.0 release notes",
        "upgrading to the 2.x line",
        "after upgrading to 2.0, clear the cache",
        "agribound 2.0 helpers on 2.0 composites",
        "composites after the 2.0 harmonisation",
        "(not measured for 2.0)",
        "any 2.x release",
        "v2.0 names every intermediate with a content",
    ],
)
def test_v2_release_pattern_catches(text):
    assert _v2_release_wording(text)


@pytest.mark.parametrize(
    "text",
    [
        "agribound 1.0.0 is a major release",
        "Changed in 1.0.0",
        "Added in 1.0:",
        "Estimated runtime (not measured for 1.0): ~15-30 minutes with a GPU (FTW,",
        "float32 reflectance x 10000, NaN nodata (1.0 layout).",
        "ftw-tools 2.0.0b5 raises",
        '"ftw-tools>=2.0.0b5,<3"',
        "Prithvi needs terratorch, while ftw-tools 2.x requires lightning<2.6.",
        "Delineate Anything v2: A Global Foundation Model",
        "#   delineate-anything  Delineate-Anything v2 (--engine-param da_model=large_v2; the",
        "``large_v2`` (default): ``DelineateAnythingv2.pt``",
        "TESSERA v2 is a sparse beta (mostly Europe)",
        "FBIS-22M for v1, FBIS-73M for v2); outputs field instances directly",
        '(``"vultr"`` for v1, ``"cambridge"`` for v1.1, ``"2B-L~beta1"`` for v2',
        "#   --tessera-version V      v1 | v1.1 | v2 (default: region's, else v1)",
        '# v2 (beta variants "2B-L~beta1"/"2B-L~beta2"): sparse, mostly Europe.',
        "Zealand) with no near-global year; v2 is a sparse beta (mostly Europe).",
        "v2 embeddings instead of PCA. Every random choice is seeded from",
        '    "v2": (2017, 2025),',
        'if version == "v2":',
        '``"v1"``, ``"v1.1"`` or ``"v2"``.',
        "Include models marked legacy (FTW v1/v2 checkpoints; default False).",
        "FTW_v2_* checkpoints",
        "Prithvi-EO-2.0-300M-TL",
        "HLS v2.0 (L30 + S30)",
        'CACHE_SCHEMA_VERSION = "2"',
        "the cache schema v2 keys",
        "uses: lycheeverse/lychee-action@v2",
        "https://api.anthropic.com/v1/messages",
        "Licensed under the Apache License, Version 2.0",
        "the NSIDC EASE-Grid 2.0 equal-area CRS",
        "The keyword arrived in shapely 2.1 (2.0.x has ``make_valid(geometry,",
        '"geedim>=2.0,<3",',
        "simplify_tolerance: 2.0         # metres",
        # Numbers, other packages' lines and upper bounds.
        "| `simplify_tolerance` (`2.0`) | Douglas-Peucker tolerance in **metres** (0 disables). |",
        "Douglas-Peucker tolerance in metres (default 2.0). ``<= 0`` returns",
        "the tolerance defaults to 2.0",
        "buffers are grown in 2.0 m steps",
        "retried after 2.0 s",
        "in 2.0 to 3.0 m",
        "ftw-tools 2.x is published on PyPI only as pre-releases (2.0.0b5 as of",
        "is capped to match ftw-tools 2.x (torchvision<0.26).",
        "# The exact text of geedim 2.x (geedim/stac.py) for an image without an ID.",
        "Backends: ``sam2`` (default; SAM 2.0 via segment-geospatial),",
        "SamGeo2 accepts only SAM 2.0 checkpoints",
        "Version 2.0 of the Apache License",
        "Harmonized Landsat Sentinel-2 v2.0 (HLSL30 + HLSS30)",
        "- **HLS v2.0**: Earth Engine stores HLS as 0-1 reflectance; values are",
        "The keyword arrived in shapely 2.1 (2.0.x has a different signature)",
        "pip install 'agribound>=1.0,<2'",
        "agribound 1.0 composites after the 1.0 harmonisation",
    ],
)
def test_v2_release_pattern_ignores_other_versions(text):
    assert not _v2_release_wording(text)


def test_v2_release_pattern_uses_neighbouring_lines():
    # "v1" on the previous or next line marks a list of dataset/model versions.
    assert not _v2_release_wording(
        '"for v2); outputs field instances directly"',
        prev='"trained on 0.25-10 m imagery (FBIS-22M for v1, FBIS-73M "',
    )
    assert not _v2_release_wording(
        "``conf_threshold`` (per model; the ``ftw`` backend uses 0.15 for v2 and",
        nxt="ftw-tools' 0.05 for v1); ``batch_size`` (tiles per forward pass, 4).",
    )
    # A product name ending the previous line owns the "v2" that starts the next.
    assert not _v2_release_wording(
        "v2 matryoshka prefixes", prev="`matryoshka_depth` clusters a prefix of TESSERA"
    )
    assert _v2_release_wording("v2 names every intermediate", prev="`dinov3_segmentation.tif`.")
    # A "2.0"/"2.x" that starts a line takes its release context from the previous line.
    assert _v2_release_wording("2.x line, clear the cache", prev="After upgrading to the")
    assert _v2_release_wording("2.0, keyword arguments are applied", prev="the named arguments. In")
    assert not _v2_release_wording("2.x require lightning<2.6", prev="terratorch and ftw-tools")
    assert not _v2_release_wording("2.0 m steps", prev="buffers grow in")


def test_python_sources_are_checked_in_comments_and_strings_only(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text(
        'v2 = DA_MODELS["large_v2"]  # the Delineate-Anything v2 weights\n'
        'MESSAGE = "not measured for v2"\n'
        "# In v2, keyword arguments are applied.\n"
        'NOTE = f"{v2} names every intermediate"\n',
        encoding="utf-8",
    )
    found = _v2_release_offenders([source], root=tmp_path)
    assert [line.split(":")[1] for line in found] == ["2", "3"]


def test_notebook_cells_are_checked(tmp_path):
    notebook = tmp_path / "sample.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Title\n", "Updated for v2.\n"]},
                    {"cell_type": "code", "source": ['v2 = "large_v2"\n', "# TESSERA v2 tiles\n"]},
                    {"cell_type": "code", "source": ["!pip install agribound\n", "# In v2, ...\n"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    found = _v2_release_offenders([notebook], root=tmp_path)
    assert [line.split(": ")[0] for line in found] == [
        "sample.ipynb:cell 1, line 2",
        "sample.ipynb:cell 3, line 2",
    ]


def test_notebook_text_outputs_are_checked(tmp_path):
    notebook = tmp_path / "sample.ipynb"
    code = {"cell_type": "code", "source": ["import agribound\n", "agribound.__version__\n"]}
    code["outputs"] = [
        {"output_type": "stream", "name": "stdout", "text": ["DA large_v2\n", "agribound 2.0.0\n"]},
        {"output_type": "execute_result", "data": {"text/plain": ["'2.0.0'"]}},
        {"output_type": "display_data", "data": {"text/markdown": "In 2.0, the default is"}},
        {"output_type": "display_data", "data": {"image/png": "djIuMC4w"}},
    ]
    notebook.write_text(json.dumps({"cells": [code]}), encoding="utf-8")
    found = _v2_release_offenders([notebook], root=tmp_path)
    assert [line.split(": ")[0] for line in found] == [
        "sample.ipynb:cell 1, output 1, line 2",
        "sample.ipynb:cell 1, output 2, line 1",
        "sample.ipynb:cell 1, output 3, line 1",
    ]


def test_matching_help_names_0_1_x():
    result = CliRunner().invoke(main, ["evaluate", "--help"], terminal_width=200)
    assert result.exit_code == 0, result.output
    help_text = " ".join(result.output.split())
    assert "many_to_one (the 0.1.x rule;" in help_text
    assert "v1 rule" not in help_text
