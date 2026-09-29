#!/usr/bin/env python
"""
Render the agribound 1.0 workflow diagram.

Usage::

    python tools/make_workflow_diagram.py            # writes assets/agribound_workflow_1.0.*
    python tools/make_workflow_diagram.py --out-dir /tmp/diagram --no-verify

Outputs (same drawing in three formats):

- ``agribound_workflow_1.0.png``: 5700 x 3645 px, 300 dpi, white background;
- ``agribound_workflow_1.0.svg``: glyphs converted to paths (no font needed to view it);
- ``agribound_workflow_1.0.pdf``: TrueType fonts embedded. The repository's
  ``.gitignore`` ignores ``*.pdf``, so only the PNG and SVG are tracked; use
  ``git add -f`` to commit the PDF (for example for a paper).

The script needs only matplotlib (>= 3.6) and its own dependencies (NumPy,
Pillow). It is deterministic: no random numbers, a fixed SVG hash salt, and
no creation dates or software versions in the file metadata, so the same
matplotlib and fonts give byte-identical files.

Design
------
The layout keeps the look of the 0.1.x workflow figure (removed from
``assets/`` in 1.0.0; it is in the git history): a flat vector look with
light rounded stage cards and slate borders, numbered dark circles, engine
chips coloured by family (task-specific segmentation, light
blue; geospatial foundation model, light orange; embedding clustering, light
green; multi-engine ensemble, light purple) with a legend row at the bottom,
a dashed teal band across the top for the optional agent layer, bold dark
arrows between the stages, and the LULC dataset stack below its stage.

- The dashed teal band holds only the optional agent layer (the flow from
  request to report and the LLM connection). The deterministic entry point
  (``delineate()`` and ``agribound delineate --config``) and the per-run
  reproducibility features are not optional, so they sit in a separate solid
  card beside the band, aligned with the stage-6 column.
- The family colours belong to the engines: the engine chips, the field
  shapes (one per family) and the embedding fields in the Inputs card, which
  only the embedding engine reads. The stage numbers are all dark (grey for
  the optional stage 2) and the LULC dataset chips are neutral, so the legend
  has one reading.
- Stage titles share one baseline, and every stage body starts the same gap
  below its title (short bodies leave whitespace at the bottom of the card,
  as in the original).
- Type scale: seven sizes, each with one role (see :data:`TYPE_SCALE`);
  :class:`Style` rejects any other size.
- Arrow tips stop :data:`TIP_CLEAR` units from the stroke of their target and
  tails start :data:`TAIL_CLEAR` units from the stroke of their source.
- Contrast: every text colour reaches WCAG 4.5:1 on the fill it is set on,
  and every line that carries meaning (borders, arrows, the dotted agent
  connectors, the dashed "optional" outlines) reaches the WCAG 1.4.11 target
  of 3:1 on white. Only decorative lines are lighter: the dividers inside
  cards and the borders of the two footer notes, whose text stands on its own.

All geometry is in layout units of 1/100 inch (the figure is 19 x 12.15 in),
with *y* growing downwards. Font sizes are in points (1 pt = 100/72 units).
Text is set in Helvetica Neue when it is installed (falling back to
Helvetica, Arial, then DejaVu Sans) and code in DejaVu Sans Mono (bundled with
matplotlib). Boxes that hold measured text (pills, notes) are sized from the
rendered text extents.

Layout checks
-------------
Before anything is written, :meth:`Diagram.check_layout` verifies that

- every text lies inside its box, with a margin (no clipping);
- no two texts overlap, and no text overlaps an arrow or a drawn shape;
- boxes are either nested or apart (no partial overlaps);
- everything lies inside the canvas.

A failed check raises :class:`LayoutError` and nothing is written.

Facts
-----
Every label states something about the code. :func:`verify_facts` checks
the statements that can be read from the package (``--no-verify`` skips it;
without an importable ``agribound`` it is skipped with a message). The
sources of truth are:

- sources, resolutions, years, value scales, engines, fine-tunable engines and
  SAM backends: ``agribound/registry.py``;
- composites (two methods: median and greenest-pixel, with ``max_ndvi`` an
  alias of ``greenest``; NAIP mosaicked rather than composited; date windows;
  UTM export grid; cloud masks and reflectance x 10 000 for Sentinel-2,
  Landsat and HLS only): ``agribound/composites/*.py`` and
  ``agribound/config.py``;
- entry points ``delineate(study_area, source, year, engine, ...)`` and
  ``agribound delineate --config``: ``agribound/pipeline.py`` and
  ``agribound/cli.py``; seed (default 42), content-addressed cache and
  ``provenance.json`` (on by default): ``agribound/config.py`` and
  ``agribound/_cache.py``;
- stage order (composite, fine-tuning, delineation, SAM refinement,
  study-area selection, merge, min-area, smooth, simplify, LULC filter,
  metadata, evaluation, export, provenance): ``agribound/pipeline.py``;
- engine architectures: ``agribound/engines/*.py`` (Delineate-Anything default
  ``large_v2`` = YOLO11x-seg; FTW default ``FTW_PRUE_EFNET_B5``, a PRUE U-Net
  with two seasonal windows; GeoAI Mask R-CNN; DINOv3 SAT-493M ViT-L/16 + DPT
  decoder; Prithvi-EO-2.0 with a UPerNet head or K-means on patch embeddings;
  embedding K-means; ensemble intersection / union / vote);
- fine-tuning (full for all four trainers, LoRA for DINOv3 and Prithvi,
  ``fine_tune_split="block"`` with 5 km blocks by default); a checkpoint, from
  fine-tuning or ``engine_params["checkpoint_path"]``, is needed by GeoAI and
  DINOv3 (no published field-boundary weights) and by Prithvi's UPerNet
  segmentation mode (its embed mode is label-free):
  ``agribound/engines/finetune/*.py`` and ``agribound/registry.py``;
- LULC datasets, years and ``auto`` routing (Annual NLCD where the area is
  covered, else Dynamic World from 2016, else C3S; CDL, CONUS only, on
  request); the two modes, server-side ``reduceRegions`` on Earth Engine or
  local zonal means on a crop raster downloaded from Earth Engine during the
  composite stage (there is no option for a user-supplied LULC raster):
  ``agribound/postprocess/lulc_filter.py``;
- GeoParquet output with fiboa-style columns (``id``,
  ``determination:method``, geometry in EPSG:4326; no fiboa collection
  metadata is written): ``agribound/io/vector.py``;
- output columns ``metrics:area``, ``metrics:perimeter``,
  ``agribound:compactness`` and ``lulc:crop_fraction``:
  ``agribound/pipeline.py`` and ``agribound/postprocess/lulc_filter.py``;
- agent layer (typed tools, hash-bound single-use approval, at most one
  execution per session, stop after the run, MCP server, Anthropic Messages
  API with ``base_url``): ``agribound/agent/*.py``;
- ``agribound tiles make / run / merge``: ``agribound/hpc/cli.py``;
  ``evaluate()``: ``agribound/evaluate.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import logging
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Fallback fonts lack weight 500; matplotlib then logs one line per lookup.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, Wedge  # noqa: E402
from PIL import Image  # noqa: E402

# ---------------------------------------------------------------------------
# Canvas, fonts, type scale and colours
# ---------------------------------------------------------------------------

#: Canvas size in layout units (1 unit = 0.01 inch).
W, H = 1900, 1215
#: Output resolution of the PNG (dots per inch).
DPI = 300
#: Layout units per typographic point.
PT = 100 / 72
#: Default output location and file stem.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "assets"
DEFAULT_STEM = "agribound_workflow_1.0"

SANS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
MONO = ["DejaVu Sans Mono", "Menlo"]

#: Type scale in points. Each size has one role.
DISPLAY = 17.0  # section titles: agent layer, entry point
TITLE = 14.5  # stage and side-card titles
NUMERAL = 13.0  # stage numbers
LABEL = 10.5  # pills, the gate, highlighted names, subtitles, legend
BODY = 9.8  # stage text, chip and dataset names, note titles
CODE = 9.4  # monospace next to BODY text (DejaVu Sans Mono runs larger)
CAPTION = 8.6  # captions, details, small headings (sans or monospace)
TYPE_SCALE = (DISPLAY, TITLE, NUMERAL, LABEL, BODY, CODE, CAPTION)

#: Colours. Text colours reach WCAG 4.5:1 on the fill they are set on (lowest
#: pairs: ``muted`` on the ensemble chip fill 4.6:1, ``teal`` on ``teal_fill``
#: 4.6:1, white numerals on ``num_grey`` 4.5:1). Lines that carry meaning reach
#: 3:1 on white (lowest: ``lulc_edge`` 3.0:1, ``dash_edge`` 3.1:1,
#: ``teal_line`` 3.2:1, the geospatial-foundation-model edge 3.5:1).
C = {
    "text": "#1b2733",
    "muted": "#5b6b7a",
    "green_text": "#357a5b",
    "teal": "#2c776d",
    "teal_fill": "#e6f1f0",
    "teal_line": "#6b9a93",
    "teal_halo": "#c4dcd8",
    "card_fill": "#f4f7f9",
    "card_edge": "#8090a0",
    "lulc_edge": "#8a96a3",
    "del_fill": "#eef2f6",
    "del_edge": "#5b6b7a",
    "dash_edge": "#8593a0",
    "arrow": "#33414d",
    "num_dark": "#42505d",
    "num_grey": "#6b7885",
    "divider": "#d3dae1",
    "note_edge": "#c6ced6",
    "white": "#ffffff",
}

#: Engine families: (fill, edge).
FAMILY = {
    "task": ("#e8eef5", "#3d6c9c"),
    "gfm": ("#fbeee1", "#c07a33"),
    "embedding": ("#e6f1ea", "#3f8a67"),
    "ensemble": ("#efe9f4", "#6d5f9c"),
}

#: Border widths in points.
CARD_LW = 1.9
DEL_LW = 2.3
PILL_LW = 1.5
FINETUNE_LW = 1.7
#: Clearance in layout units between an arrow tip and the stroke of its target,
#: and between the stroke of its source and an arrow tail.
TIP_CLEAR = 7.0
TAIL_CLEAR = 7.0


def half_stroke(lw: float) -> float:
    """Half of a border of *lw* points, in layout units (how far it extends outwards)."""
    return lw * PT / 2


class LayoutError(RuntimeError):
    """A layout check failed (clipped, overlapping or out-of-canvas element)."""


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in layout units (*y* grows downwards)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def inset(self, d: float) -> Rect:
        return Rect(self.x0 + d, self.y0 + d, self.x1 - d, self.y1 - d)

    def contains(self, other: Rect, tol: float = 0.0) -> bool:
        return (
            other.x0 >= self.x0 - tol
            and other.y0 >= self.y0 - tol
            and other.x1 <= self.x1 + tol
            and other.y1 <= self.y1 + tol
        )

    def intersects(self, other: Rect) -> bool:
        return not (
            other.x0 >= self.x1 or other.x1 <= self.x0 or other.y0 >= self.y1 or other.y1 <= self.y0
        )


def box(x: float, y: float, w: float, h: float) -> Rect:
    """Rect from a top-left corner and a size."""
    return Rect(x, y, x + w, y + h)


@dataclass(frozen=True)
class Style:
    """Text style: size in points (one of :data:`TYPE_SCALE`), colour, weight and font."""

    size: float
    color: str = C["text"]
    weight: int = 400
    mono: bool = False

    def __post_init__(self) -> None:
        if self.size not in TYPE_SCALE:
            raise ValueError(f"font size {self.size} pt is not in the type scale {TYPE_SCALE}")
        if self.size == CODE and not self.mono:
            raise ValueError(f"{CODE} pt is reserved for monospace text next to body text")

    @property
    def family(self) -> list[str]:
        return MONO if self.mono else SANS


Run = tuple[str, Style]

#: A run whose text is ARROW is drawn as a small arrow (Helvetica Neue has no U+2192).
ARROW = "→"
#: Width of a drawn ARROW run, in em.
ARROW_EM = 1.0


@dataclass
class _Text:
    artist: object
    within: Rect | None
    name: str


@dataclass
class _Box:
    rect: Rect
    name: str


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


class Diagram:
    """Figure with drawing helpers that record every text and box for the layout checks."""

    def __init__(self) -> None:
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": SANS,
                "font.monospace": MONO,
                "svg.fonttype": "path",
                "svg.hashsalt": "agribound-workflow-1.0",
                "pdf.fonttype": 42,
                "path.simplify": False,
            }
        )
        # Measure text at the output resolution, so run widths match the PNG.
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=DPI, facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()
        self.texts: list[_Text] = []
        self.boxes: list[_Box] = []
        self.obstacles: list[tuple[Rect, str]] = []
        self._z = 1.0

    # -- z-order -----------------------------------------------------------

    def _next_z(self) -> float:
        self._z += 0.01
        return self._z

    # -- text --------------------------------------------------------------

    def _extent(self, artist) -> Rect:
        bb = artist.get_window_extent(renderer=self.renderer)
        inv = self.ax.transData.inverted()
        (xa, ya), (xb, yb) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return Rect(min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))

    def measure(self, runs: Sequence[Run] | str, style: Style | None = None) -> float:
        """Width in layout units of a line of text runs (without drawing it)."""
        runs = [(runs, style)] if isinstance(runs, str) else list(runs)
        total = 0.0
        for text, st in runs:
            if text == ARROW:
                total += ARROW_EM * st.size * PT
                continue
            artist = self.ax.text(0, 0, text, fontsize=st.size, family=st.family, weight=st.weight)
            total += self._extent(artist).w
            artist.remove()
        return total

    def line(
        self,
        x: float,
        yc: float,
        runs: Sequence[Run] | str,
        style: Style | None = None,
        *,
        ha: str = "center",
        within: Rect | None = None,
        name: str = "",
        gap: float = 0.0,
    ) -> Rect:
        """Draw one line of text runs on a shared baseline.

        The baseline is placed so that the cap height of the largest run is
        centred on *yc*; smaller runs share that baseline. *runs* is a string
        (with *style*) or a list of ``(text, Style)`` runs set one after
        another with *gap* units between them. *ha* aligns the whole line at
        *x* (``"left"``, ``"center"`` or ``"right"``). Returns the line's
        extent.
        """
        runs = [(runs, style)] if isinstance(runs, str) else list(runs)
        big = max(st.size for _, st in runs)
        baseline = yc + 0.36 * big * PT
        artists = []
        widths = []
        for text, st in runs:
            if text == ARROW:
                artists.append(None)
                widths.append(ARROW_EM * st.size * PT)
                continue
            artist = self.ax.text(
                0,
                baseline,
                text,
                fontsize=st.size,
                family=st.family,
                weight=st.weight,
                color=st.color,
                ha="left",
                va="baseline",
                zorder=50,
            )
            artists.append(artist)
            widths.append(self._extent(artist).w)
        total = sum(widths) + gap * (len(runs) - 1)
        left = {"left": x, "center": x - total / 2, "right": x - total}[ha]
        cursor = left
        extents = []
        for artist, width, (_, st) in zip(artists, widths, runs, strict=True):
            if artist is None:
                em = st.size * PT
                ay = baseline - 0.3 * em
                x0, x1 = cursor + 0.1 * em, cursor + width - 0.1 * em
                self.arrow(
                    x0,
                    ay,
                    x1,
                    ay,
                    color=st.color,
                    width=0.08 * em,
                    head_len=0.34 * em,
                    head_w=0.34 * em,
                    record=False,
                )
                extents.append(Rect(x0, ay - 0.17 * em, x1, ay + 0.17 * em))
            else:
                artist.set_x(cursor)
                extents.append(self._extent(artist))
                self.texts.append(_Text(artist, within, name or artist.get_text()))
            cursor += width + gap
        return Rect(
            min(e.x0 for e in extents),
            min(e.y0 for e in extents),
            max(e.x1 for e in extents),
            max(e.y1 for e in extents),
        )

    def lines(
        self,
        x: float,
        y_first: float,
        rows: Iterable[tuple[Sequence[Run] | str, Style | None]],
        step: float,
        *,
        ha: str = "center",
        within: Rect | None = None,
    ) -> float:
        """Draw rows of text *step* units apart; returns the centre *y* of the last row."""
        yc = y_first
        for i, (runs, style) in enumerate(rows):
            yc = y_first + i * step
            if runs:
                self.line(x, yc, runs, style, ha=ha, within=within)
        return yc

    # -- shapes ------------------------------------------------------------

    def rbox(
        self,
        r: Rect,
        *,
        fill: str,
        edge: str,
        lw: float = 1.6,
        radius: float = 12,
        dashed: tuple[float, float] | None = None,
        name: str = "",
        record: bool = True,
        alpha: float = 1.0,
    ) -> Rect:
        """Rounded rectangle; *lw* in points, *dashed* = (dash, gap) in points."""
        patch = FancyBboxPatch(
            (r.x0, r.y0),
            r.w,
            r.h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            linestyle=(0, dashed) if dashed else "solid",
            joinstyle="round",
            capstyle="butt",
            alpha=alpha,
            zorder=self._next_z(),
        )
        self.ax.add_patch(patch)
        if record:
            self.boxes.append(_Box(r, name))
        return r

    def number(self, cx: float, cy: float, n: int, color: str, r: float = 17) -> None:
        """Numbered stage circle with a white ring."""
        self.ax.add_patch(
            Circle((cx, cy), r, facecolor=color, edgecolor="white", linewidth=2.2, zorder=40)
        )
        self.ax.text(
            cx,
            cy + 0.36 * NUMERAL * PT,
            str(n),
            fontsize=NUMERAL,
            color="white",
            ha="center",
            va="baseline",
            zorder=41,
            family=SANS,
        )

    def arrow(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        color: str = C["arrow"],
        width: float = 4.6,
        head_len: float = 15,
        head_w: float = 15,
        name: str = "arrow",
        record: bool = True,
    ) -> None:
        """Straight arrow from (x0, y0) to the tip (x1, y1), drawn as filled polygons."""
        length = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        px, py = -uy, ux
        bx, by = x1 - ux * head_len, y1 - uy * head_len
        hw, sw = head_w / 2, width / 2
        shaft = [
            (x0 + px * sw, y0 + py * sw),
            (bx + ux * 0.5 + px * sw, by + uy * 0.5 + py * sw),
            (bx + ux * 0.5 - px * sw, by + uy * 0.5 - py * sw),
            (x0 - px * sw, y0 - py * sw),
        ]
        head = [(x1, y1), (bx + px * hw, by + py * hw), (bx - px * hw, by - py * hw)]
        z = self._next_z() + 20
        for pts in (shaft, head):
            self.ax.add_patch(
                Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=z)
            )
        if not record:
            return
        pad = max(hw, sw)
        self.obstacles.append(
            (Rect(min(x0, x1) - pad, min(y0, y1) - pad, max(x0, x1) + pad, max(y0, y1) + pad), name)
        )

    def dotted(self, x: float, y0: float, y1: float) -> None:
        """Vertical dotted connector from the agent band to a stage."""
        self.ax.plot(
            [x, x],
            [y0, y1],
            color=C["teal_line"],
            linewidth=1.8,
            linestyle=(0, (1.0, 2.4)),
            dash_capstyle="round",
            solid_capstyle="round",
            zorder=0.5,
        )

    def divider(self, x0: float, x1: float, y: float) -> None:
        self.ax.plot([x0, x1], [y, y], color=C["divider"], linewidth=1.0, zorder=30)

    def shape_obstacle(self, r: Rect, name: str) -> None:
        self.obstacles.append((r, name))

    # -- checks ------------------------------------------------------------

    def check_layout(self, text_margin: float = 3.0, box_gap: float = 4.0) -> str:
        """Run the layout checks; raise :class:`LayoutError` listing every problem."""
        problems: list[str] = []
        canvas = Rect(0, 0, W, H).inset(8)
        extents = [(self._extent(t.artist), t) for t in self.texts]

        for ext, t in extents:
            if not canvas.contains(ext):
                problems.append(f"text outside canvas: {t.name!r} {ext}")
            if t.within is not None and not t.within.inset(text_margin).contains(ext):
                problems.append(f"text not inside its box: {t.name!r} {ext} in {t.within}")

        for i in range(len(extents)):
            a, ta = extents[i]
            a_in = a.inset(0.6)
            for j in range(i + 1, len(extents)):
                b, tb = extents[j]
                if a_in.intersects(b.inset(0.6)):
                    problems.append(f"texts overlap: {ta.name!r} and {tb.name!r}")
            for rect, oname in self.obstacles:
                if a_in.intersects(rect):
                    problems.append(f"text {ta.name!r} overlaps {oname}")

        for i in range(len(self.boxes)):
            a = self.boxes[i]
            if not canvas.contains(a.rect):
                problems.append(f"box outside canvas: {a.name!r}")
            for j in range(i + 1, len(self.boxes)):
                b = self.boxes[j]
                nested = a.rect.contains(b.rect) or b.rect.contains(a.rect)
                if nested:
                    continue
                grown = Rect(
                    a.rect.x0 - box_gap,
                    a.rect.y0 - box_gap,
                    a.rect.x1 + box_gap,
                    a.rect.y1 + box_gap,
                )
                if grown.intersects(b.rect):
                    problems.append(f"boxes overlap or touch: {a.name!r} and {b.name!r}")

        if problems:
            raise LayoutError("layout check failed:\n  " + "\n  ".join(problems))
        return (
            f"layout checks passed: {len(self.texts)} texts, {len(self.boxes)} boxes, "
            f"{len(self.obstacles)} arrows/shapes"
        )

    # -- output ------------------------------------------------------------

    def save(self, out_dir: Path, stem: str) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = [out_dir / f"{stem}.{ext}" for ext in ("png", "svg", "pdf")]
        common = {"facecolor": "white", "edgecolor": "none"}
        # The PNG is written from the Agg buffer as opaque RGB (no alpha channel).
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        if rgba.shape[:2] != (round(H / 100 * DPI), round(W / 100 * DPI)):
            raise LayoutError(f"unexpected PNG size {rgba.shape[1]} x {rgba.shape[0]}")
        if (rgba[..., 3] != 255).any():
            raise LayoutError("PNG buffer has transparent pixels")
        Image.fromarray(rgba[..., :3], mode="RGB").save(paths[0], dpi=(DPI, DPI))
        self.fig.savefig(paths[1], metadata={"Date": None, "Creator": None}, **common)
        self.fig.savefig(
            paths[2], metadata={"CreationDate": None, "Creator": None, "Producer": None}, **common
        )
        return paths


# ---------------------------------------------------------------------------
# Text styles (every size comes from TYPE_SCALE)
# ---------------------------------------------------------------------------

TITLE_ST = Style(TITLE)
LABEL_ST = Style(LABEL)
BODY_ST = Style(BODY, C["muted"])
BODY_DARK = Style(BODY)
BODY_HEAD = Style(BODY, weight=500)
CODE_ST = Style(CODE, mono=True)
CAPTION_ST = Style(CAPTION, C["muted"])
CAPTION_HEAD = Style(CAPTION, weight=500)
CAPTION_MONO = Style(CAPTION, mono=True)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

# Main row columns: (x0, width); 50-unit gaps hold the arrows.
_GAP = 50
_WIDTHS = {"inputs": 215, "c1": 205, "c3": 290, "c4": 205, "c5": 205, "c6": 205, "fb": 215}
COLS: dict[str, tuple[float, float]] = {}
_x = 30.0
for _key, _w in _WIDTHS.items():
    COLS[_key] = (_x, _w)
    _x += _w + _GAP
assert abs((_x - _GAP) - (W - 30)) < 1e-6, "columns must span the canvas with 30-unit margins"

# Top row: the optional agent band over stages 1-5 and the solid entry-point card
# over stage 6 and the field boundaries, split at the gap between columns c5 and c6.
TOP_Y0, TOP_Y1 = 28, 240
BAND = Rect(30, TOP_Y0, COLS["c5"][0] + COLS["c5"][1], TOP_Y1)
ENTRY = Rect(COLS["c6"][0], TOP_Y0, W - 30, TOP_Y1)
#: Left/right padding of the text inside the band and the entry card.
INSET = 30
#: Rows shared by the band and the entry card: titles, the flow, captions.
TITLE_CY, FLOW_CY, CAP_CY = 68, 146, 202

DEL_TOP, DEL_H = 432, 532
ROW_CY = DEL_TOP + DEL_H / 2
CARD_H = 318
SIDE_H = 370


def col_rect(key: str, height: float) -> Rect:
    x0, w = COLS[key]
    return box(x0, ROW_CY - height / 2, w, height)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def draw_agent_band(d: Diagram) -> None:
    """Dashed teal band: the optional, human-confirmed agent layer."""
    d.rbox(
        BAND, fill=C["teal_fill"], edge=C["teal"], lw=2.4, radius=24, dashed=(6, 3.5), name="band"
    )
    # Title and run-in subtitle on one baseline.
    d.line(
        BAND.x0 + INSET,
        TITLE_CY,
        [
            ("Optional agent layer", Style(DISPLAY, C["teal"])),
            (
                "human-confirmed · the LLM proposes, you approve, one plan runs",
                Style(LABEL, C["muted"]),
            ),
        ],
        ha="left",
        within=BAND,
        gap=22,
    )

    # Right: the LLM connection.
    head_st = Style(CAPTION, C["muted"], 500)
    main_st = Style(LABEL, C["teal"], 700)
    mcp_w = 2 * 27 + max(
        d.measure("MCP server or Claude API", main_st),
        d.measure("local Anthropic-compatible LLMs", CAPTION_ST),
    )
    mcp = Rect(BAND.x1 - INSET - mcp_w, 46, BAND.x1 - INSET, FLOW_CY + 30)
    d.rbox(
        mcp, fill=C["white"], edge=C["teal"], lw=1.6, radius=14, dashed=(3.5, 2.5), name="mcp box"
    )
    d.line(mcp.cx, TITLE_CY, "LLM connection", head_st, within=mcp)
    d.line(mcp.cx, 104, "MCP server or Claude API", main_st, within=mcp)
    d.line(mcp.cx, 132, "local Anthropic-compatible LLMs", CAPTION_ST, within=mcp)
    d.line(
        mcp.cx,
        152,
        [("via ", CAPTION_ST), ("base_url", Style(CAPTION, C["muted"], mono=True))],
        within=mcp,
        gap=2,
    )

    # Flow: request -> plan -> HUMAN GATE -> run once -> report · stop.
    pill_st = Style(LABEL, C["teal"], 500)
    steps = [
        ("request", "natural language"),
        ("plan (typed tools)", "investigate, then propose one config"),
        (None, "hash-bound, single-use approval"),
        ("run once", "at most one run per session"),
        ("report · stop", "no automatic re-run"),
    ]
    gate_title = Style(LABEL, C["white"], 700)
    gate_sub = Style(BODY, C["white"])
    gate_half, halo_pad = 30, 4
    icon_w, pad = 26, 18
    widths = []
    for label, _cap in steps:
        if label is None:
            inner = (
                icon_w
                + 10
                + max(
                    d.measure("HUMAN GATE", gate_title),
                    d.measure("you confirm the exact plan", gate_sub),
                )
            )
        else:
            inner = d.measure(label, pill_st)
        widths.append(max(inner + 2 * pad, 112))

    x = BAND.x0 + INSET
    flow_right = mcp.x0 - 50
    arrow_gap = (flow_right - x - sum(widths)) / (len(steps) - 1)
    assert arrow_gap >= 44, "agent flow too wide for the band"
    outer = []  # the edge an arrow meets: the halo of the gate, the stroke of a pill
    for (label, _cap), w in zip(steps, widths, strict=True):
        if label is None:
            r = box(x, FLOW_CY - gate_half, w, 2 * gate_half)
            outer.append(r.inset(-halo_pad))
        else:
            r = box(x, FLOW_CY - 21, w, 42)
            outer.append(r.inset(-half_stroke(PILL_LW)))
        x = r.x1 + arrow_gap
    for i, ((label, cap), o) in enumerate(zip(steps, outer, strict=True)):
        if label is None:
            r = o.inset(halo_pad)
            d.rbox(o, fill=C["teal_halo"], edge="none", lw=0, radius=23, record=False)
            d.rbox(r, fill=C["teal"], edge=C["teal"], lw=1.6, radius=19, name="gate")
            # Person icon.
            ix, iy = r.x0 + pad + icon_w / 2, FLOW_CY
            d.ax.add_patch(
                Circle((ix, iy - 7), 5.2, facecolor="white", edgecolor="none", zorder=45)
            )
            d.ax.add_patch(
                Wedge((ix, iy + 11), 10.5, 180, 360, facecolor="white", edgecolor="none", zorder=45)
            )
            d.shape_obstacle(Rect(ix - 11, iy - 13, ix + 11, iy + 11), "gate icon")
            tx = r.x0 + pad + icon_w + 10
            d.line(tx, FLOW_CY - 12, "HUMAN GATE", gate_title, ha="left", within=r)
            d.line(tx, FLOW_CY + 10, "you confirm the exact plan", gate_sub, ha="left", within=r)
        else:
            r = o.inset(half_stroke(PILL_LW))
            d.rbox(r, fill=C["white"], edge=C["teal"], lw=PILL_LW, radius=21, name=f"pill {label}")
            d.line(r.cx, FLOW_CY, label, pill_st, within=r)
        d.line(r.cx, CAP_CY, cap, CAPTION_ST, within=BAND)
        if i < len(steps) - 1:
            d.arrow(
                o.x1 + TAIL_CLEAR,
                FLOW_CY,
                outer[i + 1].x0 - TIP_CLEAR,
                FLOW_CY,
                color=C["teal"],
                width=2.4,
                head_len=9,
                head_w=10,
                name="flow arrow",
            )
    assert outer[-1].x1 <= mcp.x0 - 40, "agent flow runs into the LLM connection box"


def draw_entry_point(d: Diagram) -> None:
    """Solid card beside the band: the deterministic entry point used by every run."""
    r = d.rbox(
        ENTRY, fill=C["card_fill"], edge=C["card_edge"], lw=CARD_LW, radius=24, name="entry card"
    )
    d.line(r.x0 + INSET, TITLE_CY, "Deterministic entry point", Style(DISPLAY), ha="left", within=r)
    code = Rect(r.x0 + INSET, 104, r.x1 - INSET, FLOW_CY + 30)
    d.rbox(code, fill=C["white"], edge=C["card_edge"], lw=1.4, radius=10, name="entry code")
    d.line(
        code.x0 + 16,
        127,
        "delineate(study_area, source, year, engine)",
        CODE_ST,
        ha="left",
        within=code,
    )
    d.line(
        code.x0 + 16, 153, "agribound delineate --config run.yaml", CODE_ST, ha="left", within=code
    )
    d.line(
        r.x0 + INSET,
        CAP_CY,
        [
            ("every run: seed · content-addressed cache · ", CAPTION_ST),
            ("provenance.json", Style(CAPTION, C["muted"], mono=True)),
            (" (default on)", CAPTION_ST),
        ],
        ha="left",
        within=r,
    )


def card(
    d: Diagram,
    r: Rect,
    name: str,
    *,
    fill: str = C["card_fill"],
    edge: str = C["card_edge"],
    lw: float = CARD_LW,
) -> Rect:
    return d.rbox(r, fill=fill, edge=edge, lw=lw, radius=16, name=name)


def draw_inputs(d: Diagram) -> Rect:
    r = card(d, col_rect("inputs", SIDE_H), "inputs")
    d.line(r.cx, r.y0 + 38, "Inputs", TITLE_ST, within=r)
    d.lines(
        r.cx,
        r.y0 + 70,
        [("10 sources · 0.3–30 m", CAPTION_ST), ("1984–present", CAPTION_ST)],
        19,
        within=r,
    )
    y = r.y0 + 126
    rows = [
        ("Sentinel-2 · Landsat", BODY_DARK),
        ("HLS · NAIP", BODY_DARK),
        ("USGS NAIP Plus", BODY_DARK),
        ("SPOT 6/7 MS · pan", BODY_DARK),
    ]
    last = d.lines(r.cx, y, rows, 25, within=r)
    d.line(r.cx, last + 19, "(restricted access)", CAPTION_ST, within=r)
    d.line(r.cx, last + 46, "local GeoTIFF", BODY_DARK, within=r)
    div_y = last + 70
    d.divider(r.x0 + 40, r.x1 - 40, div_y)
    d.line(r.cx, div_y + 22, "embedding fields", CAPTION_ST, within=r)
    # Green: only the embedding engine (embedding-clustering family) reads these.
    d.line(r.cx, div_y + 46, "AlphaEarth · TESSERA", Style(BODY, C["green_text"], 500), within=r)
    return r


#: Body row that draws a short divider line.
DIVIDER = ("<divider>", None)
#: Body row that separates two groups of lines.
SPACER = ("", None)
SPACER_H = 14.0
#: Distance from the centre of a card's last title line to the top of its body.
BODY_GAP = 29.0


def _row_height(runs: Sequence[Run] | str, style: Style | None) -> float:
    """Height in layout units of one stage-card body row."""
    if (runs, style) == DIVIDER:
        return 18
    if not runs:
        return SPACER_H
    size = style.size if style is not None else max(st.size for _, st in runs)
    return 21 if size >= CODE else 18.5


def stage_card(d: Diagram, key: str, n: int, title: Sequence[str], body: Sequence[tuple]) -> Rect:
    """Main-row stage card: number, centred title (one or two lines) and body rows.

    Titles sit at the same height in every card, and every body starts
    :data:`BODY_GAP` below its last title line, so sibling cards read alike;
    a short body leaves whitespace at the bottom of its card.
    """
    r = card(d, col_rect(key, CARD_H), f"stage {n}")
    d.number(r.x0 + 25, r.y0 + 25, n, C["num_dark"])
    d.shape_obstacle(Rect(r.x0 + 6, r.y0 + 6, r.x0 + 44, r.y0 + 44), f"circle {n}")
    title_y = r.y0 + 64
    for i, t in enumerate(title):
        d.line(r.cx, title_y + i * 24, t, TITLE_ST, within=r)
    y = title_y + (len(title) - 1) * 24 + BODY_GAP
    heights = [_row_height(runs, st) for runs, st in body]
    over = y + sum(heights) - (r.y1 - 16)
    if over > 0:
        raise LayoutError(f"stage {n} body is {over:.0f} units too tall for its card")
    for (runs, st), h in zip(body, heights, strict=True):
        yc = y + h / 2
        if (runs, st) == DIVIDER:
            d.divider(r.cx - 50, r.cx + 50, yc)
        elif runs:
            d.line(r.cx, yc, runs, st, within=r)
        y += h
    return r


def draw_finetune(d: Diagram, del_rect: Rect) -> Rect:
    """Dashed box 2 above the delineation card."""
    r = box(del_rect.cx - 215, TOP_Y1 + 30, 430, 122)
    d.rbox(
        r,
        fill=C["white"],
        edge=C["dash_edge"],
        lw=FINETUNE_LW,
        radius=14,
        dashed=(4.5, 3),
        name="finetune",
    )
    d.number(r.x0 + 23, r.y0 + 23, 2, C["num_grey"], r=15)
    d.shape_obstacle(Rect(r.x0 + 6, r.y0 + 6, r.x0 + 40, r.y0 + 40), "circle 2")
    d.line(
        r.x0 + 48,
        r.y0 + 23,
        "Optional fine-tuning on reference boundaries",
        Style(LABEL, weight=500),
        ha="left",
        within=r,
    )
    val_st = CAPTION_ST
    # "checkpoint": these engines cannot run without a trained checkpoint, which
    # comes from fine-tuning or engine_params["checkpoint_path"] (no field-boundary
    # weights are published for GeoAI or DINOv3; Prithvi needs one only for its
    # UPerNet segmentation mode, the label-free embed mode runs without).
    rows = [
        ("full", "Delineate-Anything · GeoAI · DINOv3 · Prithvi"),
        ("LoRA", "DINOv3 · Prithvi"),
        ("checkpoint", "needed by GeoAI · DINOv3 · Prithvi UPerNet"),
        ("split", "spatially blocked train / validation (default, 5 km)"),
    ]
    kx = r.x0 + 48
    vx = kx + max(d.measure(k, CAPTION_HEAD) for k, _ in rows) + 12
    for i, (k, v) in enumerate(rows):
        yc = r.y0 + 51 + i * 18
        d.line(kx, yc, k, CAPTION_HEAD, ha="left", within=r)
        d.line(vx, yc, v, val_st, ha="left", within=r)
    return r


def draw_delineation(d: Diagram) -> Rect:
    x0, w = COLS["c3"]
    r = card(
        d,
        box(x0, DEL_TOP, w, DEL_H),
        "delineation",
        fill=C["del_fill"],
        edge=C["del_edge"],
        lw=DEL_LW,
    )
    d.number(r.x0 + 25, r.y0 + 25, 3, C["num_dark"])
    d.shape_obstacle(Rect(r.x0 + 6, r.y0 + 6, r.x0 + 44, r.y0 + 44), "circle 3")
    d.line(r.x0 + 54, r.y0 + 26, "Delineation", TITLE_ST, ha="left", within=r)
    d.line(r.cx, r.y0 + 66, "7 engines", BODY_ST, within=r)
    engines = [
        ("Delineate-Anything v2", "YOLO11 instance segmentation", "task"),
        ("Fields of The World", "PRUE U-Net · two seasonal windows", "task"),
        ("GeoAI", "Mask R-CNN instance segmentation", "task"),
        ("DINOv3", "SAT-493M ViT-L/16 + DPT decoder", "gfm"),
        ("Prithvi-EO-2.0", "UPerNet head, or K-means on embeddings", "gfm"),
        ("Embedding", "K-means on AlphaEarth / TESSERA", "embedding"),
        ("Ensemble", "intersection · union · vote", "ensemble"),
    ]
    chip_h, chip_gap = 52, 10
    y = r.y0 + 92
    for name, detail, fam in engines:
        fill, edge = FAMILY[fam]
        c = d.rbox(
            box(r.x0 + 15, y, r.w - 30, chip_h),
            fill=fill,
            edge=edge,
            lw=1.7,
            radius=10,
            name=f"chip {name}",
        )
        d.line(c.cx, c.y0 + 18, name, BODY_HEAD, within=c)
        d.line(c.cx, c.y0 + 36, detail, CAPTION_ST, within=c)
        y += chip_h + chip_gap
    assert y - chip_gap + 12 <= r.y1, "engine chips overflow the delineation card"
    return r


def draw_lulc_stack(d: Diagram, c5: Rect, right_limit: float) -> Rect:
    """LULC datasets below stage 5, with the arrow up into the card.

    The chips are neutral (white with slate edges): the family colours are
    reserved for the engines, so a land-cover dataset is not read as an engine.
    CDL, used only on request, has the dashed "optional" outline of the legend.
    """
    tail_y = c5.y1 + 54
    d.arrow(
        c5.cx,
        tail_y,
        c5.cx,
        c5.y1 + half_stroke(CARD_LW) + TIP_CLEAR,
        color=C["muted"],
        width=3.2,
        head_len=12,
        head_w=12,
        name="LULC arrow",
    )
    label = d.line(c5.cx, tail_y + 15, "auto-selected by coverage & year", CAPTION_ST)
    datasets = [
        ("USGS Annual NLCD", "CONUS · 1985–2025", False),
        ("Google Dynamic World", "2016 to last full year", False),
        ("Copernicus C3S", "global · 2000–2022", False),
        ("USDA CDL", "on request · CONUS · 2013–2023", True),
    ]
    w, h, gap = 310, 38, 8
    x0 = c5.cx - w / 2
    assert x0 + w < right_limit, "LULC stack runs into the right-hand note"
    y = label.y1 + 12
    first = None
    for name, detail, optional in datasets:
        chip = box(x0, y, w, h)
        d.rbox(
            chip,
            fill=C["white"],
            edge=C["dash_edge"] if optional else C["lulc_edge"],
            lw=1.5,
            radius=9,
            dashed=(3.5, 2.5) if optional else None,
            name=f"lulc {name}",
        )
        d.line(chip.x0 + 14, chip.cy, name, BODY_HEAD, ha="left", within=chip)
        d.line(chip.x1 - 14, chip.cy, detail, CAPTION_ST, ha="right", within=chip)
        first = first or chip
        y += h + gap
    return Rect(x0, first.y0, x0 + w, y - gap)


def draw_field_boundaries(d: Diagram) -> Rect:
    r = card(d, col_rect("fb", SIDE_H), "field boundaries")
    d.line(r.cx, r.y0 + 38, "Field boundaries", TITLE_ST, within=r)
    lw = 2.2
    # Centre-pivot circle, half pivot, square and strip fields, one per engine family.
    cx1, cx2 = r.x0 + 62, r.x0 + 152
    cy1, cy2 = r.y0 + 118, r.y0 + 200
    f, e = FAMILY["task"]
    d.ax.add_patch(Circle((cx1, cy1), 32, facecolor=f, edgecolor=e, linewidth=lw, zorder=35))
    f, e = FAMILY["gfm"]
    d.ax.add_patch(
        Wedge((cx2, cy1 + 16), 38, 195, 375, facecolor=f, edgecolor=e, linewidth=lw, zorder=35)
    )
    f, e = FAMILY["embedding"]
    d.ax.add_patch(
        Rectangle((cx1 - 29, cy2 - 29), 58, 58, facecolor=f, edgecolor=e, linewidth=lw, zorder=35)
    )
    f, e = FAMILY["ensemble"]
    d.ax.add_patch(
        Rectangle((cx2 - 42, cy2 - 22), 84, 44, facecolor=f, edgecolor=e, linewidth=lw, zorder=35)
    )
    d.shape_obstacle(Rect(cx1 - 34, cy1 - 34, cx2 + 43, cy2 + 31), "field shapes")
    d.line(r.cx, r.y0 + 262, "per-field attributes", CAPTION_ST, within=r)
    d.lines(
        r.cx,
        r.y0 + 286,
        [
            ("metrics:area", CAPTION_MONO),
            ("metrics:perimeter", CAPTION_MONO),
            ("agribound:compactness", CAPTION_MONO),
            ("lulc:crop_fraction", CAPTION_MONO),
        ],
        18.5,
        within=r,
    )
    return r


def note(
    d: Diagram, x: float, y: float, title: str, runs: Sequence[Run], name: str, *, ha: str = "left"
) -> Rect:
    """Small footer note sized to its text; *x* is its left (or, for ``ha="right"``, right) edge."""
    pad = 20
    w = max(d.measure(title, BODY_HEAD), d.measure(runs)) + 2 * pad
    x0 = x if ha == "left" else x - w
    r = box(x0, y, w, 64)
    d.rbox(r, fill=C["white"], edge=C["note_edge"], lw=1.4, radius=12, name=name)
    d.line(r.x0 + pad, r.y0 + 21, title, BODY_HEAD, ha="left", within=r)
    d.line(r.x0 + pad, r.y0 + 44, runs, ha="left", within=r)
    return r


def draw_legend(d: Diagram, y: float) -> None:
    d.line(62, y, "Engine family:", LABEL_ST, ha="left")
    items = [
        ("task", "Task-specific segmentation"),
        ("gfm", "Geospatial foundation model"),
        ("embedding", "Embedding clustering (label-free)"),
        ("ensemble", "Multi-engine ensemble"),
    ]
    xs = [262, 622, 990, 1398]
    for (fam, label), x in zip(items, xs, strict=True):
        fill, edge = FAMILY[fam]
        d.rbox(box(x, y - 14, 32, 28), fill=fill, edge=edge, lw=1.8, radius=6, name=f"legend {fam}")
        d.line(x + 46, y, label, LABEL_ST, ha="left")
    sep_x = 1680
    d.ax.plot([sep_x, sep_x], [y - 16, y + 16], color=C["divider"], linewidth=1.2, zorder=30)
    d.rbox(
        box(sep_x + 26, y - 14, 32, 28),
        fill=C["white"],
        edge=C["dash_edge"],
        lw=1.6,
        radius=6,
        dashed=(3, 2.2),
        name="legend optional",
    )
    d.line(sep_x + 72, y, "optional", LABEL_ST, ha="left")


def build() -> Diagram:
    d = Diagram()
    draw_agent_band(d)
    draw_entry_point(d)

    inputs = draw_inputs(d)
    c1 = stage_card(
        d,
        "c1",
        1,
        ["Composite"],
        [
            ("Earth Engine", BODY_ST),
            ("(or USGS · TESSERA · local)", BODY_ST),
            # Two methods: max_ndvi is an alias of greenest.
            ("median · greenest (max-NDVI)", BODY_ST),
            ("(NAIP: mosaicked)", CAPTION_ST),
            ("annual or date window", BODY_ST),
            ("UTM export grid", BODY_ST),
            DIVIDER,
            # Only these three sources are cloud-masked and scaled to reflectance.
            ("S2 · Landsat · HLS", CAPTION_HEAD),
            ("cloud-masked", BODY_ST),
            ("reflectance ×10 000", BODY_ST),
        ],
    )
    c3 = draw_delineation(d)
    ft = draw_finetune(d, c3)
    c4 = stage_card(
        d,
        "c4",
        4,
        ["Refine &", "post-process"],
        [
            ("optional SAM refinement", BODY_ST),
            ("(SAM 2 · 2.1 · 3)", BODY_ST),
            SPACER,
            ("study-area selection", BODY_ST),
            ("merge · min-area", BODY_ST),
            ("smooth · simplify", BODY_ST),
        ],
    )
    c5 = stage_card(
        d,
        "c5",
        5,
        ["LULC crop filter"],
        [
            ("removes non-crop polygons", BODY_ST),
            ("(crop fraction < 0.3)", BODY_ST),
            SPACER,
            # Raster mode downloads the crop raster from Earth Engine, then
            # computes the zonal means locally.
            ("server-side on Earth Engine", BODY_ST),
            ("or on a downloaded raster", BODY_ST),
        ],
    )
    c6 = stage_card(
        d,
        "c6",
        6,
        ["Export"],
        [
            # fiboa-style columns; no fiboa collection metadata is written.
            ("GeoParquet (fiboa-style)", BODY_ST),
            ("GPKG · GeoJSON", BODY_ST),
            SPACER,
            ([("+ ", BODY_ST), ("provenance.json", CODE_ST)], None),
            ("config hash · seed · versions", CAPTION_ST),
        ],
    )
    fb = draw_field_boundaries(d)

    # Main arrows between the stages, centred on the row.
    row = [
        (inputs, CARD_LW),
        (c1, CARD_LW),
        (c3, DEL_LW),
        (c4, CARD_LW),
        (c5, CARD_LW),
        (c6, CARD_LW),
        (fb, CARD_LW),
    ]
    for (a, a_lw), (b, b_lw) in zip(row, row[1:], strict=False):
        d.arrow(
            a.x1 + half_stroke(a_lw) + TAIL_CLEAR,
            ROW_CY,
            b.x0 - half_stroke(b_lw) - TIP_CLEAR,
            ROW_CY,
            name="stage arrow",
        )
    # Fine-tuning feeds the delineation engines.
    d.arrow(
        ft.cx,
        ft.y1 + half_stroke(FINETUNE_LW) + TAIL_CLEAR,
        ft.cx,
        c3.y0 - half_stroke(DEL_LW) - TIP_CLEAR,
        color=C["dash_edge"],
        width=2.4,
        head_len=9,
        head_w=10,
        name="finetune arrow",
    )

    # Dotted connectors from the agent band to the stages it configures.
    for target in (c1, ft, c4, c5):
        assert BAND.x0 < target.cx < BAND.x1, "dotted connector outside the agent band"
        d.dotted(target.cx, BAND.y1 + 2, target.y0 - 3)

    # LULC datasets below stage 5; footer notes either side, centred on the stack.
    stack = draw_lulc_stack(d, c5, COLS["c6"][0] + 40)
    note_y = stack.cy - 32
    grey = BODY_ST
    note(
        d,
        COLS["inputs"][0],
        note_y,
        "Scale out large study areas",
        [
            ("agribound tiles make ", CODE_ST),
            (ARROW, CODE_ST),
            (" run ", CODE_ST),
            (ARROW, CODE_ST),
            (" merge", CODE_ST),
            ("   (Slurm arrays)", grey),
        ],
        "tiles note",
    )
    right = note(
        d,
        COLS["fb"][0] + COLS["fb"][1],
        note_y,
        "Accuracy assessment",
        [("evaluate()", CODE_ST), ("   object- & area-weighted metrics", grey)],
        "evaluate note",
        ha="right",
    )
    assert right.x0 > stack.x1 + 30, "evaluate note runs into the LULC stack"

    draw_legend(d, 1166)
    return d


# ---------------------------------------------------------------------------
# Fact checks against the package
# ---------------------------------------------------------------------------


def verify_facts() -> list[str]:
    """Check the diagram's statements against the installed agribound package.

    Returns the list of mismatches (empty when every checked fact holds).
    Raises :class:`ImportError` when ``agribound.registry`` is not importable.
    """
    sys.path.insert(0, str(REPO_ROOT))
    reg = importlib.import_module("agribound.registry")
    cfg = importlib.import_module("agribound.config")
    bad: list[str] = []

    def expect(ok: bool, what: str) -> None:
        if not ok:
            bad.append(what)

    src = reg.SOURCE_REGISTRY
    expect(len(src) == 10, f"10 sources (found {len(src)})")
    expect(src["landsat"]["year_range"] == (1984, None), "Landsat 1984-present")
    expect(
        min(r[0] for r in (s["year_range"] for s in src.values()) if r) == 1984, "first year 1984"
    )
    expect(
        max(s["resolution_m"] or 0 for s in src.values()) == 30, "coarsest default resolution 30 m"
    )
    expect(src["spot"]["restricted"] and src["spot-pan"]["restricted"], "SPOT 6/7 restricted")
    for name in ("sentinel2", "landsat", "hls"):
        expect(src[name]["value_scale"] == "reflectance_x10000", f"{name} reflectance x10000")
    expect(
        set(reg.EMBEDDING_SOURCES) == {"google-embedding", "tessera-embedding"}, "embedding sources"
    )

    eng = reg.ENGINE_REGISTRY
    expect(
        list(eng)
        == ["delineate-anything", "ftw", "geoai", "dinov3", "prithvi", "embedding", "ensemble"],
        "7 engines in diagram order",
    )
    expect(
        {k for k, v in eng.items() if set(v["supported_sources"]) & set(reg.EMBEDDING_SOURCES)}
        == {"embedding"},
        "only the embedding engine reads the embedding fields (green in the Inputs card)",
    )
    expect(
        {k for k, v in eng.items() if v["fine_tunable"]}
        == {"delineate-anything", "geoai", "dinov3", "prithvi"},
        "fine-tunable engines",
    )
    expect(
        not eng["geoai"]["label_free"] and not eng["dinov3"]["label_free"],
        "GeoAI/DINOv3 need checkpoints",
    )
    expect(
        eng["prithvi"]["label_free"] and "needs a fine-tuned checkpoint" in eng["prithvi"]["notes"],
        "Prithvi label-free in embed mode, checkpoint needed for segmentation",
    )
    expect("Mask R-CNN" in eng["geoai"]["approach"], "GeoAI Mask R-CNN")
    expect("DPT" in eng["dinov3"]["approach"], "DINOv3 DPT head")
    expect("K-means" in eng["embedding"]["approach"], "embedding K-means")
    expect(
        tuple(reg.SAM_REFINE_BACKENDS) == ("sam2", "sam2.1", "sam3", "sam3-hf"), "SAM 2 / 2.1 / 3"
    )

    defaults = cfg.AgriboundConfig()
    expect(
        defaults.fine_tune_split == "block" and defaults.fine_tune_block_size_m == 5000,
        "5 km block split",
    )
    expect(defaults.lulc_crop_threshold == 0.3, "crop fraction threshold 0.3")
    expect(defaults.export_crs == "utm", "UTM export CRS")
    expect(defaults.provenance is True, "provenance.json on by default")
    expect(isinstance(defaults.seed, int), "every run has a seed")
    expect(
        tuple(cfg.VALID_COMPOSITE_METHODS) == ("median", "greenest", "max_ndvi"),
        "composite methods (median, greenest and its alias max_ndvi)",
    )
    expect(tuple(cfg.VALID_OUTPUT_FORMATS) == ("gpkg", "geojson", "parquet"), "output formats")
    expect(tuple(cfg.VALID_LULC_MODES) == ("server", "raster"), "LULC server or raster mode")
    expect(
        not any(
            f.name.startswith("lulc") and "path" in f.name
            for f in dataclasses.fields(cfg.AgriboundConfig)
        ),
        "no user-supplied LULC raster (raster mode downloads one)",
    )
    expect(
        set(cfg.VALID_LULC_DATASETS) == {"auto", "nlcd", "cdl", "dynamic_world", "c3s"},
        "LULC datasets",
    )

    optional = {
        "agribound.pipeline": lambda m: [
            (
                list(inspect.signature(m.delineate).parameters)[:4]
                == ["study_area", "source", "year", "engine"],
                "delineate(study_area, source, year, engine)",
            ),
        ],
        "agribound.cli": lambda m: [
            (
                any("--config" in p.opts for p in m.main.commands["delineate"].params),
                "agribound delineate --config",
            ),
        ],
        "agribound._cache": lambda m: [
            (callable(getattr(m, "cache_key", None)), "content-addressed cache keys"),
        ],
        "agribound.postprocess.lulc_filter": lambda m: [
            (
                (m.LULC_DATASETS["nlcd"].first_year, m.LULC_DATASETS["nlcd"].last_year)
                == (1985, 2025),
                "NLCD 1985-2025",
            ),
            (
                (m.LULC_DATASETS["cdl"].first_year, m.LULC_DATASETS["cdl"].last_year)
                == (2013, 2023),
                "CDL 2013-2023",
            ),
            (
                (
                    m.LULC_DATASETS["dynamic_world"].first_year,
                    m.LULC_DATASETS["dynamic_world"].last_year,
                )
                == (2016, None),
                "Dynamic World 2016-last full year",
            ),
            (
                (m.LULC_DATASETS["c3s"].first_year, m.LULC_DATASETS["c3s"].last_year)
                == (2000, 2022),
                "C3S 2000-2022",
            ),
            (
                callable(m.prefetch_lulc_raster) and callable(m.zonal_mean_from_raster),
                "raster mode: downloaded crop raster, local zonal means",
            ),
        ],
        "agribound.composites.gee": lambda m: [
            (
                all(
                    callable(getattr(m, f, None))
                    for f in (
                        "prepare_landsat_image",
                        "prepare_hls_image",
                        "mask_s2_scl",
                        "mask_s2_cloud_score",
                    )
                ),
                "cloud masks for Landsat, HLS and Sentinel-2",
            ),
            (m.NAIP_COLLECTION == "USDA/NAIP/DOQQ", "NAIP from Earth Engine (mosaicked)"),
            (
                '``"max_ndvi"`` is an alias of ``"greenest"``'
                in (m.apply_composite_method.__doc__ or ""),
                "max_ndvi is an alias of greenest",
            ),
        ],
        "agribound.engines.ensemble": lambda m: [
            (tuple(m.MERGE_STRATEGIES) == ("intersection", "union", "vote"), "ensemble strategies"),
        ],
        "agribound.engines.finetune": lambda m: [
            (
                set(m._TRAINERS) == {"delineate-anything", "geoai", "dinov3", "prithvi"},
                "fine-tuning trainers",
            ),
        ],
        "agribound.engines.delineate_anything": lambda m: [
            (m.DEFAULT_DA_MODEL == "large_v2", "Delineate-Anything v2 default"),
        ],
        "agribound.engines.dinov3": lambda m: [
            ("sat493m" in m.DINOV3_DEFAULT_WEIGHTS[1], "DINOv3 SAT-493M weights"),
            (m.DINOV3_DEFAULT_BACKBONE == "dinov3_vitl16", "DINOv3 ViT-L/16"),
        ],
        "agribound.agent.gate": lambda m: [
            (
                inspect.signature(m.ConfirmationGate).parameters["max_executions"].default == 1,
                "one execution per session",
            ),
        ],
    }
    for module, checks in optional.items():
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            print(f"  fact check skipped for {module}: {exc}")
            continue
        try:
            results = checks(mod)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            bad.append(f"{module}: cannot read the checked names ({type(exc).__name__}: {exc})")
            continue
        for ok, what in results:
            expect(ok, what)
    return bad


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="output directory (default: assets/)"
    )
    parser.add_argument(
        "--stem", default=DEFAULT_STEM, help=f"file name stem (default: {DEFAULT_STEM})"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="skip the fact checks against agribound"
    )
    args = parser.parse_args(argv)

    if not args.no_verify:
        try:
            bad = verify_facts()
        except ImportError as exc:
            print(f"fact checks skipped (agribound not importable: {exc})")
        else:
            if bad:
                print("fact checks FAILED; the diagram no longer matches the code:")
                for what in bad:
                    print(f"  - {what}")
                return 1
            print("fact checks passed")

    sans = font_manager.findfont(font_manager.FontProperties(family=SANS))
    mono = font_manager.findfont(font_manager.FontProperties(family=MONO))
    print(f"fonts: {Path(sans).name}, {Path(mono).name}")

    d = build()
    print(d.check_layout())
    for path in d.save(args.out_dir, args.stem):
        print(f"wrote {path}")
    plt.close(d.fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
