# assets

Figures for the README and the documentation. The README and the docs link them by URL
(`https://raw.githubusercontent.com/montimaj/agribound/main/assets/...`); they are not part
of the Python package.

| Path | What |
|---|---|
| `agribound_workflow_1.0.{png,svg}` | Workflow diagram, rendered by `tools/make_workflow_diagram.py` (the PDF it also writes is not tracked) |
| `gallery_1.0/` | Gallery images from the agribound 1.0.0 example runs, rendered by `tools/make_gallery.py`; `gallery_stats.json` holds the facts the captions quote |
| `gallery_1.0/preview/` | 1600 px WebP previews of those images, used for the embeds (each links to its 3000 px PNG) |
| `gallery_0.1x/` | The agribound 0.1.x screenshots (archived; see `docs/gallery-0.1x.md`) |
| `NM_example.png`, `Pampas_example.png` | Byte-identical copies of `gallery_0.1x/NM_example.png` and `gallery_0.1x/Pampas_example.png`, kept at their old paths because the READMEs of the published 0.1.x releases (shown on PyPI) link them there. Do not move or edit them; git stores each identical file once. |
