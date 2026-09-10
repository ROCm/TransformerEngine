#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Bundle the dashboard into a single self-contained ``dashboard.html``.

Inlines styles.css, Chart.js and app.js, and embeds each data shard as a
``<script type="text/csv" data-file="...">`` block that app.js reads via
``getText()`` instead of fetching (fetch is blocked on ``file://``). The result
opens offline by double-click -- no server, no network -- so it can be shared as
a single file (email/Teams/etc.).

Usage:
  build_bundle.py [--data-dir DIR] [--out FILE]

Defaults: --data-dir dashboard/data   --out dashboard/dist/dashboard.html
For the published demo snapshot, point at the gh-pages checkout:
  build_bundle.py --data-dir te-dash/data
"""

import argparse
import re
import sys
from pathlib import Path

DASH = Path(__file__).resolve().parent / "dashboard"


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def _inline_safe(text, tag):
    r"""Neutralize a literal ``</tag>`` inside inlined code so it can't close the
    surrounding ``<script>``/``<style>`` early. ``<\/tag`` is equivalent in JS
    (string/regex/comment) and harmless in CSS."""
    return re.sub(r"</(" + tag + r")", lambda m: "<\\/" + m.group(1), text, flags=re.IGNORECASE)


def _replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        sys.exit(f"expected exactly one '{label}' marker in index.html, found {n}")
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DASH / "data"),
                        help="dir with index.csv + perf-*.csv (default: dashboard/data)")
    parser.add_argument("--out", default=str(DASH / "dist" / "dashboard.html"),
                        help="output HTML path (default: dashboard/dist/dashboard.html)")
    args = parser.parse_args()

    html = _read(DASH / "index.html")
    css = _inline_safe(_read(DASH / "styles.css"), "style")
    chart = _inline_safe(_read(DASH / "vendor" / "chart.umd.min.js"), "script")
    app = _inline_safe(_read(DASH / "app.js"), "script")

    data_dir = Path(args.data_dir)
    if not (data_dir / "index.csv").exists():
        sys.exit(f"no index.csv in {data_dir} -- ingest some data there first")
    # index.csv first (the catalog), then every shard it can reference.
    names = ["index.csv"] + sorted(p.name for p in data_dir.glob("perf-*.csv"))
    blocks = []
    for name in names:
        text = _read(data_dir / name)
        if "</script" in text.lower():
            sys.exit(f"{name} contains '</script' which would break embedding; abort")
        blocks.append(f'<script type="text/csv" data-file="{name}">\n{text}\n</script>')
    data_blocks = "\n".join(blocks)

    html = _replace_once(html, '<link rel="stylesheet" href="./styles.css">',
                         f"<style>\n{css}\n</style>", "styles.css link")
    # Data blocks + inline Chart.js go where the vendored <script> was, so they
    # exist before app.js runs; app.js replaces its own external <script>.
    html = _replace_once(html, '<script src="./vendor/chart.umd.min.js"></script>',
                         f"{data_blocks}\n<script>\n{chart}\n</script>", "chart.js script")
    html = _replace_once(html, '<script src="./app.js"></script>',
                         f"<script>\n{app}\n</script>", "app.js script")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {len(names)} data files: {', '.join(names)})")


if __name__ == "__main__":
    main()
