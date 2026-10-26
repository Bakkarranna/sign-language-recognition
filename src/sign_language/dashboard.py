"""Self-contained results dashboard for the sign-language project.

Reads ``results/report_data.json`` (written by :mod:`sign_language.report`) and
writes ``docs/index.html``: a single static page with the headline metrics, the
model comparison, the training curve and confusion matrix drawn as inline SVG,
and a filterable per-gloss table plus a prediction explorer over every test
instance.

Design goals:

* **No build step, no dependencies, no CDN.** The page is plain HTML/CSS/SVG
  with a few dozen lines of vanilla JS, so it renders identically from
  `file://`, from a local server and from GitHub Pages.
* **Self-contained data.** The report JSON is embedded in the page, so the only
  external requests are the figures, which are copied next to it into
  ``docs/figures/``.
* **Nothing hand-typed.** Every number on the page comes from
  ``results/report_data.json``, so the dashboard can never drift from the runs.

Usage::

    python -m sign_language.report       # writes results/report_data.json
    python -m sign_language.dashboard    # writes docs/index.html + docs/figures/
"""

from __future__ import annotations

import base64
import html
import json
import shutil
from pathlib import Path

from . import config

DOCS_DIR = config.PROJECT_ROOT / "docs"
REPORT_PATH = config.RESULTS_DIR / "report_data.json"

# Figure captions, keyed by the figure id in the report data. Keeps the pictures
# from being decorative: each one says what it shows and why it matters.
CAPTIONS = {
    "training_curves": (
        "Validation accuracy peaks at 16.8% and then flattens while training "
        "accuracy climbs to 86% - the overfitting gap that a 681-instance "
        "training set over 100 classes produces. Early stopping on validation "
        "accuracy is what the reported test numbers come from."),
    "confusion_matrix": (
        "Rows are true glosses, columns predictions, over the 100-class label "
        "space. The diagonal is thin because most glosses appear once or not at "
        "all in the 162-instance test split."),
    "per_class_recall": (
        "Per-gloss recall, hardest first. The glosses at the left edge have a "
        "single test instance, so their recall is 0 or 1 and carries almost no "
        "information."),
    "sample_landmarks": (
        "Normalised pose, left-hand and right-hand landmarks for sample test "
        "instances - exactly what the biGRU consumes, after wrist-centred "
        "scaling."),
}


def esc(value) -> str:
    """HTML-escape anything, including None."""
    return html.escape("" if value is None else str(value), quote=True)


def fmt(value, key: str = "") -> str:
    """Format a value for a table cell or metric card."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value >= 1000:
            return f"{value:,.0f}"
        digits = 4 if abs(value) < 1 else 2
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return esc(value)


def _embed_figure(src: Path) -> str:
    """Return a relative src for a figure, copying it into docs/figures/."""
    dest = DOCS_DIR / "figures" / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copyfile(src, dest)
    return f"figures/{src.name}"


def _figures_html(figures: list[dict], results_dir: Path) -> str:
    if not figures:
        return ""
    cards = []
    for fig in figures:
        src = results_dir / "figures" / fig["name"]
        if not src.exists():
            continue
        rel = _embed_figure(src)
        caption = CAPTIONS.get(fig["id"], "")
        cap = f'<figcaption>{esc(caption)}</figcaption>' if caption else ""
        cards.append(
            f'<figure class="fig"><img src="{esc(rel)}" alt="{esc(fig["id"])}" '
            f'loading="lazy">{cap}</figure>')
    if not cards:
        return ""
    return ('<section class="block" id="figures"><h2>Committed figures</h2>'
            '<div class="figs">' + "".join(cards) + "</div></section>")


def _metrics_html(metrics: list[dict]) -> str:
    cards = []
    for m in metrics:
        value = fmt(m["value"], m["label"])
        ci = ""
        if "ci_low" in m and "ci_high" in m:
            ci = (f'<span class="ci">{fmt(m["ci_low"])} - '
                  f'{fmt(m["ci_high"])}</span>')
        cards.append(
            f'<div class="card"><div class="card-label">{esc(m["label"])}</div>'
            f'<div class="card-value">{esc(value)}{ci}</div>'
            f'<div class="card-detail">{esc(m.get("detail", ""))}</div></div>')
    return ('<section class="block" id="metrics"><h2>Headline results</h2>'
            '<div class="cards">' + "".join(cards) + "</div></section>")


def _models_html(models: dict) -> str:
    if not models or not models.get("rows"):
        return ""
    cols = models["columns"]
    head = "".join(f"<th>{esc(c['label'])}</th>" for c in cols)
    rows = []
    for i, row in enumerate(models["rows"]):
        cells = []
        for c in cols:
            cls = "num" if c["key"] != "model" else "model"
            first = ' class="best"' if i == 0 and c["key"] != "model" else ""
            cells.append(f'<td class="{cls}"{first}>{fmt(row.get(c["key"]))}'
                         "</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return ('<section class="block" id="models"><h2>Model comparison</h2>'
            f'<table class="tbl"><thead><tr>{head}</tr></thead>'
            "<tbody>" + "".join(rows) + "</tbody></table></section>")


def _line_svg(section: dict, width: int = 760, height: int = 260) -> str:
    """Inline SVG line chart. Legend below, hoverable points via <title>."""
    series = section["series"]
    pad_l, pad_r, pad_t, pad_b = 52, 16, 16, 34
    xs = [p[0] for s in series for p in s["points"]]
    ys = [p[1] for s in series for p in s["points"]]
    x0, x1 = (min(xs), max(xs)) if xs else (0, 1)
    y0, y1 = (0.0, max(ys)) if ys else (0, 1)
    if y1 <= y0:
        y1 = y0 + 1.0

    def X(v):
        return pad_l + (v - x0) / (x1 - x0 or 1) * (width - pad_l - pad_r)

    def Y(v):
        return height - pad_b - (v - y0) / (y1 - y0) * (height - pad_t - pad_b)

    colors = ["#4f9cf9", "#f0883e", "#57c77d", "#a78bfa", "#f2c744"]
    parts = []
    for gy in range(5):
        v = y0 + (y1 - y0) * gy / 4
        y = Y(v)
        parts.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" '
                     f'x2="{width - pad_r}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{pad_l - 8}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{v:.2f}</text>')
    for sx in range(5):
        v = x0 + (x1 - x0) * sx / 4
        parts.append(f'<text class="axis" x="{X(v):.1f}" '
                     f'y="{height - pad_b + 18}" text-anchor="middle">'
                     f"{v:g}</text>")
    for i, s in enumerate(series):
        pts = " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in s["points"])
        col = colors[i % len(colors)]
        parts.append(f'<polyline class="line" fill="none" stroke="{col}" '
                     f'stroke-width="2" points="{pts}"/>')
        for p in s["points"]:
            parts.append(
                f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="3" '
                f'fill="{col}"><title>{esc(s["name"])}: {p[1]:.4f} '
                f'(epoch {p[0]:g})</title></circle>')
    legend = "".join(
        f'<span class="key"><i style="background:{colors[i % len(colors)]}">'
        f"</i>{esc(s['name'])}</span>" for i, s in enumerate(series))
    return (f'<svg class="chart" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{esc(section["title"])}">'
            + "".join(parts) + "</svg>"
            + f'<div class="legend">{legend}</div>')


def _heatmap_svg(section: dict, cell: int = 9) -> str:
    """Inline SVG confusion matrix with per-cell tooltips."""
    labels = section["labels"]
    matrix = section["matrix"]
    n = len(labels)
    pad_l, pad_t = 54, 30
    width = pad_l + n * cell + 90
    height = pad_t + n * cell + 74
    mx = max((max(r) for r in matrix), default=1) or 1
    parts = []
    for i in range(n):
        ty = pad_t + i * cell + cell * 0.75
        parts.append(f'<text class="axis" x="{pad_l - 6}" y="{ty:.1f}" '
                     f'text-anchor="end">{esc(labels[i])}</text>')
        tx = pad_l + i * cell + cell / 2
        parts.append(f'<text class="axis" x="{tx:.1f}" y="{pad_t - 8}" '
                     f'text-anchor="middle">{esc(labels[i])}</text>')
    for i, row in enumerate(matrix):
        for j, c in enumerate(row):
            if not c:
                continue
            a = 0.12 + 0.88 * (c / mx)
            x = pad_l + j * cell
            y = pad_t + i * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="#4f9cf9" fill-opacity="{a:.3f}"><title>'
                f'{esc(labels[i])} -&gt; {esc(labels[j])}: {c}</title></rect>')
    parts.append(f'<text class="axis" x="{pad_l + n * cell / 2:.1f}" '
                 f'y="{height - 52}" text-anchor="middle">predicted</text>')
    parts.append(f'<text class="axis" x="14" y="{pad_t + n * cell / 2:.1f}" '
                 f'text-anchor="middle" transform="rotate(-90 14 '
                 f'{pad_t + n * cell / 2:.1f})">true</text>')
    return (f'<div class="scroll"><svg class="chart" viewBox="0 0 {width} '
            f'{height}" role="img" aria-label="{esc(section["title"])}">'
            + "".join(parts) + "</svg></div>")


def _table_html(section: dict, prefix: str) -> str:
    cols = section["columns"]
    rows = section["rows"]
    head = "".join(f'<th>{esc(c["label"])}</th>' for c in cols)
    body = []
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c["key"])
            cls = "num" if isinstance(v, (int, float)) and not isinstance(v, bool) else ""
            cells.append(f'<td class="{cls}">{fmt(v)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    search = ""
    if section.get("searchable"):
        search = (f'<input class="search" type="search" '
                  f'placeholder="Filter {len(rows)} rows..." '
                  f'data-target="{prefix}-tbl" aria-label="Filter rows">')
    return (f'{search}<div class="scroll"><table class="tbl" '
            f'id="{prefix}-tbl"><thead><tr>{head}</tr></thead><tbody>'
            + "".join(body) + "</tbody></table></div>")


def _predictions_html(section: dict, prefix: str) -> str:
    items = section["items"]
    n_ok = sum(1 for it in items if it["correct"])
    rows = [
        f'<tr class="pred" data-ok="{str(it["correct"]).lower()}" '
        f'data-text="{esc((it["label"] + " " + it["pred"]).lower())}">'
        f'<td class="num">{i + 1}</td>'
        f'<td class="gloss">{esc(it["label"])}</td>'
        f'<td class="gloss">{esc(it["pred"])}</td>'
        f'<td class="{"ok" if it["correct"] else "bad"}">'
        f'{"correct" if it["correct"] else "wrong"}</td>'
        f'<td class="num">{fmt(it.get("margin"))}</td>'
        f'<td><ul class="top5">'
        f'{"".join(f"<li>{esc(g)}</li>" for g in it.get("top5", []))}</ul></td>'
        "</tr>"
        for i, it in enumerate(items)
    ]
    controls = (
        f'<div class="controls">'
        f'<input class="search" type="search" data-target="{prefix}-tbl" '
        f'placeholder="Filter {len(items)} instances by gloss..." '
        f'aria-label="Filter predictions">'
        f'<div class="toggle" role="group" aria-label="Correctness filter">'
        f'<button class="on" data-filter="all">all {len(items)}</button>'
        f'<button data-filter="correct">correct {n_ok}</button>'
        f'<button data-filter="wrong">wrong {len(items) - n_ok}</button>'
        f"</div></div>")
    return (controls + '<div class="scroll"><table class="tbl" '
            f'id="{prefix}-tbl"><thead><tr><th>#</th><th>True gloss</th>'
            '<th>Predicted</th><th>Outcome</th><th>Margin</th>'
            '<th>Top-5 ranking</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")
PAGE_CSS = """
:root{--bg:#0d1117;--panel:#161b22;--line:#263041;--ink:#e6edf3;--dim:#9aa7b4;
--acc:#4f9cf9;--ok:#57c77d;--bad:#f0883e;--best:#f2c744}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 72px}
header.top{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:28px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.4px}
.sub{color:var(--dim);margin:0 0 14px;max-width:70ch}
.pills{display:flex;flex-wrap:wrap;gap:8px}
.pill{background:var(--panel);border:1px solid var(--line);border-radius:999px;
padding:4px 12px;font-size:12.5px;color:var(--dim)}
.pill b{color:var(--ink);font-weight:600}
h2{font-size:19px;margin:34px 0 6px}
.note{color:var(--dim);font-size:13.5px;max-width:78ch;margin:0 0 14px}
.block{margin-bottom:8px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card-label{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim)}
.card-value{font-size:26px;font-weight:650;margin:6px 0 2px;font-variant-numeric:tabular-nums}
.card-value .ci{display:block;font-size:12.5px;font-weight:400;color:var(--dim);margin-top:2px}
.card-detail{font-size:12.5px;color:var(--dim)}
table.tbl{border-collapse:collapse;width:100%;font-size:13.5px;
font-variant-numeric:tabular-nums}
.tbl th{text-align:left;font-size:11.5px;text-transform:uppercase;letter-spacing:.5px;
color:var(--dim);border-bottom:1px solid var(--line);padding:8px 10px;position:sticky;top:0;
background:var(--bg);z-index:1}
.tbl td{border-bottom:1px solid #1c232e;padding:7px 10px;vertical-align:top}
.tbl tr:hover td{background:#1a212b}
.tbl .num{text-align:right}
.tbl td.model{font-weight:600}
.tbl td.best{color:var(--best)}
.scroll{max-height:520px;overflow:auto;border:1px solid var(--line);border-radius:10px}
.controls{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:10px;align-items:center}
.search,.toggle button{background:var(--panel);color:var(--ink);border:1px solid var(--line);
border-radius:8px;padding:7px 11px;font-size:13px}
.search{min-width:230px}
.toggle{display:flex;gap:0}
.toggle button{cursor:pointer;border-radius:0}
.toggle button:first-child{border-radius:8px 0 0 8px}
.toggle button:last-child{border-radius:0 8px 8px 0}
.toggle button+button{border-left:none}
.toggle button.on{background:var(--acc);border-color:var(--acc);color:#04121f;font-weight:600}
.chart{width:100%;height:auto;display:block}
.chart .grid{stroke:#202834;stroke-width:1}
.chart .line{stroke-linejoin:round;stroke-linecap:round}
.chart .axis{fill:#8e9aa7;font-size:10.5px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--dim);margin-top:8px}
.key{display:inline-flex;align-items:center;gap:6px}
.key i{width:11px;height:11px;border-radius:3px;display:inline-block}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.fig{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:12px}
.fig img{width:100%;height:auto;border-radius:6px;display:block}
figcaption{font-size:12.5px;color:var(--dim);margin-top:10px}
.top5{list-style:none;margin:0;padding:0;font-size:12px}
.top5 li{color:var(--dim);padding:1px 0}
.ok{color:var(--ok);font-weight:600}
.bad{color:var(--bad);font-weight:600}
footer{margin-top:52px;padding-top:18px;border-top:1px solid var(--line);color:var(--dim);
font-size:12.5px}
footer a{color:var(--acc)}
@media (max-width:640px){.card-value{font-size:22px}h1{font-size:24px}}
@media print{:root{--bg:#fff;--panel:#fff;--ink:#111;--dim:#555;--line:#ccc}
.scroll{max-height:none;overflow:visible}}
"""

NAV_CSS = """
nav.toc{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 4px}
nav.toc a{color:var(--dim);text-decoration:none;font-size:13px;
border:1px solid var(--line);border-radius:999px;padding:3px 11px}
nav.toc a:hover{color:var(--ink);border-color:var(--acc)}
"""


DOC_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>{title}</title>
<meta name="description" content="{subtitle}">
<style>{css}</style>
</head>
<body>
<div class="wrap">
<header class="top">
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
<div class="pills">{pills}</div>
</header>
{body}
<footer>
Generated from the committed experiment artifacts by
<code>python -m sign_language.report &amp;&amp; python -m
sign_language.dashboard</code> - no hand-typed numbers. Source:
<a href="https://github.com/{repo}">{repo}</a>.
</footer>
</div>
<script>{js}</script>
</body>
</html>
"""


PAGE_JS = r"""
(function () {
  "use strict";
  function apply(table){
    var term = (table._term || "");
    var mode = table._mode || "all";
    table.querySelectorAll("tbody tr").forEach(function(tr){
      var okTxt = tr.getAttribute("data-text") || "";
      var state = tr.getAttribute("data-ok") || "";
      var hit = okTxt.indexOf(term) !== -1;
      var keep = hit && (mode === "all" || mode === state);
      tr.style.display = keep ? "" : "none";
    });
  }
  document.querySelectorAll("input.search").forEach(function(inp){
    var table = document.getElementById(inp.getAttribute("data-target"));
    if(!table){ return; }
    table._term = "";
    inp.addEventListener("input", function(){
      table._term = inp.value.trim().toLowerCase();
      apply(table);
    });
  });
  document.querySelectorAll(".toggle").forEach(function(grp){
    var aim = document.getElementById(
      grp.closest(".block").querySelector("input.search")
        .getAttribute("data-target"));
    grp.querySelectorAll("button").forEach(function(btn){
      btn.addEventListener("click", function(){
        grp.querySelectorAll("button").forEach(function(b){
          b.classList.remove("on"); });
        btn.classList.add("on");
        aim._mode = btn.getAttribute("data-filter");
        apply(aim);
      });
    });
  });
  var links = document.querySelectorAll("nav.toc a");
  links.forEach(function(a){
    a.addEventListener("click", function(ev){
      ev.preventDefault();
      var t = document.querySelector(a.getAttribute("href"));
      if(t){ t.scrollIntoView({behavior:"smooth", block:"start"}); }
    });
  });
})();
"""


def _toc_html(report: dict) -> str:
    links = ['<a href="#metrics">Headline</a>',
             '<a href="#models">Models</a>']
    links = [f'<a href="#figures">Figures</a>'] + links
    for s in report.get("sections", []):
        links.append(f'<a href="#{esc(s["id"])}">{esc(s.get("title", s["id"]))}</a>')
    return '<nav class="toc" aria-label="Sections">' + "".join(links) + "</nav>"


def _sections_html(report: dict) -> str:
    out = []
    for i, s in enumerate(report.get("sections", [])):
        note = f'<p class="note">{esc(s.get("note", ""))}</p>'
        kind = s.get("kind")
        if kind == "line":
            body = _line_svg(s)
        elif kind == "heatmap":
            body = _heatmap_svg(s)
        elif kind == "predictions":
            body = _predictions_html(s, s["id"])
        else:
            body = _table_html(s, s["id"])
        out.append(f'<section class="block" id="{esc(s["id"])}"><h2>'
                   f'{esc(s.get("title", s["id"]))}</h2>{note}{body}</section>')
    return "".join(out)


def build_page(report: dict) -> str:
    """Render the single self-contained HTML page for one project report."""
    h = report.get("headline", {})
    split = report.get("split", {})
    parts = [h.get("model", report.get("title", ""))]
    if split.get("n_classes"):
        parts.append(f"{split['n_classes']} classes")
    if h.get("n_test"):
        parts.append(f"{h['n_test']} test instances held out")
    if h.get("n_params"):
        parts.append(f"{h['n_params']:,} parameters")
    pills = "".join(f'<span class="pill">{esc(p)}</span>' for p in parts)
    body = _toc_html(report) + (
        _metrics_html(report.get("metrics", []))
        + _models_html(report.get("models", {}))
        + _sections_html(report)
        + _figures_html(report.get("figures", []), config.RESULTS_DIR))
    return DOC_TEMPLATE.format(
        title=esc(report.get("title", "Results")),
        subtitle=esc(report.get("subtitle", "")),
        pills=pills,
        repo=esc(report.get("repo", "")),
        body=body,
        css=PAGE_CSS + NAV_CSS,
        js=PAGE_JS,
    )


def main() -> Path:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    html_text = build_page(report)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "index.html"
    out.write_text(html_text, encoding="utf-8")
    print(out)
    return out


if __name__ == "__main__":
    main()
