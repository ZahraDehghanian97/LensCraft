"""An offline index for choosing exported qualitative paper figures."""
from __future__ import annotations

import html
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlsplit


_COHORTS = {"static": "Static subject", "dynamic": "Dynamic subject"}
_TASKS = {"prompt": "Prompt", "keyframe": "Keyframe", "conflict": "Conflict"}
_ASSETS = {"png": "PNG", "pdf": "PDF", "svg": "SVG", "metadata": "Metadata"}


def _relative_path(value: Any) -> str:
    """Keep gallery links inside the exported directory, including file:// use."""
    path = str(value)
    parts = PurePosixPath(path).parts
    parsed = urlsplit(path)
    if (not parts or path.startswith(("/", "\\")) or "\\" in path
            or ".." in parts or parsed.scheme or parsed.netloc
            or any(ord(char) < 32 for char in path)):
        raise ValueError(f"Gallery assets must use relative paths: {path!r}")
    return path


def _href(path: str) -> str:
    # Quote literal filename characters such as #, ? and %, not just HTML.
    return html.escape(quote(path, safe="/"), quote=True)


def _normalize(entry: Mapping[str, Any]) -> dict[str, Any]:
    item = {field: str(entry[field]) for field in ("id", "sample_id", "cohort", "task", "prompt")}
    if not item["id"] or not item["sample_id"]:
        raise ValueError("Gallery entries need non-empty id and sample_id values.")
    if item["cohort"] not in _COHORTS or item["task"] not in _TASKS:
        raise ValueError("Gallery cohort must be static/dynamic and task must be prompt/keyframe/conflict.")
    if entry.get("dataset") is not None:
        item["dataset"] = str(entry["dataset"])
    item["camera_motion"] = str(entry.get("camera_motion") or "")
    item["assets"] = {name: _relative_path(entry["assets"][name])
                      for name in _ASSETS if entry["assets"].get(name)}
    if "png" not in item["assets"]:
        raise ValueError("Gallery entries need a PNG preview.")
    item["bundle"] = _relative_path(entry["bundle"])
    return item


def _card(entry: Mapping[str, Any]) -> str:
    escape = html.escape
    links = "".join(f'<a href="{_href(path)}">{_ASSETS[name]}</a>'
                    for name, path in entry["assets"].items())
    motion = f'<span class="badge">{escape(entry["camera_motion"])}</span>' if entry["camera_motion"] else ""
    dataset = f'<span class="badge">{escape(entry["dataset"])}</span>' if entry.get("dataset") else ""
    return f'''<article class="card" data-id="{escape(entry['id'], quote=True)}">
  <a class="preview" href="{_href(entry['assets']['png'])}" aria-label="Open figure for {escape(entry['sample_id'], quote=True)}">
    <img src="{_href(entry['assets']['png'])}" alt="{escape(_COHORTS[entry['cohort']] + ': ' + entry['prompt'], quote=True)}" loading="lazy" decoding="async">
  </a>
  <div class="card-body">
    <div class="badges"><span class="badge candidate-id">{escape(entry['id'])}</span><span class="badge">{_COHORTS[entry['cohort']]}</span><span class="badge">{_TASKS[entry['task']]}</span>{motion}{dataset}</div>
    <h2>{escape(entry['sample_id'])}</h2>
    <p class="prompt" dir="auto">{escape(entry['prompt'])}</p>
    <div class="card-footer"><label class="pick"><input type="checkbox" class="selection" aria-label="Select {escape(entry['id'], quote=True)}"> Select figure</label><a href="{_href(entry['bundle'])}" download>Bundle</a></div>
    <nav class="assets" aria-label="Formats for {escape(entry['id'], quote=True)}">{links}</nav>
  </div>
</article>'''


_STYLE = """
:root { color-scheme: light; font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #253044; background: #f4f5f7; }
* { box-sizing: border-box; }
body { margin: 0; }
main { max-width: 1540px; margin: auto; padding: 40px 28px 72px; }
h1 { font-size: clamp(27px, 4vw, 38px); letter-spacing: -.04em; line-height: 1.15; margin: 6px 0 14px; }
.eyebrow { text-transform: uppercase; letter-spacing: .15em; font-size: 11px; font-weight: 750; color: #607086; }
.intro { color: #607086; max-width: 800px; margin: 0 0 18px; }
.totals { display: flex; flex-wrap: wrap; gap: 8px 24px; margin-bottom: 26px; }
.totals strong { font-size: 22px; margin-right: 4px; }
.toolbar { display: flex; flex-wrap: wrap; align-items: end; gap: 14px; padding: 18px; border: 1px solid #dce1e7; border-radius: 12px; background: white; }
.field { display: grid; gap: 5px; font-size: 12px; font-weight: 650; }
.search-field { flex: 1; min-width: 220px; }
select, input[type=search], button { font: inherit; padding: 9px 12px; border: 1px solid #cbd3df; border-radius: 6px; background: white; color: inherit; }
select, input[type=search] { width: 100%; font-size: 14px; font-weight: 400; }
button { cursor: pointer; font-size: 13px; }
button:hover { background: #eef2f7; }
button:disabled { opacity: .5; cursor: default; }
button.primary { background: #253f65; color: white; border-color: #253f65; }
:focus-visible { outline: 3px solid #92b5e6; outline-offset: 3px; }
.selection-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin: 16px 0 6px; }
.selection-bar strong { margin-right: auto; }
.status-line { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 5px 20px; margin: 0 0 18px; font-size: 12px; color: #607086; }
.gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 410px), 1fr)); gap: 22px; align-items: start; }
.card { min-width: 0; background: white; border: 1px solid #dce1e7; border-radius: 12px; overflow: hidden; }
.card.selected { border-color: #356bae; box-shadow: 0 0 0 2px #356bae; }
.card[hidden], [hidden] { display: none !important; }
.preview { display: flex; align-items: center; justify-content: center; height: 330px; background: #fff; border-bottom: 1px solid #e9edf2; padding: 10px; }
.preview img { display: block; width: 100%; height: 100%; object-fit: contain; }
.card-body { padding: 18px; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.badge { font-size: 11px; padding: 3px 8px; border-radius: 4px; background: #edf1f6; color: #53657b; }
.candidate-id { background: #253f65; color: white; font-weight: 650; overflow-wrap: anywhere; }
h2 { font-size: 15px; margin: 0 0 9px; overflow-wrap: anywhere; }
.prompt { font-size: 13px; margin: 0 0 20px; color: #526078; white-space: pre-wrap; overflow-wrap: anywhere; }
.card-footer { display: flex; align-items: center; justify-content: space-between; gap: 12px; font-size: 13px; }
.pick { display: inline-flex; align-items: center; gap: 8px; cursor: pointer; }
input[type=checkbox] { accent-color: #356bae; width: 17px; height: 17px; margin: 0; }
a { color: #285b99; text-underline-offset: 3px; }
.assets { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 16px; padding-top: 12px; border-top: 1px solid #edf0f4; font-size: 12px; }
.empty { text-align: center; color: #607086; padding: 48px 0; }
noscript p { padding: 12px; background: #fff; }
@media (max-width: 620px) { main { padding: 26px 14px 48px; } .toolbar .field { flex: 1; min-width: 140px; } .toolbar .search-field { flex-basis: 100%; } .preview { height: 290px; } }
@media print { .toolbar, .selection-bar, .status-line, .pick { display: none; } main { padding: 0; } .gallery { grid-template-columns: 1fr 1fr; } .card { break-inside: avoid; } .preview { height: 230px; } }
"""


_SCRIPT = """
(() => {
  'use strict';
  const entries = JSON.parse(document.getElementById('gallery-data').textContent);
  const byId = new Map(entries.map(entry => [entry.id, entry]));
  const sampleKey = entry => JSON.stringify([entry.dataset || '', entry.sample_id]);
  const cards = Array.from(document.querySelectorAll('.card'));
  const cohort = document.getElementById('cohort');
  const task = document.getElementById('task');
  const search = document.getElementById('search');
  const download = document.getElementById('download');
  const storageKey = 'lenscraft-paper-selection-v1:' + location.pathname;
  let selected = new Set();
  try {
    const stored = JSON.parse(localStorage.getItem(storageKey) || '[]');
    if (Array.isArray(stored)) selected = new Set(stored.filter(id => byId.has(id)));
  } catch (_) {
    document.getElementById('storage-note').textContent = 'Download your selection to keep a copy.';
  }
  function refreshSelection() {
    for (const card of cards) {
      const picked = selected.has(card.dataset.id);
      card.classList.toggle('selected', picked);
      card.querySelector('.selection').checked = picked;
    }
    document.getElementById('selected-count').textContent = selected.size + ' selected';
    download.disabled = selected.size === 0;
    document.getElementById('clear').disabled = selected.size === 0;
    try { localStorage.setItem(storageKey, JSON.stringify(Array.from(selected))); }
    catch (_) { document.getElementById('storage-note').textContent = 'Download your selection to keep a copy.'; }
  }
  function filter() {
    const query = search.value.trim().toLocaleLowerCase();
    const sampleIds = new Set();
    let visible = 0;
    for (const card of cards) {
      const entry = byId.get(card.dataset.id);
      const matches = (!cohort.value || entry.cohort === cohort.value)
        && (!task.value || entry.task === task.value)
        && (!query || [entry.id, entry.sample_id, entry.prompt, entry.camera_motion].join(' ').toLocaleLowerCase().includes(query));
      card.hidden = !matches;
      if (matches) { visible++; sampleIds.add(sampleKey(entry)); }
    }
    document.getElementById('visible-count').textContent = visible + ' figures from ' + sampleIds.size + ' unique samples shown';
    document.getElementById('empty').hidden = visible !== 0;
    document.getElementById('select-visible').disabled = visible === 0;
  }
  for (const card of cards) {
    card.querySelector('.selection').addEventListener('change', event => {
      if (event.target.checked) selected.add(card.dataset.id);
      else selected.delete(card.dataset.id);
      refreshSelection();
    });
  }
  cohort.addEventListener('change', filter);
  task.addEventListener('change', filter);
  search.addEventListener('input', filter);
  document.getElementById('select-visible').addEventListener('click', () => {
    cards.filter(card => !card.hidden).forEach(card => selected.add(card.dataset.id));
    refreshSelection();
  });
  document.getElementById('clear').addEventListener('click', () => { selected.clear(); refreshSelection(); });
  download.addEventListener('click', () => {
    const chosen = entries.filter(entry => selected.has(entry.id));
    const payload = {
      schema_version: 1,
      purpose: 'Qualitative paper figure selection',
      figure_count: chosen.length,
      unique_sample_count: new Set(chosen.map(sampleKey)).size,
      entries: chosen,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2) + '\\n'], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'paper-selection.json';
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  refreshSelection();
  filter();
})();
"""


def write_gallery(entries: Sequence[Mapping[str, Any]], output_dir: str | Path) -> Path:
    """Write ``index.html`` with local previews, filters and a downloadable shortlist.

    Each entry has ``id``, ``sample_id``, ``cohort`` (static/dynamic), ``task``
    (prompt/keyframe/conflict), ``prompt``, optional ``camera_motion``, ``assets``
    (png and optional pdf/svg/metadata), and ``bundle``. All paths are relative
    to ``output_dir``. A sample may have several figure variants; sample counts
    deduplicate ``(dataset, sample_id)`` (dataset is optional), while selection
    uses the unique figure ``id``.
    """
    items = [_normalize(entry) for entry in entries]
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("Gallery figure IDs must be unique.")
    # Script data is text, not executable JavaScript. Escaping '<' also prevents
    # user-supplied </script> strings from terminating the data element.
    data = json.dumps(items, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    counts = {cohort: sum(item["cohort"] == cohort for item in items) for cohort in _COHORTS}
    unique = len({(item.get("dataset", ""), item["sample_id"]) for item in items})
    cards = "\n".join(_card(item) for item in items)
    content = f'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LensCraft · Paper figure candidates</title><style>{_STYLE}</style></head>
<body><main>
<header><div class="eyebrow">LensCraft / Qualitative figures</div><h1>Paper figure candidates</h1>
<p class="intro">Browse, compare and shortlist figures for the paper. These candidates are for qualitative selection; they are not a quantitative ranking. Multiple figures can share the same underlying sample.</p>
<div class="totals"><span><strong id="figure-total">{len(items)}</strong> figures</span><span><strong id="sample-total">{unique}</strong> unique samples</span><span><strong>{counts['static']}</strong> static</span><span><strong>{counts['dynamic']}</strong> dynamic</span></div></header>
<section class="toolbar" aria-label="Filter figures">
<label class="field search-field" for="search">Search<input id="search" type="search" placeholder="Prompt, sample ID or camera motion"></label>
<label class="field" for="cohort">Subject<select id="cohort"><option value="">All subjects</option value="static">Static subject</option><option value="dynamic">Dynamic subject</option></select></label>
<label class="field" for="task">Task<select id="task"><option value="">All tasks</option><option value="prompt">Prompt</option><option value="keyframe">Keyframe</option><option value="conflict">Conflict</option></select></label>
</section>
<div class="selection-bar"><strong id="selected-count" aria-live="polite">0 selected</strong><button id="select-visible" type="button">Select visible</button><button id="clear" type="button" disabled>Clear selection</button><button id="download" class="primary" type="button" disabled>Download selection JSON</button></div>
<div class="status-line"><span id="visible-count" aria-live="polite">{len(items)} figures from {unique} unique samples shown</span><span id="storage-note">Selection is saved in this browser. Download JSON to share it.</span></div>
<noscript><p>Enable JavaScript to filter and save a selection. Figure previews and file links work without it.</p></noscript>
<section class="gallery" aria-label="Figure candidates">{cards}</section>
<p id="empty" class="empty"{' hidden' if items else ''}>No figures match these filters.</p>
</main><script id="gallery-data" type="application/json">{data}</script><script>{_SCRIPT}</script></body></html>
'''
    destination = Path(output_dir) / "index.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination
