#!/usr/bin/env python3
"""wiki-topic-filter 결과를 단일 HTML 파일로 렌더.

검색/필터(keep|drop, category)/정렬 가능한 정적 페이지.
- 입력:
    data/wiki-topic-filter/titles.jsonl   (preview 포함)
    data/wiki-topic-filter/judgments.jsonl
- 출력:
    data/wiki-topic-filter/results.html
"""
from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

ROOT = Path("data/wiki-topic-filter")
TITLES = ROOT / "titles.jsonl"
JUDGMENTS = ROOT / "judgments.jsonl"
OUT = ROOT / "results.html"

PREVIEW_MAX = 180  # HTML에 embed할 preview 길이


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def main() -> None:
    titles = {t["id"]: t for t in load_jsonl(TITLES)}
    judgments = load_jsonl(JUDGMENTS)

    rows = []
    for j in judgments:
        t = titles.get(j["id"], {})
        preview = (t.get("preview") or "")[:PREVIEW_MAX]
        rows.append({
            "id": j["id"],
            "title": j.get("title", ""),
            "keep": bool(j.get("keep")),
            "category": j.get("category") or "",
            "reason": j.get("reason") or "",
            "preview": preview,
        })

    n_total = len(rows)
    n_keep = sum(1 for r in rows if r["keep"])
    n_drop = n_total - n_keep
    cats = Counter(r["category"] for r in rows if r["keep"] and r["category"])
    cats_sorted = cats.most_common()

    data_js = json.dumps(rows, ensure_ascii=False)
    cats_options = "".join(
        f'<option value="{html.escape(c)}">{html.escape(c)} ({n})</option>' for c, n in cats_sorted
    )
    cat_bars = "".join(
        f'<div class="bar"><span class="bar-label">{html.escape(c)}</span>'
        f'<span class="bar-fill" style="width:{n / max(1, cats_sorted[0][1]) * 100:.1f}%">{n}</span></div>'
        for c, n in cats_sorted
    )

    page = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Wiki Topic Filter Results — {n_total} entries</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1500px; margin: 0 auto; padding: 16px; color: #111; }}
  h1 {{ margin: 0 0 12px; font-size: 18px; }}
  .summary {{ display: flex; gap: 16px; margin-bottom: 12px; flex-wrap: wrap; }}
  .summary .stat {{ background: #f3f4f6; padding: 6px 12px; border-radius: 6px; font-size: 13px; }}
  .summary .stat b {{ font-size: 16px; margin-right: 4px; }}
  .cats {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px; margin-bottom: 12px; max-width: 720px; }}
  .bar {{ display: flex; align-items: center; gap: 8px; font-size: 12px; }}
  .bar-label {{ width: 110px; color: #444; }}
  .bar-fill {{ background: #93c5fd; padding: 2px 6px; border-radius: 3px; color: #1e3a8a; text-align: right; min-width: 24px; }}
  .filters {{ display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; align-items: center; font-size: 13px; }}
  input[type=text] {{ padding: 6px 10px; width: 260px; font-size: 13px; }}
  select {{ padding: 6px; font-size: 13px; }}
  button {{ padding: 5px 10px; cursor: pointer; font-size: 12px; border: 1px solid #d1d5db; background: white; border-radius: 4px; }}
  button.active {{ background: #2563eb; color: white; border-color: #2563eb; }}
  #count {{ margin-left: auto; color: #6b7280; font-size: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th, td {{ padding: 5px 8px; border-bottom: 1px solid #eee; vertical-align: top; text-align: left; }}
  th {{ background: #f9fafb; position: sticky; top: 0; cursor: pointer; user-select: none; }}
  th:hover {{ background: #f3f4f6; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10.5px; font-weight: 600; }}
  .keep {{ background: #dcfce7; color: #166534; }}
  .drop {{ background: #fee2e2; color: #991b1b; }}
  .cat {{ background: #e0e7ff; color: #3730a3; padding: 1px 5px; border-radius: 3px; font-size: 10.5px; }}
  .preview {{ color: #6b7280; max-width: 460px; }}
  .reason {{ color: #374151; max-width: 200px; }}
  .title {{ font-weight: 500; }}
  .footer {{ margin-top: 12px; color: #9ca3af; font-size: 11px; text-align: center; }}
</style>
</head>
<body>
<h1>Wiki Topic Filter — SimpleWiki {n_total} titles → DeepSeek v4 Flash 분류</h1>

<div class="summary">
  <div class="stat"><b>{n_total:,}</b> total</div>
  <div class="stat"><b style="color:#166534">{n_keep:,}</b> keep ({n_keep / n_total * 100:.1f}%)</div>
  <div class="stat"><b style="color:#991b1b">{n_drop:,}</b> drop ({n_drop / n_total * 100:.1f}%)</div>
  <div class="stat">{len(cats_sorted)} categories</div>
</div>

<div class="cats">{cat_bars}</div>

<div class="filters">
  <input id="q" placeholder="title / reason / preview 검색...">
  <button data-f="all" class="active">전체</button>
  <button data-f="keep">KEEP only</button>
  <button data-f="drop">DROP only</button>
  <select id="cat"><option value="">all categories</option>{cats_options}</select>
  <span id="count"></span>
</div>

<table id="t">
  <thead><tr>
    <th data-k="id">id</th>
    <th data-k="keep">keep</th>
    <th data-k="category">category</th>
    <th data-k="title">title</th>
    <th data-k="reason">reason</th>
    <th>preview</th>
  </tr></thead>
  <tbody id="rows"></tbody>
</table>

<div class="footer">생성: scripts/build_wiki_topic_filter_html.py</div>

<script>
const DATA = {data_js};
let state = {{ filter: 'all', cat: '', q: '', sortKey: 'id', sortAsc: true }};
const $ = id => document.getElementById(id);

function rowHtml(r) {{
  const keep = r.keep
    ? '<span class="badge keep">KEEP</span>'
    : '<span class="badge drop">DROP</span>';
  const cat = r.category ? `<span class="cat">${{r.category}}</span>` : '';
  const esc = s => (s ?? '').toString()
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  return `<tr>
    <td>${{r.id}}</td>
    <td>${{keep}}</td>
    <td>${{cat}}</td>
    <td class="title">${{esc(r.title)}}</td>
    <td class="reason">${{esc(r.reason)}}</td>
    <td class="preview">${{esc(r.preview)}}</td>
  </tr>`;
}}

function render() {{
  const q = state.q.toLowerCase();
  let rows = DATA.filter(r => {{
    if (state.filter === 'keep' && !r.keep) return false;
    if (state.filter === 'drop' && r.keep) return false;
    if (state.cat && r.category !== state.cat) return false;
    if (q) {{
      const hay = `${{r.title}} ${{r.reason}} ${{r.preview}}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }}
    return true;
  }});
  const k = state.sortKey;
  rows.sort((a, b) => {{
    let av = a[k], bv = b[k];
    if (typeof av === 'string') av = av.toLowerCase();
    if (typeof bv === 'string') bv = bv.toLowerCase();
    return (av < bv ? -1 : av > bv ? 1 : 0) * (state.sortAsc ? 1 : -1);
  }});
  const MAX = 2000;
  const head = rows.slice(0, MAX);
  $('rows').innerHTML = head.map(rowHtml).join('');
  const more = rows.length > MAX ? ` (showing first ${{MAX}})` : '';
  $('count').textContent = `${{rows.length.toLocaleString()}} / ${{DATA.length.toLocaleString()}}${{more}}`;
}}

$('q').addEventListener('input', e => {{ state.q = e.target.value; render(); }});
$('cat').addEventListener('change', e => {{ state.cat = e.target.value; render(); }});
document.querySelectorAll('.filters button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.filter = btn.dataset.f;
    render();
  }});
}});
document.querySelectorAll('th[data-k]').forEach(th => {{
  th.addEventListener('click', () => {{
    const k = th.dataset.k;
    state.sortAsc = state.sortKey === k ? !state.sortAsc : true;
    state.sortKey = k;
    render();
  }});
}});

render();
</script>
</body>
</html>
"""
    OUT.write_text(page)
    print(f"wrote {OUT}  size={OUT.stat().st_size / 1024:.0f} KB  entries={n_total}")


if __name__ == "__main__":
    main()
