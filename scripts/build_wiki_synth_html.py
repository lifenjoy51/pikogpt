#!/usr/bin/env python3
"""wiki-synth 결과를 단일 HTML 파일로 렌더 (검수용).

- 입력:
    data/wiki-synth/raw.jsonl
    data/wiki-topic-filter/titles.jsonl  (preview)
    data/external-all-raw/wiki.txt       (원문 일부)
- 출력:
    data/wiki-synth/wiki.html
"""
from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

ROOT = Path("data/wiki-synth")
RAW = ROOT / "raw.jsonl"
VALIDATION = ROOT / "validation.jsonl"
TITLES = Path("data/wiki-topic-filter/titles.jsonl")
WIKI_BODY = Path("data/external-all-raw/wiki.txt")
OUT = ROOT / "wiki.html"

ARTICLE_PREVIEW_CHARS = 600   # HTML에 보여줄 원문 일부 길이


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def load_wiki_bodies() -> dict[int, str]:
    out: dict[int, str] = {}
    with WIKI_BODY.open() as f:
        for i, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            out[i] = line
    return out


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"missing {RAW} — run scripts/run_wiki_synth.py first")

    raws = load_jsonl(RAW)
    # 중복 id시 latest 우선
    by_id: dict[int, dict] = {}
    for r in raws:
        by_id[r["id"]] = r
    rows_src = sorted(by_id.values(), key=lambda r: r["id"])

    bodies = load_wiki_bodies()
    validations: dict[int, dict] = {}
    if VALIDATION.exists():
        for ln in VALIDATION.read_text().splitlines():
            if ln.strip():
                v = json.loads(ln)
                validations[v["id"]] = v

    rows = []
    for r in rows_src:
        body = bodies.get(r["id"], "")
        body_chars = len(body)
        body_preview = body.replace("\\n", " ")[:ARTICLE_PREVIEW_CHARS]
        text = r.get("text") or ""
        text_chars = len(text)
        ratio = (text_chars / body_chars * 100) if body_chars else 0.0
        v = validations.get(r["id"], {})
        rows.append({
            "id": r["id"],
            "title": r.get("title", ""),
            "category": r.get("category") or "",
            "ok": bool(r.get("ok")),
            "error": r.get("error") or "",
            "text": text,
            "chars": text_chars,
            "words": len(text.split()),
            "src_chars": body_chars,
            "ratio": ratio,
            "elapsed": (r.get("meta") or {}).get("elapsed_s") or 0,
            "body_preview": body_preview,
            "hard": v.get("hard_count", 0),
            "hard_ratio": v.get("hard_ratio", 0.0),
            "hard_words": v.get("hard_words", []),
            "sent_words": v.get("avg_sent_words", 0.0),
            "issues": v.get("issues", []),
        })

    n_total = len(rows)
    n_ok = sum(1 for r in rows if r["ok"])
    n_fail = n_total - n_ok
    cats = Counter(r["category"] for r in rows if r["ok"] and r["category"])
    cats_sorted = cats.most_common()
    avg_chars = (sum(r["chars"] for r in rows if r["ok"]) / max(1, n_ok))
    avg_words = (sum(r["words"] for r in rows if r["ok"]) / max(1, n_ok))
    avg_src = (sum(r["src_chars"] for r in rows if r["ok"]) / max(1, n_ok))
    avg_ratio = (sum(r["ratio"] for r in rows if r["ok"]) / max(1, n_ok))
    total_src = sum(r["src_chars"] for r in rows if r["ok"])
    total_text = sum(r["chars"] for r in rows if r["ok"])
    total_ratio = (total_text / total_src * 100) if total_src else 0.0
    avg_hard = (sum(r["hard"] for r in rows if r["ok"]) / max(1, n_ok))
    avg_hard_ratio = (sum(r["hard_ratio"] for r in rows if r["ok"]) / max(1, n_ok))
    n_with_issue = sum(1 for r in rows if r["ok"] and r["issues"])
    issue_counter: dict[str, int] = {}
    for r in rows:
        if not r["ok"]:
            continue
        for issue in r["issues"]:
            issue_counter[issue] = issue_counter.get(issue, 0) + 1
    issues_sorted = sorted(issue_counter.items(), key=lambda x: -x[1])

    data_js = json.dumps(rows, ensure_ascii=False)
    cats_options = "".join(
        f'<option value="{html.escape(c)}">{html.escape(c)} ({n})</option>' for c, n in cats_sorted
    )
    issue_options = (
        f'<option value="any">any issue ({n_with_issue})</option>'
        + "".join(
            f'<option value="{html.escape(k)}">{html.escape(k)} ({n})</option>'
            for k, n in issues_sorted
        )
    )
    cat_bars = "".join(
        f'<div class="bar"><span class="bar-label">{html.escape(c)}</span>'
        f'<span class="bar-fill" style="width:{n / max(1, cats_sorted[0][1]) * 100:.1f}%">{n}</span></div>'
        for c, n in cats_sorted
    ) if cats_sorted else ""

    page = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Wiki Synth Results — {n_total} entries</title>
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
  input[type=text] {{ padding: 6px 10px; width: 280px; font-size: 13px; }}
  select {{ padding: 6px; font-size: 13px; }}
  button {{ padding: 5px 10px; cursor: pointer; font-size: 12px; border: 1px solid #d1d5db; background: white; border-radius: 4px; }}
  button.active {{ background: #2563eb; color: white; border-color: #2563eb; }}
  #count {{ margin-left: auto; color: #6b7280; font-size: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; text-align: left; }}
  th {{ background: #f9fafb; position: sticky; top: 0; cursor: pointer; user-select: none; }}
  th:hover {{ background: #f3f4f6; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10.5px; font-weight: 600; }}
  .ok {{ background: #dcfce7; color: #166534; }}
  .fail {{ background: #fee2e2; color: #991b1b; }}
  .cat {{ background: #e0e7ff; color: #3730a3; padding: 1px 5px; border-radius: 3px; font-size: 10.5px; }}
  .num {{ text-align: right; color: #6b7280; font-variant-numeric: tabular-nums; }}
  .text-cell {{ max-width: 500px; white-space: pre-wrap; }}
  .body-cell {{ max-width: 320px; color: #6b7280; font-size: 11.5px; }}
  .hardw {{ color: #92400e; font-size: 10.5px; max-width: 180px; }}
  .issues-cell {{ max-width: 160px; }}
  .issue {{ display: inline-block; padding: 1px 5px; margin: 1px; border-radius: 3px; background: #fef3c7; color: #92400e; font-size: 10.5px; }}
  .long-sent {{ background: #fde68a; padding: 0 2px; border-radius: 2px; }}
  .title {{ font-weight: 500; }}
  .err {{ color: #991b1b; font-size: 11px; }}
  .footer {{ margin-top: 12px; color: #9ca3af; font-size: 11px; text-align: center; }}
</style>
</head>
<body>
<h1>Wiki Synth — kept_titles → 5세 explainer ({n_total} entries)</h1>

<div class="summary">
  <div class="stat"><b>{n_total:,}</b> total</div>
  <div class="stat"><b style="color:#166534">{n_ok:,}</b> ok</div>
  <div class="stat"><b style="color:#991b1b">{n_fail:,}</b> fail</div>
  <div class="stat">avg <b>{avg_chars:.0f}</b> chars (synth)</div>
  <div class="stat">avg <b>{avg_words:.1f}</b> words</div>
  <div class="stat">avg <b>{avg_src:.0f}</b> chars (src)</div>
  <div class="stat">avg ratio <b>{avg_ratio:.1f}%</b></div>
  <div class="stat">total <b>{total_text:,}</b> / <b>{total_src:,}</b> chars = <b>{total_ratio:.1f}%</b></div>
  <div class="stat">avg <b>{avg_hard:.1f}</b> hard words ({avg_hard_ratio:.1f}%)</div>
  <div class="stat"><b>{n_with_issue:,}</b> with issues ({n_with_issue / max(1, n_ok) * 100:.1f}%)</div>
  <div class="stat">{len(cats_sorted)} categories</div>
</div>

<div class="cats">{cat_bars}</div>

<div class="filters">
  <input id="q" placeholder="title / text / body / hard / issue 검색...">
  <button data-f="all" class="active">전체</button>
  <button data-f="ok">OK only</button>
  <button data-f="fail">FAIL only</button>
  <select id="issue"><option value="">issue 필터 없음</option>{issue_options}</select>
  <select id="cat"><option value="">all categories</option>{cats_options}</select>
  <span id="count"></span>
</div>

<table id="t">
  <thead><tr>
    <th data-k="id">id</th>
    <th data-k="ok">ok</th>
    <th data-k="category">category</th>
    <th data-k="title">title</th>
    <th data-k="words">words</th>
    <th data-k="chars">chars</th>
    <th data-k="src_chars">src chars</th>
    <th data-k="ratio">ratio</th>
    <th data-k="hard">hard</th>
    <th data-k="sent_words">sent w</th>
    <th>issues</th>
    <th>text</th>
    <th>article (source)</th>
  </tr></thead>
  <tbody id="rows"></tbody>
</table>

<div class="footer">생성: scripts/build_wiki_synth_html.py</div>

<script>
const DATA = {data_js};
let state = {{ filter: 'all', issue: '', cat: '', q: '', sortKey: 'id', sortAsc: true }};
const $ = id => document.getElementById(id);
const esc = s => (s ?? '').toString()
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

const CLAUSE_LONG_THRESHOLD = 20;
function highlightLongSentences(text) {{
  // sub-clause 단위 split (.!?,;: 모두 break). separator는 캡처해서 그대로 유지.
  const parts = text.split(/([.!?,;:]+\s*)/);
  let out = '';
  for (const p of parts) {{
    if (!p) continue;
    if (/^[.!?,;:]/.test(p)) {{ out += esc(p); continue; }}  // separator는 그대로
    const wc = (p.match(/[A-Za-z]+/g) || []).length;
    if (wc > CLAUSE_LONG_THRESHOLD) {{
      out += `<span class="long-sent">${{esc(p)}}</span>`;
    }} else {{
      out += esc(p);
    }}
  }}
  return out;
}}

function rowHtml(r) {{
  const ok = r.ok ? '<span class="badge ok">OK</span>' : '<span class="badge fail">FAIL</span>';
  const cat = r.category ? `<span class="cat">${{esc(r.category)}}</span>` : '';
  const textHtml = r.ok ? highlightLongSentences(r.text) : `<span class="err">${{esc(r.error)}}</span>`;
  const ratio = (r.ratio || 0).toFixed(1);
  const hardCell = `${{r.hard}} (${{(r.hard_ratio || 0).toFixed(1)}}%)` +
    (r.hard_words && r.hard_words.length
      ? `<div class="hardw">${{r.hard_words.map(esc).join(', ')}}</div>` : '');
  const issuesHtml = (r.issues || []).map(i =>
    `<span class="issue">${{esc(i)}}</span>`).join(' ');
  return `<tr>
    <td>${{r.id}}</td>
    <td>${{ok}}</td>
    <td>${{cat}}</td>
    <td class="title">${{esc(r.title)}}</td>
    <td class="num">${{r.words}}</td>
    <td class="num">${{r.chars}}</td>
    <td class="num">${{r.src_chars.toLocaleString()}}</td>
    <td class="num">${{ratio}}%</td>
    <td class="num">${{hardCell}}</td>
    <td class="num">${{(r.sent_words || 0).toFixed(1)}}</td>
    <td class="issues-cell">${{issuesHtml}}</td>
    <td class="text-cell">${{textHtml}}</td>
    <td class="body-cell">${{esc(r.body_preview)}}</td>
  </tr>`;
}}

function render() {{
  const q = state.q.toLowerCase();
  let rows = DATA.filter(r => {{
    if (state.filter === 'ok' && !r.ok) return false;
    if (state.filter === 'fail' && r.ok) return false;
    if (state.issue === 'any' && (!r.issues || r.issues.length === 0)) return false;
    if (state.issue && state.issue !== 'any' && !(r.issues || []).includes(state.issue)) return false;
    if (state.cat && r.category !== state.cat) return false;
    if (q) {{
      const hay = `${{r.title}} ${{r.text}} ${{r.body_preview}} ${{(r.hard_words||[]).join(' ')}} ${{(r.issues||[]).join(' ')}}`.toLowerCase();
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
$('issue').addEventListener('change', e => {{ state.issue = e.target.value; render(); }});
document.querySelectorAll('.filters button[data-f]').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.filters button[data-f]').forEach(b => b.classList.remove('active'));
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
