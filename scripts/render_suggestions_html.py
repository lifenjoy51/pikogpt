#!/usr/bin/env python3
"""suggestions_unique.jsonl → suggestions.html (체크박스 UI)."""
import json
from collections import Counter
from pathlib import Path

import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else ""  # "pro" 또는 "" (flash default)
SUFFIX = f"_{TAG}" if TAG else ""
JSONL = Path(f"data/lemma-anchor-filter/suggestions_unique{SUFFIX}.jsonl")
OUT = Path(f"data/lemma-anchor-filter/suggestions{SUFFIX}.html")


def main() -> None:
    rows = [json.loads(ln) for ln in JSONL.read_text().splitlines() if ln.strip()]
    cat_counts = Counter(r.get("category", "?") for r in rows)
    max_cat = max(cat_counts.values()) if cat_counts else 1
    cat_bars = "".join(
        f'<div class="bar"><span class="bar-label">{c}</span>'
        f'<span class="bar-fill" style="width:{n / max_cat * 100:.1f}%">{n}</span></div>'
        for c, n in cat_counts.most_common()
    )
    rows_json = json.dumps(
        [{"lemma": r["lemma"], "category": r.get("category", ""), "reason": r.get("reason", "")}
         for r in rows],
        ensure_ascii=False,
    )
    html = f"""<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8">
<title>Lemma Anchor Suggestions — 기존 3047개에 없는 신규 추천</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 16px; color: #1f2937; }}
  h1 {{ font-size: 18px; margin: 0 0 12px 0; }}
  .summary {{ display: flex; gap: 16px; margin-bottom: 12px; padding: 10px 14px; background: #f3f4f6; border-radius: 6px; font-size: 13px; flex-wrap: wrap; }}
  .summary .stat b {{ font-size: 16px; margin-right: 4px; }}
  .cats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; margin-bottom: 12px; max-width: 900px; }}
  .bar {{ display: flex; align-items: center; gap: 8px; font-size: 12px; }}
  .bar-label {{ width: 90px; color: #444; }}
  .bar-fill {{ background: #93c5fd; padding: 2px 6px; border-radius: 3px; color: #1e3a8a; text-align: right; min-width: 24px; }}
  .filters {{ display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; align-items: center; font-size: 13px; }}
  input[type=text] {{ padding: 6px 10px; width: 260px; font-size: 13px; }}
  select {{ padding: 6px; font-size: 13px; }}
  button {{ padding: 5px 10px; cursor: pointer; font-size: 12px; border: 1px solid #d1d5db; background: white; border-radius: 4px; }}
  button.primary {{ background: #2563eb; color: white; border-color: #2563eb; }}
  #count {{ margin-left: auto; color: #6b7280; font-size: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: middle; text-align: left; }}
  th {{ background: #f9fafb; cursor: pointer; user-select: none; }}
  .lemma {{ font-weight: 500; font-family: ui-monospace, monospace; }}
  .cat {{ background: #e0e7ff; color: #3730a3; padding: 1px 5px; border-radius: 3px; font-size: 11px; }}
  .reason {{ color: #374151; }}
  tr.kept {{ background: #ecfdf5; }}
  tr.rejected {{ background: #fef2f2; opacity: 0.5; }}
  #output {{ margin-top: 12px; padding: 10px; background: #f3f4f6; border-radius: 6px; font-family: ui-monospace, monospace; font-size: 12px; white-space: pre-wrap; max-height: 200px; overflow: auto; }}
</style></head><body>

<h1>Lemma Anchor Suggestions — 기존 3,047개에 없는 신규 추천</h1>

<div class="summary">
  <div class="stat"><b>{len(rows)}</b> suggestions</div>
  <div class="stat">{len(cat_counts)} categories</div>
  <div class="stat">model: deepseek-v4-flash (1 batch, requested 1000, returned 104→{len(rows)} unique)</div>
</div>

<div class="cats">{cat_bars}</div>

<div class="filters">
  <input id="q" placeholder="lemma / reason 검색...">
  <select id="cat"><option value="">all categories</option></select>
  <button id="keepAll">전체 KEEP</button>
  <button id="clearAll">전체 해제</button>
  <button id="export" class="primary">선택 export (clipboard)</button>
  <span id="count"></span>
</div>

<table id="tbl">
  <thead><tr>
    <th>KEEP</th>
    <th data-sort="lemma">lemma</th>
    <th data-sort="category">category</th>
    <th data-sort="reason">reason</th>
  </tr></thead>
  <tbody></tbody>
</table>

<div id="output"></div>

<script>
const ROWS = {rows_json};
const tbody = document.querySelector("#tbl tbody");
const countEl = document.querySelector("#count");
const qInput = document.querySelector("#q");
const catSel = document.querySelector("#cat");
const out = document.querySelector("#output");

const cats = [...new Set(ROWS.map(r => r.category).filter(Boolean))].sort();
for (const c of cats) {{
  const o = document.createElement("option"); o.value = c; o.textContent = c; catSel.appendChild(o);
}}

const kept = new Set(ROWS.map(r => r.lemma));  // 기본 전체 keep (사용자가 reject만 표시)
let sortKey = "category", sortDesc = false;

function render() {{
  const q = qInput.value.trim().toLowerCase();
  const cat = catSel.value;
  const filtered = ROWS.filter(r => {{
    if (cat && r.category !== cat) return false;
    if (q && !(r.lemma + " " + r.category + " " + r.reason).toLowerCase().includes(q)) return false;
    return true;
  }});
  filtered.sort((a, b) => {{
    const av = a[sortKey] || "", bv = b[sortKey] || "";
    const r = String(av).localeCompare(String(bv));
    return sortDesc ? -r : r;
  }});
  tbody.innerHTML = filtered.map(r => {{
    const k = kept.has(r.lemma);
    return `<tr class="${{k ? 'kept' : 'rejected'}}">
      <td><input type="checkbox" data-lemma="${{r.lemma}}" ${{k ? 'checked' : ''}}></td>
      <td class="lemma">${{r.lemma}}</td>
      <td><span class="cat">${{r.category || ''}}</span></td>
      <td class="reason">${{r.reason || ''}}</td>
    </tr>`;
  }}).join("");
  for (const cb of tbody.querySelectorAll('input[type=checkbox]')) {{
    cb.addEventListener("change", () => {{
      const l = cb.dataset.lemma;
      if (cb.checked) kept.add(l); else kept.delete(l);
      cb.closest("tr").className = cb.checked ? "kept" : "rejected";
      updateCount();
    }});
  }}
  updateCount();
}}

function updateCount() {{
  countEl.textContent = `${{kept.size}}/${{ROWS.length}} kept`;
}}

qInput.addEventListener("input", render);
catSel.addEventListener("change", render);
document.querySelector("#keepAll").addEventListener("click", () => {{ ROWS.forEach(r => kept.add(r.lemma)); render(); }});
document.querySelector("#clearAll").addEventListener("click", () => {{ kept.clear(); render(); }});
document.querySelector("#export").addEventListener("click", () => {{
  const list = ROWS.filter(r => kept.has(r.lemma)).map(r => r.lemma);
  const text = list.join(" ");
  navigator.clipboard.writeText(text).then(() => {{
    out.textContent = `${{list.length}} lemmas copied to clipboard:\\n${{text}}`;
  }});
}});

for (const th of document.querySelectorAll("th[data-sort]")) {{
  th.addEventListener("click", () => {{
    const k = th.dataset.sort;
    if (sortKey === k) sortDesc = !sortDesc; else {{ sortKey = k; sortDesc = false; }}
    render();
  }});
}}
render();
</script>
</body></html>"""
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
