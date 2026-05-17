#!/usr/bin/env python3
"""data/lemma-anchor-filter/judgments.jsonl → results.html (search/filter/sort UI)."""
import html
import json
from collections import Counter
from pathlib import Path

import sys
USE_MERGED = "--merged" in sys.argv
JUDG = Path("data/lemma-anchor-filter/judgments_merged.jsonl") if USE_MERGED \
       else Path("data/lemma-anchor-filter/judgments.jsonl")
OUT = Path("data/lemma-anchor-filter/results_merged.html") if USE_MERGED \
      else Path("data/lemma-anchor-filter/results.html")


def main() -> None:
    rows = [json.loads(ln) for ln in JUDG.read_text().splitlines() if ln.strip()]
    total = len(rows)
    keep_n = sum(1 for r in rows if r.get("keep"))
    drop_n = total - keep_n
    cat_counts = Counter(r.get("category") or "—" for r in rows if r.get("keep"))
    max_cat = max(cat_counts.values()) if cat_counts else 1

    cat_bars = "".join(
        f'<div class="bar"><span class="bar-label">{html.escape(c)}</span>'
        f'<span class="bar-fill" style="width:{n / max_cat * 100:.1f}%">{n}</span></div>'
        for c, n in cat_counts.most_common()
    )

    rows_json = json.dumps(
        [
            {
                "lemma": r.get("lemma", ""),
                "freq": r.get("freq", 0),
                "keep": bool(r.get("keep")),
                "category": r.get("category") or "",
                "reason": r.get("reason") or "",
            }
            for r in rows
        ],
        ensure_ascii=False,
    )

    html_out = f"""<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8">
<title>Lemma Anchor Filter — DeepSeek v4 Flash 5세 적합성 판정</title>
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
  .lemma {{ font-weight: 500; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .freq {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #6b7280; text-align: right; }}
  .reason {{ color: #374151; max-width: 240px; }}
  .footer {{ margin-top: 12px; color: #9ca3af; font-size: 11px; text-align: center; }}
</style>
</head>
<body>
<h1>Lemma Anchor Filter — ccmc-all 누락 lemma 1330개 → DeepSeek v4 Flash 5세 적합성 판정</h1>

<div class="summary">
  <div class="stat"><b>{total:,}</b> total</div>
  <div class="stat"><b style="color:#166534">{keep_n:,}</b> keep ({keep_n / total * 100:.1f}%)</div>
  <div class="stat"><b style="color:#991b1b">{drop_n:,}</b> drop ({drop_n / total * 100:.1f}%)</div>
  <div class="stat">{len(cat_counts)} categories</div>
  <div class="stat">batch: 133 × 10 / shuffle seed: 51 / cost: $0.015</div>
</div>

<div class="cats">{cat_bars}</div>

<div class="filters">
  <input id="q" placeholder="lemma / reason / category 검색...">
  <button data-f="all" class="active">전체</button>
  <button data-f="keep">keep</button>
  <button data-f="drop">drop</button>
  <select id="cat"><option value="">all categories</option></select>
  <select id="freq">
    <option value="0">all freq</option>
    <option value="500">freq ≥ 500</option>
    <option value="100">freq ≥ 100</option>
    <option value="50">freq ≥ 50</option>
  </select>
  <span id="count"></span>
</div>

<table id="tbl">
  <thead><tr>
    <th data-sort="freq">freq ▼</th>
    <th data-sort="lemma">lemma</th>
    <th data-sort="keep">decision</th>
    <th data-sort="category">category</th>
    <th data-sort="reason">reason</th>
  </tr></thead>
  <tbody></tbody>
</table>

<div class="footer">data/lemma-anchor-filter/judgments.jsonl — 생성: scripts/render_lemma_anchor_html.py</div>

<script>
const ROWS = {rows_json};
const tbody = document.querySelector("#tbl tbody");
const countEl = document.querySelector("#count");
const qInput = document.querySelector("#q");
const catSel = document.querySelector("#cat");
const freqSel = document.querySelector("#freq");

// populate category dropdown
const cats = [...new Set(ROWS.filter(r => r.keep && r.category).map(r => r.category))].sort();
for (const c of cats) {{
  const o = document.createElement("option"); o.value = c; o.textContent = c; catSel.appendChild(o);
}}

let curFilter = "all"; let sortKey = "freq"; let sortDesc = true;

function rowMatches(r) {{
  if (curFilter === "keep" && !r.keep) return false;
  if (curFilter === "drop" && r.keep) return false;
  const cat = catSel.value;
  if (cat && r.category !== cat) return false;
  const fmin = parseInt(freqSel.value, 10);
  if (fmin && r.freq < fmin) return false;
  const q = qInput.value.trim().toLowerCase();
  if (q) {{
    const hay = (r.lemma + " " + r.category + " " + r.reason).toLowerCase();
    if (!hay.includes(q)) return false;
  }}
  return true;
}}

function render() {{
  const filtered = ROWS.filter(rowMatches);
  filtered.sort((a, b) => {{
    let av = a[sortKey], bv = b[sortKey];
    if (sortKey === "keep") {{ av = a.keep ? 1 : 0; bv = b.keep ? 1 : 0; }}
    if (typeof av === "string") {{
      const r = av.localeCompare(bv);
      return sortDesc ? -r : r;
    }}
    return sortDesc ? bv - av : av - bv;
  }});
  tbody.innerHTML = filtered.slice(0, 2000).map(r => `<tr>
    <td class="freq">${{r.freq.toLocaleString()}}</td>
    <td class="lemma">${{r.lemma}}</td>
    <td><span class="badge ${{r.keep ? "keep" : "drop"}}">${{r.keep ? "KEEP" : "DROP"}}</span></td>
    <td>${{r.category ? `<span class="cat">${{r.category}}</span>` : ""}}</td>
    <td class="reason">${{r.reason}}</td>
  </tr>`).join("");
  const limited = filtered.length > 2000 ? " (showing 2000)" : "";
  countEl.textContent = `${{filtered.length.toLocaleString()}} rows${{limited}}`;
}}

qInput.addEventListener("input", render);
catSel.addEventListener("change", render);
freqSel.addEventListener("change", render);
for (const btn of document.querySelectorAll(".filters button")) {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".filters button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    curFilter = btn.dataset.f;
    render();
  }});
}}
for (const th of document.querySelectorAll("th[data-sort]")) {{
  th.addEventListener("click", () => {{
    const k = th.dataset.sort;
    if (sortKey === k) sortDesc = !sortDesc; else {{ sortKey = k; sortDesc = (k === "freq" || k === "keep"); }}
    document.querySelectorAll("th").forEach(t => t.textContent = t.textContent.replace(/ ▼| ▲/, ""));
    th.textContent += sortDesc ? " ▼" : " ▲";
    render();
  }});
}}
render();
</script>
</body></html>"""

    OUT.write_text(html_out)
    print(f"wrote {OUT} ({len(html_out):,} bytes)")


if __name__ == "__main__":
    main()
