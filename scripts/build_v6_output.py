#!/usr/bin/env python3
"""v6 axis raw_{axis}.jsonl → text 파일 빌드.

axis A (wiki):
  포맷: f"{original_title}\\n\\n{escaped_text}\\n" — literal \\n\\n 구분, 실제 newline은 줄 끝에만.
  text 안의 실제 newline은 literal "\\n"으로 escape (기존 wiki.txt 컨벤션).
  출력: data/v6-axes/output/wiki_v2.txt

axis B/C/D (cause_seq / chained / counting):
  포맷: 1 line = 1 paragraph (normalize: 실제 newline → 공백, multiple space → single).
  출력: data/v6-axes/output/{cause_seq|chained|counting}.txt

stats.json: axis × category × skip_rate 매트릭스 + reason 빈도 (단순 통계).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

OUT_DIR = Path("data/v6-axes/output")

AXIS_OUT_FNAME = {
    "a": "wiki_v2.txt",
    "b": "cause_seq.txt",
    "c": "chained.txt",
    "d": "counting.txt",
}


def normalize_inline(text: str) -> str:
    """B/C/D용 — 실제 newline / literal \\n / 다중 공백을 단일 공백으로."""
    text = text.replace("\\n", " ").replace("\r\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_wiki_v2(rows: list[dict]) -> tuple[str, int, int]:
    """A axis — title\\n\\nbody literal \\n 포맷."""
    lines: list[str] = []
    ok, total = 0, 0
    for r in rows:
        total += 1
        if not (r.get("ok") and r.get("text")):
            continue
        text = r["text"].replace("\r\n", "\n")
        text_literal = text.replace("\n", "\\n")
        title = r["original_title"]
        lines.append(f"{title}\\n\\n{text_literal}")
        ok += 1
    return "\n".join(lines) + "\n", ok, total


def build_paragraph(rows: list[dict]) -> tuple[str, int, int]:
    """B/C/D axis — 1 line = 1 paragraph."""
    lines: list[str] = []
    ok, total = 0, 0
    for r in rows:
        total += 1
        if not (r.get("ok") and r.get("text")):
            continue
        text = normalize_inline(r["text"])
        if text:
            lines.append(text)
            ok += 1
    return "\n".join(lines) + "\n", ok, total


def write_stats(axis: str, rows: list[dict]) -> dict:
    """axis별 category × skip rate, reason 분포."""
    cat_total = Counter()
    cat_skip = Counter()
    reasons = Counter()
    for r in rows:
        cat = r["category"] or "—"
        cat_total[cat] += 1
        if r.get("ok") and r.get("text") is None:
            cat_skip[cat] += 1
            reasons[(r.get("reason") or "")] += 1
        elif not r.get("ok"):
            reasons[f"FAIL: {(r.get('error') or '')[:50]}"] += 1
    cat_rates = {c: {"n": cat_total[c], "skip": cat_skip[c],
                     "skip_rate": cat_skip[c] / cat_total[c] if cat_total[c] else 0.0}
                 for c in cat_total}
    return {
        "axis": axis,
        "total": len(rows),
        "ok": sum(1 for r in rows if r.get("ok") and r.get("text")),
        "skip": sum(1 for r in rows if r.get("ok") and r.get("text") is None),
        "fail": sum(1 for r in rows if not r.get("ok")),
        "by_category": cat_rates,
        "reasons_top20": reasons.most_common(20),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=list(AXIS_OUT_FNAME))
    args = ap.parse_args()
    axis = args.axis

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = Path(f"data/v6-axes/raw_{axis}.jsonl")
    if not raw_path.exists():
        raise SystemExit(f"ERROR: {raw_path} not found")
    rows = [json.loads(ln) for ln in raw_path.read_text().splitlines() if ln.strip()]

    # dedup by anchor (마지막 등장 우선; 재시작/재호출 대비)
    by_anchor: dict[str, dict] = {}
    for r in rows:
        by_anchor[r["anchor"]] = r
    rows_dedup = list(by_anchor.values())

    # 안정 정렬: anchor 알파벳
    rows_dedup.sort(key=lambda r: r["anchor"])

    out_path = OUT_DIR / AXIS_OUT_FNAME[axis]
    if axis == "a":
        body, ok, total = build_wiki_v2(rows_dedup)
    else:
        body, ok, total = build_paragraph(rows_dedup)
    out_path.write_text(body)
    print(f"wrote {out_path}  ({out_path.stat().st_size:,} bytes, {ok} ok / {total} total)")

    # stats
    stats = write_stats(axis, rows_dedup)
    stats_path = Path("data/v6-axes/stats.json")
    if stats_path.exists():
        existing = json.loads(stats_path.read_text())
        if not isinstance(existing, dict):
            existing = {}
    else:
        existing = {}
    existing[axis] = stats
    stats_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"stats: ok={stats['ok']}  skip={stats['skip']}  fail={stats['fail']}  → {stats_path}")


if __name__ == "__main__":
    main()
