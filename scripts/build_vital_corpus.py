#!/usr/bin/env python3
"""
vital_titles_resolved.json 의 매칭된 표제어를 사용해
simplewiki_clean.jsonl + simplewiki_vital_extras.jsonl 에서 본문을 모아
level/category 메타데이터가 부착된 단일 코퍼스 jsonl 을 만든다.

산출:
  data/simplewiki/simplewiki_vital_corpus.jsonl
    각 라인 = {"title": str, "text": str, "level": int, "category": str|null, "source": "clean"|"extras"}

향후 학습 코퍼스 빌드 시 이 파일을 입력으로 사용. level/category로
가중 샘플링·학습 데이터 separator(예: <|level1|> 같은 special token) 추가 가능.
"""

import collections
import json
from pathlib import Path

VITAL_DIR = Path("data/external/vital_articles")
CLEAN = Path("data/simplewiki/simplewiki_clean.jsonl")
EXTRAS = Path("data/simplewiki/simplewiki_vital_extras.jsonl")
OUT = Path("data/simplewiki/simplewiki_vital_corpus.jsonl")


def main():
    resolved = json.load(open(VITAL_DIR / "vital_titles_resolved.json"))
    # subcategory 정보는 vital_titles.json 에 있으므로 함께 로드
    vital_meta = json.load(open(VITAL_DIR / "vital_titles.json"))

    # simplewiki_title (lowercase) -> {level, category, subcategory, source}
    # 같은 simplewiki page를 가리키는 vital_title이 여러 level에 있을 경우
    # 가장 작은 level (위계상 더 중요한 것)을 유지
    target = {}
    for orig, info in resolved.items():
        sw_t = info.get("simplewiki_title")
        if not sw_t:
            continue
        key = sw_t.lower()
        cur = target.get(key)
        if cur is None or info["level"] < cur["level"]:
            meta_full = vital_meta.get(orig, {})
            target[key] = {
                "vital_title": orig,
                "level": info["level"],
                "category": info.get("category") or meta_full.get("category"),
                "subcategory": meta_full.get("subcategory"),
                "source": info.get("source", "clean"),
            }

    n_clean = 0
    n_extras = 0
    out_records = []

    # clean에서 추출
    with open(CLEAN) as f:
        for line in f:
            d = json.loads(line)
            t = d.get("title", "").strip()
            meta = target.get(t.lower())
            if meta is None:
                continue
            out_records.append({
                "title": t,
                "text": d.get("text", ""),
                "level": meta["level"],
                "category": meta["category"],
                "subcategory": meta.get("subcategory"),
                "source": "clean",
            })
            n_clean += 1

    # extras에서 추출
    with open(EXTRAS) as f:
        for line in f:
            d = json.loads(line)
            t = d.get("title", "").strip()
            meta = target.get(t.lower())
            if meta is None:
                continue
            out_records.append({
                "title": t,
                "text": d.get("text", ""),
                "level": meta["level"],
                "category": meta["category"],
                "subcategory": meta.get("subcategory"),
                "source": "extras",
            })
            n_extras += 1

    # level 작은 순 → category → title 순으로 정렬
    out_records.sort(key=lambda r: (r["level"], r["category"] or "", r["title"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 통계
    print(f"wrote {OUT}  ({len(out_records):,} docs)")
    print(f"  from clean: {n_clean:,}")
    print(f"  from extras: {n_extras:,}")
    print()

    by_level = collections.defaultdict(lambda: {"docs": 0, "chars": 0, "words": 0})
    by_cat = collections.defaultdict(lambda: {"docs": 0, "chars": 0, "words": 0})
    total = {"docs": 0, "chars": 0, "words": 0}

    for rec in out_records:
        c = len(rec["text"])
        w = len(rec["text"].split())
        by_level[rec["level"]]["docs"] += 1
        by_level[rec["level"]]["chars"] += c
        by_level[rec["level"]]["words"] += w
        if rec["category"]:
            by_cat[rec["category"]]["docs"] += 1
            by_cat[rec["category"]]["chars"] += c
            by_cat[rec["category"]]["words"] += w
        total["docs"] += 1
        total["chars"] += c
        total["words"] += w

    print("=== per level ===")
    print(f"  {'level':<6s}{'docs':>8s}{'chars':>14s}{'words':>14s}{'avg w/doc':>12s}")
    for lv in sorted(by_level):
        s = by_level[lv]
        avg = s["words"] / s["docs"] if s["docs"] else 0
        print(f"  L{lv:<5d}{s['docs']:>8,d}{s['chars']:>14,d}{s['words']:>14,d}{avg:>12,.0f}")
    print(f"  {'TOTAL':<6s}{total['docs']:>8,d}{total['chars']:>14,d}{total['words']:>14,d}")

    print()
    print(f"chars: {total['chars']/1e6:.1f} MB")
    print(f"words: {total['words']/1e6:.2f} M")
    print(f"est BPE tokens (vocab 2k, ratio 1.4-1.6/word): "
          f"{total['words']*1.4/1e6:.1f}M – {total['words']*1.6/1e6:.1f}M")
    print()
    print("=== L4 category breakdown ===")
    print(f"  {'category':<32s}{'docs':>7s}{'words':>12s}{'avg':>8s}")
    for cat in sorted(by_cat):
        s = by_cat[cat]
        avg = s["words"] / s["docs"] if s["docs"] else 0
        print(f"  {cat:<32s}{s['docs']:>7,d}{s['words']:>12,d}{avg:>8,.0f}")


if __name__ == "__main__":
    main()
