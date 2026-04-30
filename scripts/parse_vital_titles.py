#!/usr/bin/env python3
"""
en.wikipedia.org Vital Articles Level 1-5 wikitext에서 표제어 추출.

전제: 다음 파일들이 미리 다운로드돼 있어야 함
  data/external/vital_articles/level{1,2,3}.wiki
  data/external/vital_articles/level4/{People,History,...}.wiki   (11 sub-pages)
  data/external/vital_articles/level5/{path-with-_-sep}.wiki      (34 sub-pages)
  data/external/vital_articles/level5_subpages.txt                (path 목록)

산출:
  data/external/vital_articles/vital_titles.json
    각 entry: { title: { "level": N, "category": str|null, "subcategory": str|null } }
    - L1, L2, L3: category=null, subcategory=null
    - L4: category=sub-page 이름 (People, History, ...)  subcategory=null
    - L5: category=first segment, subcategory=second segment (있으면)
  data/external/vital_articles/vital_titles.txt
    "L{level}\\t{category|-}\\t{subcategory|-}\\t{title}" 한 줄 한 표제어
"""

import collections
import json
import re
from pathlib import Path

DIR = Path("data/external/vital_articles")
L4_DIR = DIR / "level4"
L5_DIR = DIR / "level5"
L4_SUBPAGES = [
    "People", "History", "Geography", "Arts", "Everyday_life",
    "Philosophy_and_religion", "Society_and_social_sciences",
    "Biology_and_health_sciences", "Physical_sciences",
    "Technology", "Mathematics",
]

LEVELS_SINGLE = {1: "level1.wiki", 2: "level2.wiki", 3: "level3.wiki"}

LINE_RE = re.compile(r"^[*#]+\s")
LINK_RE = re.compile(r"\[\[([^\]\|#]+?)(?:\|[^\]]*)?\]\]")
NS_PREFIX = (
    "File:", "Image:", "Category:", "Wikipedia:",
    "Help:", "Portal:", "Template:", "Special:", "Talk:",
)


def parse_titles(path: Path):
    out = []
    with path.open() as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not LINE_RE.match(line):
                continue
            m = LINK_RE.search(line)
            if not m:
                continue
            title = m.group(1).strip()
            if title.startswith(NS_PREFIX):
                continue
            out.append(title)
    return out


def upsert(vital, title, level, category, subcategory):
    cur = vital.get(title)
    if cur is None or level < cur["level"]:
        vital[title] = {"level": level, "category": category, "subcategory": subcategory}
        return
    if cur["level"] == level:
        # 같은 level에 다른 카테고리에서 또 등장 — 첫 카테고리만 유지
        if cur["category"] is None and category is not None:
            cur["category"] = category
            cur["subcategory"] = subcategory


def main():
    vital = {}
    raw_counts = {}

    # L1, L2, L3
    for level, fname in LEVELS_SINGLE.items():
        titles = parse_titles(DIR / fname)
        raw_counts[f"L{level}"] = len(titles)
        for t in titles:
            upsert(vital, t, level, None, None)

    # L4
    for sub in L4_SUBPAGES:
        path = L4_DIR / f"{sub}.wiki"
        if not path.exists():
            print(f"  WARN: {path} missing — skip")
            continue
        titles = parse_titles(path)
        raw_counts[f"L4/{sub}"] = len(titles)
        category = sub.replace("_", " ")
        for t in titles:
            upsert(vital, t, 4, category, None)

    # L5
    l5_paths_file = DIR / "level5_subpages.txt"
    if l5_paths_file.exists():
        l5_paths = [ln.strip() for ln in l5_paths_file.read_text().splitlines() if ln.strip()]
    else:
        l5_paths = []

    for path in l5_paths:
        parts = path.split("/", 1)
        category = parts[0]
        subcategory = parts[1] if len(parts) > 1 else None
        fname = path.replace("/", "_").replace(" ", "_") + ".wiki"
        full = L5_DIR / fname
        if not full.exists():
            print(f"  WARN: {full} missing — skip")
            continue
        titles = parse_titles(full)
        raw_counts[f"L5/{path}"] = len(titles)
        for t in titles:
            upsert(vital, t, 5, category, subcategory)

    out_json = DIR / "vital_titles.json"
    out_txt = DIR / "vital_titles.txt"

    with out_json.open("w") as f:
        json.dump(
            {t: vital[t] for t in sorted(vital.keys())},
            f, ensure_ascii=False, indent=2,
        )

    with out_txt.open("w") as f:
        for t in sorted(vital, key=lambda x: (
            vital[x]["level"], vital[x]["category"] or "", vital[x]["subcategory"] or "", x
        )):
            cat = vital[t]["category"] or "-"
            sub = vital[t]["subcategory"] or "-"
            f.write(f"L{vital[t]['level']}\t{cat}\t{sub}\t{t}\n")

    c = collections.Counter(v["level"] for v in vital.values())
    print("=== raw extraction (per source) ===")
    total_raw = 0
    for k, n in raw_counts.items():
        print(f"  {k}: {n:,}")
        total_raw += n
    print(f"  raw total: {total_raw:,}")
    print()
    print("=== unique by level ===")
    for lv in sorted(c):
        print(f"  L{lv}: {c[lv]:,}")
    print(f"  total unique: {len(vital):,}")
    print()
    print(f"wrote {out_json}")
    print(f"wrote {out_txt}")


if __name__ == "__main__":
    main()
