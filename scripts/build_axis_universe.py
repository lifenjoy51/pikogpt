#!/usr/bin/env python3
"""v6 axis 합성용 anchor universe 빌드.

입력:
  data/ccmc-all-raw/lemma_anchors.tsv  (3,288 lemma, lemma\\tcategory\\tsource)
  data/ccmc-all-raw/wiki.txt           (3,940 records, title\\n\\nbody literal \\n)
  data/wiki-topic-filter/judgments.jsonl  (keep=true 3,940건의 id ↔ title mapping)
  data/external-all-raw/wiki.txt       (8,942 lines, line number = id)

출력:
  data/v6-axes/universe.jsonl
    각 record: {anchor, original_title, source, category, article_id, article}
    - anchor: lowercase + trim (dedup 키)
    - original_title: 출력 wiki_v2.txt 표제어 표기용 (sentence case 등)
    - source: "both" | "lemma_only" | "wiki_only"
    - category: lemma_anchors 우선, 없으면 judgments
    - article_id: int or null (judgments.id)
    - article: str or null (BODY_CHAR_CAP=8000 trim, literal \\n 그대로)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LEMMA_TSV = Path("data/ccmc-all-raw/lemma_anchors.tsv")
CCMC_WIKI = Path("data/ccmc-all-raw/wiki.txt")
JUDGMENTS = Path("data/wiki-topic-filter/judgments.jsonl")
EXTERNAL_WIKI = Path("data/external-all-raw/wiki.txt")
OUT = Path("data/v6-axes/universe.jsonl")

BODY_CHAR_CAP = 8000


def load_lemma_anchors() -> dict[str, dict]:
    """lowercase lemma → {category, source}"""
    out: dict[str, dict] = {}
    with LEMMA_TSV.open() as f:
        header = next(f)
        assert header.rstrip("\n").split("\t") == ["lemma", "category", "source"], header
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            lemma = parts[0].strip().lower()
            if not lemma:
                continue
            out[lemma] = {"category": parts[1].strip(), "source_tag": parts[2].strip()}
    return out


def load_wiki_titles() -> list[tuple[str, str]]:
    """ccmc-all-raw/wiki.txt에서 (lowercase_title, original_title) 추출.

    각 줄 형식: "Title\\n\\nbody..." (literal \\n)
    """
    out: list[tuple[str, str]] = []
    with CCMC_WIKI.open() as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            m = re.split(r"\\n\\n", ln, maxsplit=1)
            if not m:
                continue
            title = m[0].strip()
            out.append((title.lower(), title))
    return out


def load_judgments_keep() -> dict[str, dict]:
    """lowercase title → {id, category}  (keep=true만)"""
    out: dict[str, dict] = {}
    for ln in JUDGMENTS.read_text().splitlines():
        if not ln.strip():
            continue
        j = json.loads(ln)
        if not j.get("keep"):
            continue
        t = (j.get("title") or "").strip().lower()
        if not t:
            continue
        out[t] = {"id": j["id"], "category": j.get("category") or ""}
    return out


def load_external_bodies() -> dict[int, str]:
    """line N (1-indexed) → body (BODY_CHAR_CAP trim)"""
    out: dict[int, str] = {}
    with EXTERNAL_WIKI.open() as f:
        for i, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            if len(line) > BODY_CHAR_CAP:
                line = line[:BODY_CHAR_CAP] + "..."
            out[i] = line
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    lemmas = load_lemma_anchors()
    wiki_titles = load_wiki_titles()
    judgments_keep = load_judgments_keep()
    external = load_external_bodies()

    print(f"lemma_anchors:  {len(lemmas):>5} unique lemma")
    print(f"wiki titles:    {len(wiki_titles):>5} records ({len(set(t for t, _ in wiki_titles))} unique lower)")
    print(f"judgments keep: {len(judgments_keep):>5}")
    print(f"external wiki:  {len(external):>5} lines")

    # title → original 매핑 (마지막 등장 우선; 거의 1:1)
    title_to_original: dict[str, str] = {}
    for t_lower, t_orig in wiki_titles:
        title_to_original[t_lower] = t_orig

    # 합집합
    all_anchors = set(lemmas.keys()) | set(title_to_original.keys())
    print(f"union (deduped lowercase): {len(all_anchors):>5}")

    # 출력 빌드
    records: list[dict] = []
    counts = {"both": 0, "lemma_only": 0, "wiki_only": 0}
    article_attached = 0

    for anchor in sorted(all_anchors):
        in_lemma = anchor in lemmas
        in_wiki = anchor in title_to_original

        if in_lemma and in_wiki:
            source = "both"
        elif in_lemma:
            source = "lemma_only"
        else:
            source = "wiki_only"
        counts[source] += 1

        # category 우선순위: lemma_anchors > judgments
        category = ""
        if in_lemma and lemmas[anchor]["category"] and lemmas[anchor]["category"] != "—":
            category = lemmas[anchor]["category"]
        if not category and in_wiki and anchor in judgments_keep:
            category = judgments_keep[anchor]["category"]

        # original_title
        if in_wiki:
            original_title = title_to_original[anchor]
        else:
            original_title = " ".join(w.capitalize() for w in anchor.split())

        # ARTICLE join
        article_id = None
        article: str | None = None
        if in_wiki and anchor in judgments_keep:
            article_id = judgments_keep[anchor]["id"]
            article = external.get(article_id)
            if article:
                article_attached += 1

        records.append({
            "anchor": anchor,
            "original_title": original_title,
            "source": source,
            "category": category,
            "article_id": article_id,
            "article": article,
        })

    print(f"\nsource distribution:")
    for k in ("both", "lemma_only", "wiki_only"):
        print(f"  {k:<12} {counts[k]:>5}")
    print(f"ARTICLE attached: {article_attached:>5} ({article_attached/len(records)*100:.1f}%)")
    expected_article = counts["both"] + counts["wiki_only"]
    if article_attached != expected_article:
        print(f"  WARN: expected {expected_article} (both+wiki_only); got {article_attached}")

    # 출력 JSONL
    with OUT.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT} ({OUT.stat().st_size:,} bytes, {len(records):,} records)")


if __name__ == "__main__":
    main()
