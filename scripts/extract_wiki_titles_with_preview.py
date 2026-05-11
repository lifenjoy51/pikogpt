#!/usr/bin/env python3
"""SimpleWiki wiki.txt → titles.jsonl with previews (seed=42 shuffled).

각 line = 1 doc. 내부는 literal "\\n" 문자열로 paragraph 구분.
- parts[0]: title
- parts[1]: '' (\\n\\n 사이)
- parts[2..]: paragraphs

출력:
  data/wiki-topic-filter/titles.jsonl  (1 line=1 entry, 결정적 shuffle된 순서)
포맷: {"id": <int>, "title": "<str>", "preview": "<≤200 char str>"}

id는 원본 line 번호 그대로(1..N) 유지하되, 파일에 쓰는 순서는 shuffle.
driver는 이 파일 순서대로 batch를 만들어 호출하므로 batch별 카테고리 편향이 줄어듦.
"""
import json
import random
from pathlib import Path

SRC = Path("data/external-all-raw/wiki.txt")
OUT = Path("data/wiki-topic-filter/titles.jsonl")
PREVIEW_MAX_CHARS = 200
SHUFFLE_SEED = 42


def build_preview(parts: list[str]) -> str:
    """parts[2:]에서 의미 있는 paragraph만 모아 PREVIEW_MAX_CHARS까지 합친다."""
    chunks: list[str] = []
    used = 0
    for p in parts[2:]:
        p = p.strip()
        if not p:
            continue
        remain = PREVIEW_MAX_CHARS - used
        if remain <= 0:
            break
        if len(p) <= remain:
            chunks.append(p)
            used += len(p) + 1
        else:
            chunks.append(p[:remain].rstrip() + "...")
            break
    return " ".join(chunks)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    with SRC.open() as fin:
        for i, raw in enumerate(fin, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\\n")
            title = parts[0].strip()
            preview = build_preview(parts) if len(parts) > 2 else ""
            entries.append({"id": i, "title": title, "preview": preview})

    random.Random(SHUFFLE_SEED).shuffle(entries)

    with OUT.open("w") as fout:
        for e in entries:
            fout.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}  entries={len(entries)}  shuffle_seed={SHUFFLE_SEED}")


if __name__ == "__main__":
    main()
