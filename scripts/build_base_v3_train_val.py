#!/usr/bin/env python3
"""base 코퍼스 빌드 — vital corpus jsonl → train.txt / val.txt.

`docs/base-v3-recipe.md §11.1` 절차:
  1. data/simplewiki/simplewiki_vital_corpus.jsonl 입력
  2. level <= 4 필터 (L5 폐기)
  3. doc 단위 random.shuffle (seed 고정 재현성)
  4. 90:10 train/val split
  5. clean_preserve_paragraphs: smart-quote → ASCII, non-ASCII 제거, ws 정규화
     (paragraph 구분 \\n 보존)
  6. it train.txt 합본 char-freq 임계로 저빈도 char 제거 (BPE vocab 일관성)
  7. doc wrap: <|bos|>\\n{text}\\n<|eos|>, doc 사이는 빈 줄
  8. data/two-stage-v3/base/{train,val}.txt 작성

ASCII 정제 로직은 scripts/text_cleaning.py 라이브러리 사용.

Usage:
  python3 scripts/build_base_v3_train_val.py
  python3 scripts/build_base_v3_train_val.py --max-level 4 --val-frac 0.10 --seed 42
"""

import argparse
import json
import os
import random
import re
import sys

# scripts/ 를 import path에 추가 (text_cleaning import용)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_cleaning import (
    clean_preserve_paragraphs, build_allowed_chars, filter_low_freq_chars,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "data/simplewiki/simplewiki_vital_corpus.jsonl")
IT_TRAIN = os.path.join(ROOT, "data/two-stage-v2/raw/it-v2/train.txt")
OUT_DIR = os.path.join(ROOT, "data/two-stage-v3/base")

WS_RE = re.compile(r"[ \t]+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--it-train", default=IT_TRAIN,
                    help="IT-V2 train.txt — char-freq 임계 결정용 합본 (없으면 skip)")
    ap.add_argument("--max-level", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1) jsonl 로드 + level 필터
    docs = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("level", 99) > args.max_level:
                continue
            docs.append(d)
    print(f"loaded: {len(docs):,} docs (level <= {args.max_level})", file=sys.stderr)

    # 2) shuffle + split
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    n_val = max(1, int(len(docs) * args.val_frac))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]
    print(f"split: train={len(train_docs):,}  val={len(val_docs):,}", file=sys.stderr)

    # 3) paragraph 구조 보존하며 정제
    def render(doc_list):
        out = []
        for d in doc_list:
            text = clean_preserve_paragraphs(d["text"])
            if text:
                out.append(text)
        return out

    train_texts = render(train_docs)
    val_texts = render(val_docs)

    # 4) it-v2 합본으로 char-freq 임계 결정
    it_text = ""
    if os.path.isfile(args.it_train):
        with open(args.it_train, encoding="utf-8") as f:
            it_text = f.read()
        print(f"it-v2 train.txt: {len(it_text):,} chars", file=sys.stderr)
    else:
        print(f"WARN: {args.it_train} not found — base-v3 단독으로 임계 결정",
              file=sys.stderr)

    combined = "\n".join(train_texts) + "\n" + it_text
    allowed, removed, threshold = build_allowed_chars(combined)
    print(f"char-freq pivot=`_` 빈도 → 임계 {threshold} → 보존 chars {len(allowed)}",
          file=sys.stderr)
    print(f"제거 chars {len(removed)}개 (top 20):", file=sys.stderr)
    for c, freq in sorted(removed.items(), key=lambda x: -x[1])[:20]:
        label = repr(c) if not c.isalnum() and c != ' ' else c
        print(f"  {label!r:>8s}  {freq:>10,d}", file=sys.stderr)

    # 5) filter + wrap (paragraph 구분 \n\n 보존)
    def normalize_after_filter_preserving_paragraphs(text: str) -> str:
        """필터 후 multi-space squeeze + 라인 trim, 단 빈 줄은 paragraph 구분으로 살림."""
        text = WS_RE.sub(' ', text)
        lines = [ln.strip() for ln in text.splitlines()]
        text = '\n'.join(lines)
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        return text.strip()

    def wrap(texts):
        out = []
        for t in texts:
            t = filter_low_freq_chars(t, allowed)
            t = normalize_after_filter_preserving_paragraphs(t)
            if t:
                out.append(f"<|bos|>\n{t}\n<|eos|>")
        return out

    train_blocks = wrap(train_texts)
    val_blocks = wrap(val_texts)

    # 6) write — doc 사이는 빈 줄로 구분, doc 내부는 paragraph 그대로
    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.txt")
    val_path = os.path.join(args.out_dir, "val.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(train_blocks) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(val_blocks) + "\n")

    train_size = os.path.getsize(train_path)
    val_size = os.path.getsize(val_path)
    train_words = sum(len(t.split()) for t in train_blocks)
    val_words = sum(len(t.split()) for t in val_blocks)
    train_lines = train_blocks  # alias
    val_lines = val_blocks

    print(file=sys.stderr)
    print(f"=== 결과 ===", file=sys.stderr)
    print(f"  {train_path}: {len(train_lines):,} docs / "
          f"{train_words:,} words / {train_size:,} bytes",
          file=sys.stderr)
    print(f"  {val_path}: {len(val_lines):,} docs / "
          f"{val_words:,} words / {val_size:,} bytes",
          file=sys.stderr)


if __name__ == "__main__":
    main()
