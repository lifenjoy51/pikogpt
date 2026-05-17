#!/usr/bin/env python3
"""ccmc-all-raw 텍스트 6개를 word-level로 토크나이즈해 어휘 빈도 + 의심 패턴 추출.

목적: 희귀어/오탈자/비영어 추리기 — 학습 corpus 품질 검토.
입력: data/ccmc-all-raw/{lemma_sentences,stories,dialogues,wiki,cause_seq,chained,counting}.txt
출력: stdout (요약) + data/ccmc-all-raw/_vocab_review/{rare.tsv, suspicious.tsv, per_file.json}
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("data/ccmc-all-raw")
FILES = [
    "lemma_sentences.txt",
    "stories.txt",
    "dialogues.txt",
    "wiki.txt",
    "cause_seq.txt",
    "chained.txt",
    "counting.txt",
]
OUT = ROOT / "_vocab_review"
OUT.mkdir(parents=True, exist_ok=True)

# special markers — 토큰화 전 제거
SPECIAL_RE = re.compile(r"<\|(?:bos|eos|sep|turn)\|>")
# literal \n / \r / \t — wiki.txt가 escape 형태 보존하므로 사전에 word boundary로 치환
LITERAL_ESCAPE_RE = re.compile(r"\\[nrt]")
# 단어 추출 — apostrophe / hyphen 보존, 그 외는 split
WORD_RE = re.compile(r"[A-Za-z]+(?:[\-'][A-Za-z]+)*")
NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
DIGIT_RE = re.compile(r"\d")

# 의심 패턴
LONG_WORD_MIN = 14         # 너무 긴 단어 (영문 평균 ≪ 14)
TRIPLE_REPEAT_RE = re.compile(r"([a-z])\1\1", re.IGNORECASE)  # aaa/eee 3+ 반복
NO_VOWEL_RE = re.compile(r"^[bcdfghjklmnpqrstvwxyz]{4,}$", re.IGNORECASE)


def normalize_for_count(tok: str) -> str:
    """소문자화. apostrophe/hyphen 보존."""
    return tok.lower()


def main() -> None:
    global_counter: Counter[str] = Counter()
    per_file: dict[str, dict] = {}
    nonascii_examples: dict[str, list[tuple[int, str]]] = defaultdict(list)
    digit_lines: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for fname in FILES:
        path = ROOT / fname
        if not path.exists():
            print(f"SKIP {fname} — not found")
            continue
        local: Counter[str] = Counter()
        total_tokens = 0
        n_lines = 0
        with path.open("r", encoding="utf-8") as f:
            for ln_no, raw in enumerate(f, start=1):
                line = SPECIAL_RE.sub(" ", raw)
                line = LITERAL_ESCAPE_RE.sub(" ", line)
                n_lines += 1
                # non-ASCII 탐지 — 처음 30개 라인만 보존
                if NON_ASCII_RE.search(line) and len(nonascii_examples[fname]) < 30:
                    nonascii_examples[fname].append((ln_no, raw.rstrip()[:200]))
                # 숫자 포함 라인 — 처음 5개만 (stories에서 흔할 듯)
                if DIGIT_RE.search(line) and len(digit_lines[fname]) < 5:
                    digit_lines[fname].append((ln_no, raw.rstrip()[:200]))
                for m in WORD_RE.finditer(line):
                    tok = normalize_for_count(m.group(0))
                    local[tok] += 1
                    total_tokens += 1
        unique = len(local)
        per_file[fname] = {
            "lines": n_lines,
            "tokens": total_tokens,
            "unique": unique,
        }
        global_counter.update(local)
        print(f"{fname:24s} lines={n_lines:>6d}  tokens={total_tokens:>9,d}  unique={unique:>6,d}")

    # 글로벌 통계
    print()
    print(f"=== GLOBAL ===  unique={len(global_counter):,d}  total_tokens={sum(global_counter.values()):,d}")

    # 빈도 분포
    freq_buckets = Counter()
    for w, c in global_counter.items():
        if c == 1:
            freq_buckets["c1 hapax"] += 1
        elif c == 2:
            freq_buckets["c2"] += 1
        elif c <= 5:
            freq_buckets["c3-5"] += 1
        elif c <= 10:
            freq_buckets["c6-10"] += 1
        elif c <= 100:
            freq_buckets["c11-100"] += 1
        else:
            freq_buckets["c>100"] += 1
    print("\nFrequency buckets:")
    for k in ["c1 hapax", "c2", "c3-5", "c6-10", "c11-100", "c>100"]:
        print(f"  {k:12s} {freq_buckets[k]:>6,d}")

    # rare (c=1) 단어 dump
    rare_path = OUT / "rare_hapax.tsv"
    with rare_path.open("w") as f:
        f.write("word\tcount\n")
        for w in sorted(w for w, c in global_counter.items() if c == 1):
            f.write(f"{w}\t1\n")
    print(f"\nrare hapax → {rare_path}  ({freq_buckets['c1 hapax']:,d} words)")

    # 의심 단어 — 너무 길거나, 모음 없거나, triple repeat
    suspicious_long: list[tuple[str, int]] = []
    suspicious_no_vowel: list[tuple[str, int]] = []
    suspicious_triple: list[tuple[str, int]] = []
    for w, c in global_counter.items():
        clean = w.replace("'", "").replace("-", "")
        if len(clean) >= LONG_WORD_MIN:
            suspicious_long.append((w, c))
        if NO_VOWEL_RE.match(clean):
            suspicious_no_vowel.append((w, c))
        if TRIPLE_REPEAT_RE.search(clean):
            suspicious_triple.append((w, c))

    susp_path = OUT / "suspicious.tsv"
    with susp_path.open("w") as f:
        f.write("category\tword\tcount\n")
        for w, c in sorted(suspicious_long, key=lambda x: (-x[1], x[0])):
            f.write(f"long\t{w}\t{c}\n")
        for w, c in sorted(suspicious_no_vowel, key=lambda x: (-x[1], x[0])):
            f.write(f"no_vowel\t{w}\t{c}\n")
        for w, c in sorted(suspicious_triple, key=lambda x: (-x[1], x[0])):
            f.write(f"triple_repeat\t{w}\t{c}\n")
    print(f"suspicious → {susp_path}  long={len(suspicious_long)} no_vowel={len(suspicious_no_vowel)} triple={len(suspicious_triple)}")

    # non-ASCII / digit summary
    if nonascii_examples:
        print("\nNon-ASCII lines (per file, up to 30):")
        for fname, examples in nonascii_examples.items():
            print(f"  {fname}: {len(examples)} lines flagged")
    digit_path = OUT / "digit_lines.txt"
    with digit_path.open("w") as f:
        for fname, examples in digit_lines.items():
            f.write(f"=== {fname} ===\n")
            for ln, raw in examples:
                f.write(f"L{ln}: {raw}\n")
    print(f"digit examples → {digit_path}")

    # non-ASCII dump
    nonascii_path = OUT / "nonascii.txt"
    with nonascii_path.open("w", encoding="utf-8") as f:
        for fname, examples in nonascii_examples.items():
            f.write(f"=== {fname} ({len(examples)}) ===\n")
            for ln, raw in examples:
                f.write(f"L{ln}: {raw}\n")
    print(f"non-ASCII → {nonascii_path}")

    per_file_path = OUT / "per_file.json"
    per_file_path.write_text(json.dumps(per_file, ensure_ascii=False, indent=2))
    print(f"per_file → {per_file_path}")


if __name__ == "__main__":
    main()
