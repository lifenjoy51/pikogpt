#!/usr/bin/env python3
"""현재 corpus 기준으로 rare_hapax.tsv 처음부터 재생성.

1) corpus 7개 파일에서 단어 빈도 재계산 (count=1 = hapax)
2) 각 hapax에 sentence 매핑 — split boundary는 [.!?] + <|turn|>등 special marker + literal \\n/\\r/\\t

입력: data/ccmc-all-raw/{lemma_sentences,stories,dialogues,wiki,cause_seq,chained,counting}.txt
출력: data/ccmc-all-raw/_vocab_review/rare_hapax.tsv (word\tcount\tsentence) — 덮어쓰기
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ROOT = Path("data/ccmc-all-raw")
OUT_TSV = ROOT / "_vocab_review" / "rare_hapax.tsv"
FILES = [
    "lemma_sentences.txt", "stories.txt", "dialogues.txt", "wiki.txt",
    "cause_seq.txt", "chained.txt", "counting.txt",
]

# 단어 카운트용 — special marker + literal \n 모두 공백으로 치환 후 word regex
SPECIAL_RE = re.compile(r"<\|(?:bos|eos|sep|turn)\|>")
LITERAL_ESCAPE_RE = re.compile(r"\\[nrt]")
WORD_RE = re.compile(r"[A-Za-z]+(?:[\-'][A-Za-z]+)*")
# sentence split용 — boundary: special marker OR literal escape
BOUNDARY_RE = re.compile(r"<\|(?:bos|eos|sep|turn)\|>|\\[nrt]")
# 종결자 직후 공백 — 문장 split
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WS_RE = re.compile(r"\s+")


def split_sentences(line: str) -> list[str]:
    out = []
    for chunk in BOUNDARY_RE.split(line):
        for s in SENT_SPLIT_RE.split(chunk):
            s = WS_RE.sub(" ", s).strip()
            if s:
                out.append(s)
    return out


def main() -> None:
    counter: Counter[str] = Counter()
    print("=== 1) 단어 빈도 ===")
    for fname in FILES:
        path = ROOT / fname
        if not path.exists():
            print(f"  SKIP {fname}")
            continue
        local = 0
        with path.open() as f:
            for raw in f:
                line = SPECIAL_RE.sub(" ", raw)
                line = LITERAL_ESCAPE_RE.sub(" ", line)
                for m in WORD_RE.finditer(line):
                    counter[m.group(0).lower()] += 1
                    local += 1
        print(f"  {fname:24s} tokens={local:>9,d}")
    total = sum(counter.values())
    unique = len(counter)
    print(f"  total tokens: {total:,d}  unique: {unique:,d}")

    hapax = sorted(w for w, c in counter.items() if c == 1)
    hapax_set = set(hapax)
    print(f"  hapax (count=1): {len(hapax):,d}")

    print("\n=== 2) sentence 매핑 ===")
    sentence_by_word: dict[str, str] = {}
    for fname in FILES:
        path = ROOT / fname
        if not path.exists():
            continue
        with path.open() as f:
            for raw in f:
                for sent in split_sentences(raw):
                    sent_words = set()
                    for m in WORD_RE.finditer(sent):
                        sent_words.add(m.group(0).lower())
                    for w in (sent_words & hapax_set):
                        if w not in sentence_by_word:
                            sentence_by_word[w] = sent

    matched = sum(1 for w in hapax if w in sentence_by_word)
    print(f"  matched: {matched:,d}  unmatched: {len(hapax) - matched:,d}")

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    out_lines = ["word\tcount\tsentence"]
    for w in hapax:
        sent = sentence_by_word.get(w, "")
        out_lines.append(f"{w}\t1\t{sent}")
    OUT_TSV.write_text("\n".join(out_lines) + "\n")
    print(f"\nwrote {OUT_TSV}  ({len(hapax):,d} hapax rows)")


if __name__ == "__main__":
    main()
