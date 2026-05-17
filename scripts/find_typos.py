#!/usr/bin/env python3
"""hapax 단어 중 진짜 typo 후보 추출.

전략: hapax (count=1)인데 빈도 ≥5인 다른 단어와 edit distance=1인 경우 → typo 강력 의심.
이런 단어는 LLM이 한 번 실수로 친 흔적일 가능성 높음 (corpus 다른 위치에 정상 spelling 다수).

길이 ≥4 단어만 (3자 미만은 alphabet bigram 확률로 노이즈).
"""
from __future__ import annotations

import re
from collections import Counter
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

SPECIAL_RE = re.compile(r"<\|(?:bos|eos|sep|turn)\|>")
LITERAL_ESCAPE_RE = re.compile(r"\\[nrt]")
WORD_RE = re.compile(r"[A-Za-z]+(?:[\-'][A-Za-z]+)*")


def edit_distance_one(a: str, b: str) -> bool:
    """a, b가 정확히 edit distance 1인지 (substitute / insert / delete 중 하나)."""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = sum(1 for x, y in zip(a, b) if x != y)
        return diffs == 1
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    found = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            if found:
                return False
            found = True
            j += 1
    return True


def main() -> None:
    counter: Counter[str] = Counter()
    for fname in FILES:
        path = ROOT / fname
        if not path.exists():
            continue
        with path.open() as f:
            for raw in f:
                line = LITERAL_ESCAPE_RE.sub(" ", SPECIAL_RE.sub(" ", raw))
                for m in WORD_RE.finditer(line):
                    counter[m.group(0).lower()] += 1

    hapax = sorted(w for w, c in counter.items() if c == 1 and len(w) >= 4 and "-" not in w and "'" not in w)
    common = sorted(w for w, c in counter.items() if c >= 5 and len(w) >= 4)
    print(f"hapax candidates: {len(hapax)}, common reference: {len(common)}")

    # 효율: hapax별로 같은 길이/±1 길이 common 단어와 비교 + 첫글자/마지막글자가 다르면 빠르게 거름
    by_len: dict[int, list[str]] = {}
    for w in common:
        by_len.setdefault(len(w), []).append(w)

    suspects: list[tuple[str, str, int]] = []
    for h in hapax:
        candidates = []
        for L in (len(h) - 1, len(h), len(h) + 1):
            candidates.extend(by_len.get(L, []))
        for c in candidates:
            if abs(len(h) - len(c)) <= 1 and edit_distance_one(h, c):
                suspects.append((h, c, counter[c]))
                break

    suspects.sort(key=lambda x: -x[2])
    out = ROOT / "_vocab_review" / "typo_candidates.tsv"
    with out.open("w") as f:
        f.write("hapax\tlikely_correct\tcorrect_count\n")
        for h, c, n in suspects:
            f.write(f"{h}\t{c}\t{n}\n")
    print(f"typo candidates → {out}  ({len(suspects)} words)")
    print("\ntop 60:")
    for h, c, n in suspects[:60]:
        print(f"  {h:25s} → {c:25s} ({n})")


if __name__ == "__main__":
    main()
