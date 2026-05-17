#!/usr/bin/env python3
"""lemma_sentences.txt — 한 줄에 묶인 lemma 예시 문장들을 문장 단위로 분리.

before: 한 줄 = 한 lemma의 ~25개 예시 문장 (semantically disconnected)
after:  한 줄 = 한 문장

split boundary: [.!?] + whitespace (5세 어휘 strict 코퍼스라 약어 충돌 거의 없음)
empty/whitespace-only sentence drop.

in-place 수정: data/ccmc-all-raw/lemma_sentences.txt
"""
from __future__ import annotations

import re
from pathlib import Path

PATH = Path("data/ccmc-all-raw/lemma_sentences.txt")
SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WS_RE = re.compile(r"\s+")


def main() -> None:
    raw = PATH.read_text()
    in_lines = raw.splitlines()
    out_sentences: list[str] = []
    for line in in_lines:
        line = line.strip()
        if not line:
            continue
        for s in SPLIT_RE.split(line):
            s = WS_RE.sub(" ", s).strip()
            if s:
                out_sentences.append(s)
    PATH.write_text("\n".join(out_sentences) + "\n")
    print(f"in_lines={len(in_lines):,}  out_sentences={len(out_sentences):,}")
    print(f"avg sentences/lemma ≈ {len(out_sentences)/len(in_lines):.1f}")


if __name__ == "__main__":
    main()
