#!/usr/bin/env python3
"""rare_hapax.tsv 남아있는 word에 대해 sentence 컬럼 재추출.

split boundary 확장: [.!?] 종결자 + <|bos|>/<|eos|>/<|sep|>/<|turn|> special marker + literal \\n/\\r/\\t.
기존 sentence는 '.'만 boundary로 써서 dialogues 등에서 turn 가로지르는 문제가 있었음.

입력: data/ccmc-all-raw/_vocab_review/rare_hapax.tsv  (word\tcount\tsentence)
출력: 같은 파일 in-place — sentence 컬럼 교체
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("data/ccmc-all-raw")
HAPAX_TSV = ROOT / "_vocab_review" / "rare_hapax.tsv"
FILES = [
    "lemma_sentences.txt", "stories.txt", "dialogues.txt", "wiki.txt",
    "cause_seq.txt", "chained.txt", "counting.txt",
]

# boundary: special markers + literal escapes (turn/bos/eos/sep + \n \r \t)
BOUNDARY_RE = re.compile(r"<\|(?:bos|eos|sep|turn)\|>|\\[nrt]")
# 종결자 직후 공백 — 문장 split
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WS_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z]+(?:[\-'][A-Za-z]+)*")


def split_sentences(line: str) -> list[str]:
    out = []
    for chunk in BOUNDARY_RE.split(line):
        for s in SENT_SPLIT_RE.split(chunk):
            s = WS_RE.sub(" ", s).strip()
            if s:
                out.append(s)
    return out


def load_hapax() -> list[tuple[str, str, str]]:
    rows = []
    with HAPAX_TSV.open() as f:
        next(f)
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
            elif len(parts) == 2:
                rows.append((parts[0], parts[1], ""))
    return rows


def main() -> None:
    rows = load_hapax()
    hapax_set = {w for w, _, _ in rows}
    print(f"hapax: {len(hapax_set)}")

    new_sentence: dict[str, str] = {}

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
                        if w not in new_sentence:
                            new_sentence[w] = sent

    matched = sum(1 for w in hapax_set if w in new_sentence)
    print(f"matched: {matched}  unmatched: {len(hapax_set) - matched}")

    out_lines = ["word\tcount\tsentence"]
    changed = 0
    for w, c, old_sent in rows:
        new_sent = new_sentence.get(w, "")
        if new_sent != old_sent:
            changed += 1
        out_lines.append(f"{w}\t{c}\t{new_sent}")
    HAPAX_TSV.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {HAPAX_TSV}  ({len(rows)} rows, {changed} changed)")


if __name__ == "__main__":
    main()
