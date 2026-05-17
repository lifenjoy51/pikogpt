#!/usr/bin/env python3
"""rare_hapax.tsv에 sentence 컬럼 추가 — 각 hapax 단어가 등장한 문장 (직전 '.' 다음부터 다음 '.'까지).

입력: data/ccmc-all-raw/_vocab_review/rare_hapax.tsv  (word\tcount)
출력: data/ccmc-all-raw/_vocab_review/rare_hapax.tsv  (word\tcount\tsentence)  in-place
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

SPECIAL_RE = re.compile(r"<\|(?:bos|eos|sep|turn)\|>")
LITERAL_ESCAPE_RE = re.compile(r"\\[nrt]")
WS_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z]+(?:[\-'][A-Za-z]+)*")


def load_hapax() -> list[tuple[str, str]]:
    rows = []
    with HAPAX_TSV.open() as f:
        header = next(f).rstrip("\n")
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))
    return rows


def main() -> None:
    rows = load_hapax()
    hapax_set = {w for w, _ in rows}
    print(f"hapax: {len(hapax_set)}")

    sentence_by_word: dict[str, str] = {}

    for fname in FILES:
        path = ROOT / fname
        if not path.exists():
            print(f"SKIP {fname}")
            continue
        with path.open() as f:
            for raw in f:
                line = SPECIAL_RE.sub(" ", raw)
                line = LITERAL_ESCAPE_RE.sub(" ", line)
                # '.' 기준 split. 사용자가 '?'/'!'을 안 언급해서 마침표만.
                for sent in line.split("."):
                    sent_clean = WS_RE.sub(" ", sent).strip()
                    if not sent_clean:
                        continue
                    # 이 문장에 등장하는 hapax 단어 추출
                    sent_words = set()
                    for m in WORD_RE.finditer(sent_clean):
                        sent_words.add(m.group(0).lower())
                    for w in (sent_words & hapax_set):
                        if w not in sentence_by_word:
                            # 마침표 다시 붙임 (사용자 정의: '. 다음 ~ . 까지')
                            sentence_by_word[w] = sent_clean + "."

    matched = sum(1 for w, _ in rows if w in sentence_by_word)
    unmatched = len(rows) - matched
    print(f"matched: {matched}  unmatched: {unmatched}")

    # 갱신
    out_lines = ["word\tcount\tsentence"]
    for w, c in rows:
        sent = sentence_by_word.get(w, "")
        # TSV-safe: 탭/newline은 위에서 이미 공백 정규화됨
        out_lines.append(f"{w}\t{c}\t{sent}")
    HAPAX_TSV.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {HAPAX_TSV}  ({len(rows)} rows)")

    # 미매치 단어 dump
    if unmatched:
        miss_path = HAPAX_TSV.parent / "rare_hapax_unmatched.tsv"
        with miss_path.open("w") as f:
            f.write("word\tcount\n")
            for w, c in rows:
                if w not in sentence_by_word:
                    f.write(f"{w}\t{c}\n")
        print(f"unmatched dump → {miss_path}")


if __name__ == "__main__":
    main()
