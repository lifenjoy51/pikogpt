#!/usr/bin/env python3
"""rare_hapax_rewritten.tsv의 변경된 row를 corpus 파일에 적용 + rare_hapax.tsv에서 제거.

변경된 row = rewritten 비어있지 않고 rewritten != sentence
적용 전략: corpus 라인에 sentence 문자열이 그대로 등장하면 rewritten으로 substring 교체.
매치 안 되면 (wiki.txt 등 literal \\n 포맷) skip — 해당 word는 rare_hapax.tsv에 남김.

입력:
  data/ccmc-all-raw/_vocab_review/rare_hapax_rewritten.tsv
  data/ccmc-all-raw/_vocab_review/rare_hapax.tsv
  data/ccmc-all-raw/{lemma_sentences,stories,dialogues,wiki,cause_seq,chained,counting}.txt

출력 (in-place):
  data/ccmc-all-raw/*.txt — sentence → rewritten 교체
  data/ccmc-all-raw/_vocab_review/rare_hapax.tsv — 적용된 word row 제거
  data/ccmc-all-raw/_vocab_review/apply_log.tsv — applied / skipped 기록
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path("data/ccmc-all-raw")
REWRITTEN_TSV = ROOT / "_vocab_review" / "rare_hapax_rewritten.tsv"
HAPAX_TSV = ROOT / "_vocab_review" / "rare_hapax.tsv"
LOG_TSV = ROOT / "_vocab_review" / "apply_log.tsv"

CORPUS_FILES = [
    "lemma_sentences.txt", "stories.txt", "dialogues.txt", "wiki.txt",
    "cause_seq.txt", "chained.txt", "counting.txt",
]


def load_changes() -> list[tuple[str, str, str]]:
    rows = []
    with REWRITTEN_TSV.open() as f:
        next(f)
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) == 4:
                word, _, sent, rw = parts
                if rw and rw != sent:
                    rows.append((word, sent, rw))
    return rows


def main() -> None:
    changes = load_changes()
    print(f"changes to attempt: {len(changes)}")

    applied: dict[str, str] = {}  # word → fname where applied
    pending: dict[str, tuple[str, str]] = {(c[0]): (c[1], c[2]) for c in changes}

    for fname in CORPUS_FILES:
        path = ROOT / fname
        if not path.exists():
            print(f"SKIP {fname}")
            continue
        text = path.read_text(encoding="utf-8")
        applied_in_file = 0
        for word, (sent, rw) in list(pending.items()):
            idx = text.find(sent)
            if idx == -1:
                continue
            text = text[:idx] + rw + text[idx + len(sent):]
            applied[word] = fname
            applied_in_file += 1
            del pending[word]
        if applied_in_file:
            path.write_text(text, encoding="utf-8")
            print(f"  {fname}: applied {applied_in_file}")
        else:
            print(f"  {fname}: applied 0")

    skipped: dict[str, tuple[str, str]] = dict(pending)
    print(f"\napplied={len(applied)}  skipped={len(skipped)}")

    # log
    with LOG_TSV.open("w") as f:
        f.write("status\tword\tfile_or_reason\n")
        for word, fname in sorted(applied.items()):
            f.write(f"applied\t{word}\t{fname}\n")
        for word in sorted(skipped):
            f.write(f"skipped\t{word}\tno substring match\n")
    print(f"log → {LOG_TSV}")

    # rare_hapax.tsv 갱신 — applied된 word만 제거
    out_lines = []
    removed = 0
    with HAPAX_TSV.open() as f:
        header = next(f)
        out_lines.append(header.rstrip("\n"))
        for ln in f:
            word = ln.split("\t", 1)[0]
            if word in applied:
                removed += 1
                continue
            out_lines.append(ln.rstrip("\n"))
    HAPAX_TSV.write_text("\n".join(out_lines) + "\n")
    print(f"rare_hapax.tsv: removed {removed} rows  (now {len(out_lines)-1} data rows)")


if __name__ == "__main__":
    main()
