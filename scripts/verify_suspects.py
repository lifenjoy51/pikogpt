#!/usr/bin/env python3
"""Flash가 의심으로 분류한 단어들의 corpus 컨텍스트 추출."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("data/ccmc-all-raw")
FILES = [
    "lemma_sentences.txt", "stories.txt", "dialogues.txt", "wiki.txt",
    "cause_seq.txt", "chained.txt", "counting.txt",
]
SPECIAL = re.compile(r"<\|(?:bos|eos|sep|turn)\|>")
ESC = re.compile(r"\\[nrt]")

SUSPECTS = """teh ur botony platos heian fars fon canel flos garc trac stika pusath
beyonc budapestr burkinab popocat rden rquez hus hin sn eur cze djurg
bie ds eaw ett fc fm pok oddi rk rmqi sa spmi cangdi dededo diktys
ifmmp iai geul fuark madrile lsch nstika nairobbery toy-add toyco
hel-lo hi-yah hoo-ray cu-ckoo ta-gore ra-bin-dra-nath tap-tap-tap
something-box sun-origin step-step-pause horny waw-tah urimal vah""".split()


def main() -> None:
    found: dict[str, tuple[str, int, str]] = {}
    for fname in FILES:
        path = ROOT / fname
        if not path.exists():
            continue
        with path.open() as f:
            for ln_no, raw in enumerate(f, 1):
                line = ESC.sub(" ", SPECIAL.sub(" ", raw))
                low = line.lower()
                for w in SUSPECTS:
                    if w in found:
                        continue
                    pat = re.compile(r"\b" + re.escape(w) + r"\b")
                    m = pat.search(low)
                    if m:
                        s = max(0, m.start() - 60)
                        e = min(len(line), m.end() + 60)
                        found[w] = (fname, ln_no, line[s:e].strip())

    for w in SUSPECTS:
        if w in found:
            fname, ln_no, ctx = found[w]
            print(f"{w:18s} {fname}:L{ln_no}  ...{ctx}...")
        else:
            print(f"{w:18s} <NOT FOUND in corpus>")


if __name__ == "__main__":
    main()
