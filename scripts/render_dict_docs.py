"""
merged_entries.jsonl → 자연어 doc 변환 + 90:10 split.

각 entry는 위키 doc과 동일한 `<|bos|>\\n# Word\\n\\n{문단}\\n<|eos|>` 형식으로 wrap.
모델이 위키 본문과 동일한 doc 패턴을 학습하도록 일관성 유지.

doc 예시:
  <|bos|>
  # Apple

  An apple is a round fruit that grows on trees.
  An apple is a kind of fruit.
  Similar words: fruit.
  <|eos|>

산출:
  data/three-stage-v4/dict/{train.txt, val.txt}
"""

from __future__ import annotations
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from text_cleaning import SMART, NON_ASCII

import re as _re

_WS = _re.compile(r"\s+")


def clean_inline(text: str) -> str:
    """1라인=1doc용. SMART/non-ASCII 정제 + 모든 공백을 단일 공백으로."""
    text = text.translate(SMART)
    text = NON_ASCII.sub("", text)
    return _WS.sub(" ", text).strip()

DICT_DIR = ROOT / "data" / "dictionaries"
OUT_DIR = ROOT / "data" / "three-stage-v4" / "dict"

VAL_FRAC = 0.10
SEED = 42

VOWELS = set("aeiou")


def article_for(word: str) -> str:
    return "An" if word and word[0].lower() in VOWELS else "A"


def render_meaning_line(word: str, art: str, cap: str, pos: str, definition: str, is_first: bool) -> str:
    if pos == "Noun":
        if is_first:
            return f"{art} {word} is {definition}"
        return f"{art} {word} can also be {definition}"
    # Verb / Adjective / Adverb / 그 외
    if is_first:
        return f"{cap} means {definition}"
    return f"{cap} can also mean {definition}"


def render_doc(entry: dict) -> str:
    """1 line = 1 doc. doc 내부 단락 구분은 리터럴 \\n으로 (실제 개행 X)."""
    word = entry["word"]
    cap = word.capitalize()
    art = article_for(word)
    lines: list[str] = [f"# {cap}"]

    meanings = entry.get("meanings") or []
    first_pos = ((meanings[0].get("pos") if meanings else "") or "").strip()

    for idx, m in enumerate(meanings):
        pos = (m.get("pos") or "").strip()
        definition = m["definition"].rstrip(".") + "."
        lines.append(render_meaning_line(word, art, cap, pos, definition, idx == 0))

    if first_pos == "Noun" and entry.get("hypernym"):
        hyp = entry["hypernym"].rstrip(".").lower()
        if hyp and hyp != word:
            hyp_art = "an" if hyp[0] in VOWELS else "a"
            lines.append(f"{art} {word} is {hyp_art} kind of {hyp}.")

    # 별도 정보(syns/ants) 앞에 빈 줄 — 본문과 분리
    if meanings:
        syns = [s.lower() for s in (meanings[0].get("synonyms") or []) if s and s.lower() != word]
        ants = [a.lower() for a in (meanings[0].get("antonyms") or []) if a and a.lower() != word]
        if syns or ants:
            lines.append("")
            if syns:
                lines.append(f"Similar: {', '.join(syns)}.")
            if ants:
                lines.append(f"Opposite: {', '.join(ants)}.")

    body = "\\n".join(lines)  # 리터럴 \n
    return f"<|bos|>\\n{body}\\n<|eos|>"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = DICT_DIR / "merged_entries.jsonl"
    entries = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    print(f"loaded: {len(entries):,} entries from {src.name}")

    docs = [render_doc(e) for e in entries]
    # 1라인=1doc 정제: SMART/non-ASCII + 모든 공백을 단일 공백으로
    docs = [clean_inline(d) for d in docs]
    docs = [d for d in docs if d]

    rng = random.Random(SEED)
    rng.shuffle(docs)

    n_val = max(1, int(len(docs) * VAL_FRAC))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]
    print(f"split: train={len(train_docs):,} val={len(val_docs):,} (val_frac={VAL_FRAC})")

    train_path = OUT_DIR / "train.txt"
    val_path = OUT_DIR / "val.txt"
    # 1 line = 1 doc — doc 사이도 \n 한 개 (빈 줄 X)
    train_path.write_text("\n".join(train_docs) + "\n")
    val_path.write_text("\n".join(val_docs) + "\n")

    train_words = sum(len(d.split()) for d in train_docs)
    val_words = sum(len(d.split()) for d in val_docs)
    print(f"train.txt: {train_path.stat().st_size:,} bytes, ~{train_words:,} words")
    print(f"val.txt:   {val_path.stat().st_size:,} bytes, ~{val_words:,} words")

    print("\n[샘플 — train_docs[0..2]]")
    for d in train_docs[:3]:
        print("---")
        print(d)


if __name__ == "__main__":
    main()
