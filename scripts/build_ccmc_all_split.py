#!/usr/bin/env python3
"""ccmc-all-raw → ccmc-all-v4096-v2/{train_lemma,train_other,val_lemma,val_other}.txt 분리.

기존 build_ccmc_all.py(7파일 단일 합본)와 달리, train·val 모두 두 stream으로 분리:
  - train_lemma.txt / val_lemma.txt: lemma_sentences.txt에서 온 라인
  - train_other.txt / val_other.txt: 나머지 6개 소스에서 온 라인

목적: 학습·평가 모두 weighted source loader로 secondaryProb=p_lemma sampling.
train/eval 분포 정확히 일치 → mismatch 없는 평가 신호.

각 stream별로 95:5 split (각각 독립 split → 분포 안정).
셔플 시드: 51 고정.
"""
import random
import re
from pathlib import Path

RAW = Path("data/ccmc-all-raw")
OUT = Path("data/ccmc-all-v4096-v2")
SEED = 51
VAL_RATIO = 0.05

_LITERAL_DOUBLE_NL = re.compile(r"\\n\\n")
_LITERAL_NL = re.compile(r"\\n")


def normalize_wiki_line(s: str) -> str:
    s = _LITERAL_DOUBLE_NL.sub(". ", s)
    s = _LITERAL_NL.sub(". ", s)
    s = re.sub(r"\.\s*\.\s*", ". ", s)
    return " ".join(s.split())


def load_source(name: str, transform=None) -> list[str]:
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    out: list[str] = []
    with path.open() as fin:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if transform is not None:
                line = transform(line)
                if not line:
                    continue
            out.append(f"<|bos|> {line} <|eos|>")
    return out


def split_records(records: list[str], rng: random.Random) -> tuple[list[str], list[str]]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * VAL_RATIO)
    n_train = len(shuffled) - n_val
    return shuffled[:n_train], shuffled[n_train:]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    lemma = load_source("lemma_sentences.txt")
    others: list[str] = []
    other_counts: dict[str, int] = {}
    for name, transform in [
        ("stories.txt", None),
        ("dialogues.txt", None),
        ("wiki.txt", normalize_wiki_line),
        ("cause_seq.txt", None),
        ("chained.txt", None),
        ("counting.txt", None),
    ]:
        recs = load_source(name, transform)
        others.extend(recs)
        other_counts[name] = len(recs)

    rng_lemma = random.Random(SEED)
    rng_other = random.Random(SEED)

    train_lemma, val_lemma = split_records(lemma, rng_lemma)
    train_other, val_other = split_records(others, rng_other)

    (OUT / "train_lemma.txt").write_text("\n".join(train_lemma) + "\n")
    (OUT / "train_other.txt").write_text("\n".join(train_other) + "\n")
    (OUT / "val_lemma.txt").write_text("\n".join(val_lemma) + "\n")
    (OUT / "val_other.txt").write_text("\n".join(val_other) + "\n")

    print(f"# Source counts (raw):")
    print(f"  lemma_sentences.txt: {len(lemma):,}")
    for name, n in other_counts.items():
        print(f"  {name}: {n:,}")
    print(f"# Shuffle seed: {SEED}, val ratio: {VAL_RATIO}")
    print(f"# train_lemma: {len(train_lemma):,} / val_lemma: {len(val_lemma):,}")
    print(f"# train_other: {len(train_other):,} / val_other: {len(val_other):,}")
    print(f"# Output: {OUT}/{{train_lemma,train_other,val_lemma,val_other}}.txt")


if __name__ == "__main__":
    main()
