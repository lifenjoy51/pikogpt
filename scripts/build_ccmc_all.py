#!/usr/bin/env python3
"""ccmc-all-raw 4개 파일 → ccmc-all/{train,val}.txt 합본 + 95:5 split.

입력: data/ccmc-all-raw/
- lemma_sentences.txt  (plain prose, 한 줄 = 1 record)
- stories.txt          (plain prose, 한 줄 = 1 record)
- dialogues.txt        (plain prose, 한 줄 = 1 record, turn 경계 `<|turn|>`)
- wiki.txt             (`title\\n\\nbody` 형식, 리터럴 `\\n` 보존 — 표제어 구분용)

wiki.txt 처리:
- 리터럴 `\\n\\n` (literal backslash+n × 2) → `. ` (마침표 자연화, 옵션 D)
- 표제어를 문장처럼 본문 앞에 붙여 자연 영어 흐름화. 특수 토큰 없음.
- 본문 내 추가 `\\n\\n`이 있는 wiki도 동일하게 마침표 치환

다른 3개 파일은 이미 build_ccmc_all_raw.py에서 정규화됨 (그대로 사용).

셔플 시드: 51 고정 (재현성).
Split 비율: train 95% / val 5%.

출력: data/ccmc-all/
- train.txt / val.txt
"""
import random
import re
from pathlib import Path

RAW = Path("data/ccmc-all-raw")
OUT = Path("data/ccmc-all")
SEED = 51
VAL_RATIO = 0.05

# 리터럴 `\n\n` (2-char `\\n` × 2 sequence) → `. `
_LITERAL_DOUBLE_NL = re.compile(r"\\n\\n")
_LITERAL_NL = re.compile(r"\\n")


def normalize_wiki_line(s: str) -> str:
    """wiki.txt 라인의 리터럴 `\\n\\n`을 `. `로 치환 (표제어 자연화).

    잔여 `\\n`(쌍이 아닌 단독)도 안전하게 `. `로 치환 (이론상 없지만 안전망).
    중복 마침표 (e.g. `U.S.\\n\\n...` → `U.S.. ...`)는 후처리에서 정리.
    """
    s = _LITERAL_DOUBLE_NL.sub(". ", s)
    s = _LITERAL_NL.sub(". ", s)
    s = re.sub(r"\.\s*\.\s*", ". ", s)  # `. . ` → `. ` (중복 마침표 정리)
    return " ".join(s.split())


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    records: list[str] = []
    counts: dict[str, int] = {}

    for name, transform in [
        ("lemma_sentences.txt", None),
        ("stories.txt", None),
        ("dialogues.txt", None),
        ("wiki.txt", normalize_wiki_line),
        ("cause_seq.txt", None),
        ("chained.txt", None),
        ("counting.txt", None),
    ]:
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(f"missing: {path}")
        with path.open() as fin:
            n = 0
            for line in fin:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                if transform is not None:
                    line = transform(line)
                    if not line:
                        continue
                records.append(f"<|bos|> {line} <|eos|>")
                n += 1
            counts[name] = n

    rng = random.Random(SEED)
    rng.shuffle(records)

    n_total = len(records)
    n_val = round(n_total * VAL_RATIO)
    n_train = n_total - n_val
    train_records = records[:n_train]
    val_records = records[n_train:]

    (OUT / "train.txt").write_text("\n".join(train_records) + "\n")
    (OUT / "val.txt").write_text("\n".join(val_records) + "\n")

    print(f"# Source counts (raw):")
    for name, n in counts.items():
        print(f"  {name}: {n:,}")
    print(f"  total: {n_total:,}")
    print(f"# Shuffle seed: {SEED}")
    print(f"# Split: train {n_train:,} ({n_train/n_total*100:.1f}%)"
          f" / val {n_val:,} ({n_val/n_total*100:.1f}%)")
    print(f"# Output: {OUT}/train.txt, {OUT}/val.txt")


if __name__ == "__main__":
    main()
