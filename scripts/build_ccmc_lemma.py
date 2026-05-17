#!/usr/bin/env python3
"""lemma_sentences.txt만 사용해 BOS/EOS 래핑된 train/val 데이터셋 생성.

입력: data/ccmc-all-raw/lemma_sentences.txt (한 줄 = 1 sentence)
출력: data/ccmc-lemma-v1024/{train,val}.txt — BPE 학습용 입력
      이후 `./gradlew runBpe --args="data/ccmc-lemma-v1024 1024"` 로 vocab=1024 BPE 학습

build_ccmc_all.py 패턴 그대로:
  - 빈 줄 무시
  - 각 record를 "<|bos|> <line> <|eos|>" 로 래핑
  - SEED=51로 셔플, 95:5 split
"""
import random
from pathlib import Path

RAW = Path("data/ccmc-all-raw")
OUT = Path("data/ccmc-lemma-v1024")
SEED = 51
VAL_RATIO = 0.05


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    src = RAW / "lemma_sentences.txt"
    if not src.exists():
        raise FileNotFoundError(f"missing: {src}")

    records: list[str] = []
    with src.open() as fin:
        for line in fin:
            line = line.rstrip("\n").strip()
            if not line:
                continue
            records.append(f"<|bos|> {line} <|eos|>")

    rng = random.Random(SEED)
    rng.shuffle(records)

    n_total = len(records)
    n_val = round(n_total * VAL_RATIO)
    n_train = n_total - n_val
    train_records = records[:n_train]
    val_records = records[n_train:]

    (OUT / "train.txt").write_text("\n".join(train_records) + "\n")
    (OUT / "val.txt").write_text("\n".join(val_records) + "\n")

    print(f"# lemma_sentences.txt → ccmc-lemma-v1024")
    print(f"  총 records: {n_total:,}")
    print(f"  train: {n_train:,}")
    print(f"  val:   {n_val:,}")
    print(f"  → {OUT}/train.txt")
    print(f"  → {OUT}/val.txt")
    print()
    print("다음 단계: BPE 학습")
    print(f'  ./gradlew runBpe --args="{OUT.as_posix()} 1024"')


if __name__ == "__main__":
    main()
