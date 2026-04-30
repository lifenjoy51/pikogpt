"""
v3 데이터의 unique word 빈도를 base/it 소스별로 집계하고 low-freq 분포를 정리.

shared/unique_words.txt는 BpePrep이 lowercase=true 가정 정규식으로 만들어
대문자가 잘려나간 잘못된 결과(예: What → hat). 직접 cased 토큰화로 다시 집계.

단어 정의:
  - special token (<|bos|>, <|eos|>, <|turn|>)은 먼저 공백으로 격리
  - [A-Za-z]+(?:'[A-Za-z]+)? 패턴 (apostrophe 1개까지 허용)
  - 숫자/구두점은 분리자 취급
"""

from __future__ import annotations
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/joey51/works/pikogpt/data/two-stage-v3")
SPECIAL = ["<|bos|>", "<|eos|>", "<|turn|>"]
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def count_words(path: Path) -> Counter:
    text = path.read_text()
    for tok in SPECIAL:
        text = text.replace(tok, " ")
    return Counter(WORD_RE.findall(text))


def freq_buckets(counter: Counter) -> dict[str, int]:
    buckets = {"=1": 0, "2-5": 0, "6-10": 0, "11-50": 0, "51-100": 0, ">100": 0}
    for c in counter.values():
        if c == 1:
            buckets["=1"] += 1
        elif c <= 5:
            buckets["2-5"] += 1
        elif c <= 10:
            buckets["6-10"] += 1
        elif c <= 50:
            buckets["11-50"] += 1
        elif c <= 100:
            buckets["51-100"] += 1
        else:
            buckets[">100"] += 1
    return buckets


def main():
    base = count_words(ROOT / "base" / "train.txt")
    it = count_words(ROOT / "it" / "train.txt")

    print(f"{'metric':<25} {'base':>12} {'it':>12} {'union':>12}")
    print("-" * 65)
    print(f"{'total tokens':<25} {sum(base.values()):>12,} {sum(it.values()):>12,} {sum(base.values())+sum(it.values()):>12,}")
    print(f"{'unique types':<25} {len(base):>12,} {len(it):>12,} {len(set(base) | set(it)):>12,}")

    print("\n[freq buckets - per source]")
    bb = freq_buckets(base)
    ib = freq_buckets(it)
    print(f"{'bucket':<10} {'base types':>12} {'base %':>8} {'it types':>12} {'it %':>8}")
    print("-" * 60)
    for k in ["=1", "2-5", "6-10", "11-50", "51-100", ">100"]:
        bp = 100.0 * bb[k] / max(1, len(base))
        ip = 100.0 * ib[k] / max(1, len(it))
        print(f"{k:<10} {bb[k]:>12,} {bp:>7.2f}% {ib[k]:>12,} {ip:>7.2f}%")

    # 교집합/단독 출현
    only_base = set(base) - set(it)
    only_it = set(it) - set(base)
    both = set(base) & set(it)

    print("\n[set membership]")
    print(f"  base-only types : {len(only_base):>10,}")
    print(f"  it-only   types : {len(only_it):>10,}")
    print(f"  both      types : {len(both):>10,}")

    # base-only 중 freq가 낮은 단어 (모델이 학습 안정적으로 배우기 어려운 영역)
    print("\n[base-only, low freq → IT 단계에서 안 보여 전이학습에 약점]")
    low_only_base = {w: base[w] for w in only_base if base[w] <= 5}
    print(f"  base-only & base_freq<=5 : {len(low_only_base):,} types")
    print(f"  그 중 freq=1 hapax       : {sum(1 for c in low_only_base.values() if c == 1):,}")

    # it-only 중 freq 낮은 단어
    print("\n[it-only, low freq → BASE에 없는 신어. dialog 도메인 특화 어휘]")
    low_only_it = {w: it[w] for w in only_it if it[w] <= 5}
    print(f"  it-only & it_freq<=5     : {len(low_only_it):,} types")
    print(f"  그 중 freq=1 hapax       : {sum(1 for c in low_only_it.values() if c == 1):,}")

    # 양쪽에 있지만 한쪽이 매우 희귀
    print("\n[both: low in one, common in other]")
    base_low_it_high = sum(1 for w in both if base[w] <= 5 and it[w] >= 50)
    it_low_base_high = sum(1 for w in both if it[w] <= 5 and base[w] >= 50)
    print(f"  base<=5 & it>=50  : {base_low_it_high:,}  (base에선 희귀, it는 흔함)")
    print(f"  it<=5 & base>=50  : {it_low_base_high:,}  (it에선 희귀, base는 흔함)")

    # base-only 샘플
    print("\n[샘플: base-only freq=1 (alphabet, 앞 30개)]")
    bo1 = sorted(w for w in only_base if base[w] == 1)
    print("  " + "  ".join(bo1[:30]))

    print("\n[샘플: it-only freq=1 (alphabet, 앞 30개)]")
    io1 = sorted(w for w in only_it if it[w] == 1)
    print("  " + "  ".join(io1[:30]))

    # base-only freq>=10 인 단어 = base에서 중요한데 it엔 한 번도 안 나오는 것
    print("\n[샘플: base-only & base_freq>=20 상위 30개 (백과 도메인 어휘)]")
    important_base_only = sorted(
        ((w, base[w]) for w in only_base if base[w] >= 20),
        key=lambda x: -x[1],
    )[:30]
    for w, c in important_base_only:
        print(f"  {w:<25} {c:>6,}")

    # it-only freq>=10 인 단어 = it에서 중요한데 base엔 한 번도 안 나오는 것
    print("\n[샘플: it-only & it_freq>=20 상위 30개 (대화 도메인 어휘)]")
    important_it_only = sorted(
        ((w, it[w]) for w in only_it if it[w] >= 20),
        key=lambda x: -x[1],
    )[:30]
    for w, c in important_it_only:
        print(f"  {w:<25} {c:>6,}")


if __name__ == "__main__":
    main()
