"""
v4 데이터의 unique word 빈도를 dict/wiki/it 소스별로 집계.

v3와 동일한 cased 토큰화로 직접 단어 빈도를 측정. 사전 추가 후
hapax(=1) 비율 / base-only 어휘 변화를 정량 비교하기 위한 진단.
"""

from __future__ import annotations
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/joey51/works/pikogpt/data/three-stage-v4")
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
    dict_c = count_words(ROOT / "dict" / "train.txt")
    wiki_c = count_words(ROOT / "wiki" / "train.txt")
    it_c = count_words(ROOT / "it" / "train.txt")
    base_c = count_words(ROOT / "base" / "train.txt")  # dict + wiki

    print(f"{'metric':<25} {'dict':>10} {'wiki':>12} {'it':>12} {'base(d+w)':>12} {'union':>12}")
    print("-" * 90)
    union_types = set(dict_c) | set(wiki_c) | set(it_c)
    print(
        f"{'total tokens':<25} "
        f"{sum(dict_c.values()):>10,} {sum(wiki_c.values()):>12,} "
        f"{sum(it_c.values()):>12,} {sum(base_c.values()):>12,} "
        f"{sum(dict_c.values())+sum(wiki_c.values())+sum(it_c.values()):>12,}"
    )
    print(
        f"{'unique types':<25} "
        f"{len(dict_c):>10,} {len(wiki_c):>12,} {len(it_c):>12,} "
        f"{len(base_c):>12,} {len(union_types):>12,}"
    )

    print("\n[freq buckets - per source]")
    print(f"{'bucket':<10} {'dict':>10} {'dict %':>8} {'wiki':>10} {'wiki %':>8} {'it':>10} {'it %':>8}")
    print("-" * 70)
    db = freq_buckets(dict_c)
    wb = freq_buckets(wiki_c)
    ib = freq_buckets(it_c)
    for k in ["=1", "2-5", "6-10", "11-50", "51-100", ">100"]:
        dp = 100.0 * db[k] / max(1, len(dict_c))
        wp = 100.0 * wb[k] / max(1, len(wiki_c))
        ip = 100.0 * ib[k] / max(1, len(it_c))
        print(f"{k:<10} {db[k]:>10,} {dp:>7.2f}% {wb[k]:>10,} {wp:>7.2f}% {ib[k]:>10,} {ip:>7.2f}%")

    only_wiki = set(wiki_c) - set(it_c)
    only_it = set(it_c) - set(wiki_c)
    only_dict = set(dict_c) - (set(wiki_c) | set(it_c))
    dict_helps_wiki = set(dict_c) & {w for w in only_wiki if wiki_c[w] <= 5}
    dict_helps_it = set(dict_c) & {w for w in only_it if it_c[w] <= 5}

    print("\n[set membership]")
    print(f"  only_wiki (not in it)        : {len(only_wiki):>10,}")
    print(f"  only_it (not in wiki)        : {len(only_it):>10,}")
    print(f"  only_dict (not in wiki/it)   : {len(only_dict):>10,}")
    print(f"  dict ∩ (wiki-only & freq≤5)  : {len(dict_helps_wiki):>10,}  (dict가 wiki의 저빈도 어휘 보강)")
    print(f"  dict ∩ (it-only & freq≤5)    : {len(dict_helps_it):>10,}  (dict가 it의 저빈도 어휘 보강)")


if __name__ == "__main__":
    main()
