"""
사전 데이터 다운로드 + WordNet hypernym 추출.

산출 (data/dictionaries/):
  - simple_dict.json   : nightblade9/simple-english-dictionary processed/filtered.json (121K entries)
  - wordnet_mini.json  : NLTK WordNet에서 lemma → {hypernym, gloss} 추출 dump

KidiIT/Free--Dictionary-API는 entry 1개(`hello`)뿐인 데모라 제외.
"""

from __future__ import annotations
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dictionaries"

SIMPLE_DICT_URL = (
    "https://raw.githubusercontent.com/nightblade9/simple-english-dictionary/"
    "main/processed/filtered.json"
)


def download_simple_dict():
    dst = OUT / "simple_dict.json"
    if dst.exists():
        print(f"skip: {dst.name} already exists ({dst.stat().st_size:,} bytes)")
        return
    print(f"fetching {SIMPLE_DICT_URL}")
    urllib.request.urlretrieve(SIMPLE_DICT_URL, dst)
    print(f"saved: {dst} ({dst.stat().st_size:,} bytes)")


def build_wordnet_mini():
    """lemma → {hypernym, gloss} 첫 번째만 (다중 의미 단순화)."""
    dst = OUT / "wordnet_mini.json"
    if dst.exists():
        print(f"skip: {dst.name} already exists")
        return
    import nltk
    nltk.download("wordnet", quiet=True)
    from nltk.corpus import wordnet as wn

    out: dict[str, dict] = {}
    for synset in wn.all_synsets():
        gloss = synset.definition()
        hyps = synset.hypernyms()
        hypernym_lemma = None
        if hyps:
            hyp_lemmas = hyps[0].lemma_names()
            if hyp_lemmas:
                hypernym_lemma = hyp_lemmas[0].replace("_", " ")
        for lemma in synset.lemma_names():
            key = lemma.replace("_", " ").lower()
            if key in out:
                continue  # 첫 synset 우선 (가장 흔한 의미)
            out[key] = {"gloss": gloss, "hypernym": hypernym_lemma, "pos": synset.pos()}

    with open(dst, "w") as f:
        json.dump(out, f)
    print(f"saved: {dst} ({len(out):,} entries, {dst.stat().st_size:,} bytes)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    download_simple_dict()
    build_wordnet_mini()


if __name__ == "__main__":
    main()
