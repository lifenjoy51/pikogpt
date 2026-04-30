"""
표제어 병합 + 빈도 화이트리스트 + WordNet hypernym 보강.

화이트리스트 전략:
  v3 base + it 합본에서 cased 단어 빈도 카운트(analyze_v3_low_freq.py와 동일 정규식),
  빈도 ≥ FREQ_THRESHOLD인 lowercase 표제어만 채택.
  → 이미 가진 코퍼스에 등장하는 어휘에만 정의를 부여 → low-freq 문제 직접 보완,
    어른용 niche 어휘는 자동 차단(코퍼스에 안 나오므로 화이트리스트에 없음).

산출:
  data/dictionaries/merged_entries.jsonl  — entry당 한 줄
    {word, pos, definition, synonyms, antonyms, hypernym, v3_freq}
"""

from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DICT_DIR = ROOT / "data" / "dictionaries"
V3_DIR = ROOT / "data" / "two-stage-v3"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SPECIAL = ["<|bos|>", "<|eos|>", "<|turn|>"]
FREQ_THRESHOLD = 10  # v3 코퍼스 등장 빈도 임계

# Simple Dict의 정의에서 종종 보이는 메타 표기 정제
META_BRACKET_RE = re.compile(r"\s*\[[^\]]+\]")  # "[archaic]", "[obs.]"
PHONETIC_RE = re.compile(r"\s*\([^)]*?(?:pron|IPA|/[^)]+/)[^)]*\)", re.IGNORECASE)


def count_v3_words() -> Counter:
    counter: Counter = Counter()
    for src in [V3_DIR / "base" / "train.txt", V3_DIR / "it" / "train.txt"]:
        text = src.read_text()
        for tok in SPECIAL:
            text = text.replace(tok, " ")
        for w in WORD_RE.findall(text):
            counter[w.lower()] += 1
    return counter


def normalize_simple_dict(simple: dict) -> dict:
    """원본 키는 cased + dot 포함. lowercase 단일 단어 키만 채택."""
    out: dict[str, dict] = {}
    for k, v in simple.items():
        if not isinstance(v, dict) or "MEANINGS" not in v:
            continue
        kl = k.strip().lower()
        # 단일 단어만 (apostrophe 1개 허용)
        if not WORD_RE.fullmatch(kl):
            continue
        if kl in out:
            continue  # 첫 번째 entry 우선 (alphabet 순이므로 보통 정상 정의)
        out[kl] = v
    return out


def clean_definition(text: str) -> str:
    if not text:
        return ""
    s = text
    s = META_BRACKET_RE.sub("", s)
    s = PHONETIC_RE.sub("", s)
    s = s.strip().rstrip(".") + "."  # 종결부호 통일
    return s


MAX_MEANINGS = 5  # entry당 의미 상한 (doc 길이 제어)


def extract_simple_entry(raw: dict) -> dict | None:
    """다중 의미 보존. 각 의미: {pos, definition, synonyms, antonyms}."""
    meanings_raw = raw.get("MEANINGS", [])
    if not meanings_raw:
        return None

    out_meanings: list[dict] = []
    for m in meanings_raw[:MAX_MEANINGS]:
        if not isinstance(m, list) or len(m) < 2:
            continue
        pos = m[0] if m[0] else None
        definition = clean_definition(m[1] if len(m) >= 2 else "")
        if not definition or len(definition.split()) < 2:
            continue
        syns = m[2] if len(m) >= 3 and isinstance(m[2], list) else []
        ants = m[3] if len(m) >= 4 and isinstance(m[3], list) else []
        # entry-level fallback (의미별 비어있을 때)
        if not syns:
            syns = raw.get("SYNONYMS", []) or []
        if not ants:
            ants = raw.get("ANTONYMS", []) or []
        # 단일 단어 동의어/반의어만 (다단어 표현은 제외)
        syns = [s for s in syns if isinstance(s, str) and WORD_RE.fullmatch(s.strip())][:4]
        ants = [a for a in ants if isinstance(a, str) and WORD_RE.fullmatch(a.strip())][:4]
        out_meanings.append({
            "pos": pos,
            "definition": definition,
            "synonyms": syns,
            "antonyms": ants,
        })

    if not out_meanings:
        return None
    return {"meanings": out_meanings}


def main():
    print("counting v3 word frequencies...")
    freq = count_v3_words()
    whitelist = {w for w, c in freq.items() if c >= FREQ_THRESHOLD}
    print(f"v3 unique types: {len(freq):,}")
    print(f"freq >= {FREQ_THRESHOLD} whitelist: {len(whitelist):,}")

    print("loading sources...")
    with open(DICT_DIR / "simple_dict.json") as f:
        simple = json.load(f)
    with open(DICT_DIR / "wordnet_mini.json") as f:
        wordnet = json.load(f)
    simple_lower = normalize_simple_dict(simple)
    print(f"simple_dict normalized (single-word, lowercase): {len(simple_lower):,}")

    out: list[dict] = []
    matched_simple = 0
    matched_wordnet = 0
    skipped_short = 0
    skipped_proper = 0
    for word in sorted(whitelist):
        # 1글자 단어 (a, i)는 정의가 의미 없음
        if len(word) < 2:
            skipped_short += 1
            continue
        raw = simple_lower.get(word)
        if not raw:
            continue
        entry = extract_simple_entry(raw)
        if not entry:
            continue
        # 인명 휴리스틱: 첫 의미가 "United States" / "English" / 직업·신분 단어 포함이면 proper noun
        d = entry["meanings"][0]["definition"]
        proper_starts = ("united states ", "english ", "british ", "american ", "french ", "german ", "russian ", "italian ", "spanish ", "greek ")
        if d.lower().startswith(proper_starts) and any(t in d.lower() for t in (" who ", " whose ", " born ", "(", "novelist", "poet", "writer", "actor", "actress", "president", "athlete", "player", "scientist", "physicist", "chemist", "philosopher", "general", "king", "queen", "emperor")):
            skipped_proper += 1
            continue
        matched_simple += 1
        wn_entry = wordnet.get(word, {})
        hyp = wn_entry.get("hypernym")
        # hypernym이 word 자체이거나 다단어 추상명사(블랙리스트)면 제외
        bad_hyp = {"blood group", "associate degree", "letter", "letter of the alphabet"}
        if hyp and hyp.lower() != word and hyp.lower() not in bad_hyp:
            entry["hypernym"] = hyp
            matched_wordnet += 1
        else:
            entry["hypernym"] = None
        entry["word"] = word
        entry["v3_freq"] = freq[word]
        out.append(entry)

    out_path = DICT_DIR / "merged_entries.jsonl"
    with open(out_path, "w") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print()
    print(f"merged entries: {len(out):,}")
    pct_simple = 100 * matched_simple / max(1, len(whitelist))
    pct_wn = 100 * matched_wordnet / max(1, len(out))
    print(f"  simple dict 매칭: {matched_simple:,} ({pct_simple:.1f}% of whitelist)")
    print(f"  wordnet hypernym 보강: {matched_wordnet:,} ({pct_wn:.1f}% of merged)")
    print(f"  skipped short (<2): {skipped_short:,}")
    print(f"  skipped proper noun: {skipped_proper:,}")
    print(f"saved: {out_path}")

    meaning_counts = sorted(len(e["meanings"]) for e in out)
    if meaning_counts:
        n = len(meaning_counts)
        print(f"meanings per entry: p50={meaning_counts[n//2]}, p90={meaning_counts[int(n*0.9)]}, max={meaning_counts[-1]}")
    total_defs = sum(len(e["meanings"]) for e in out)
    print(f"total definitions across all entries: {total_defs:,}")

    print("\n[샘플 — 처음 3 entries (다중 의미)]")
    for e in out[:3]:
        print(f"  {e['word']:<15} ({len(e['meanings'])} 의미, freq={e['v3_freq']})")
        for i, m in enumerate(e["meanings"]):
            print(f"    [{i}] {m['pos']:<10} {m['definition'][:60]}...")
        if e.get("hypernym"):
            print(f"    hypernym: {e['hypernym']}")


if __name__ == "__main__":
    main()
