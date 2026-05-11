#!/usr/bin/env python3
"""wiki-synth 결과 자동 검증 — 어휘(A) + 구조 sanity(C).

입력:
  data/wiki-synth/raw.jsonl
  data/external-all-raw/wiki.txt                            (id → 원본 body length)
  data/ccmc-v2-pro.20260506_v01/shared/unique_words.txt     (5세 어휘 ground truth proxy)

출력:
  data/wiki-synth/validation.jsonl
    {id, hard_count, hard_words(top5), unique_tokens, hard_ratio,
     sentence_count, avg_sent_words, has_title_in_first, issues}

검증 항목:
  A. 어휘
     - text의 단어 중 CCMC vocab(freq≥5) ∪ Dolch 220에 없는 단어 count
     - 단순 stem(trailing s/es/ed/ing/'s) 시도 후 미매칭만 hard로 카운트
  C. 구조 sanity (issues list로 누적)
     - 질문문자(?)가 있음 → 'has_question'
     - markdown 패턴(# / ** / 행 시작 - / 행 시작 *) → 'has_markdown'
     - 문장 안의 sub-clause(. ! ? , ; : 로 분리) 중 가장 긴 것의 word count > 20
       → 'sentence_length_outlier' (list-style "a, b, c, d" 같은 건 제외됨)
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

RAW = Path("data/wiki-synth/raw.jsonl")
WIKI_BODY = Path("data/external-all-raw/wiki.txt")
CCMC_VOCAB = Path("data/ccmc-v2-pro.20260506_v01/shared/unique_words.txt")
OUT = Path("data/wiki-synth/validation.jsonl")

CCMC_MIN_FREQ = 5

# Dolch 220 sight words (5세 일상 어휘 안전망)
DOLCH_220 = set("""
a about after again all always am an and any are around as ask at ate away
be because been before best better big black blue both bring brown but buy by
call came can carry clean cold come could cut did do does done don't down draw drink
eat eight every fall far fast find first five fly for found four from full funny
gave get give go goes going good got green grow had has have he her here him his
hold hot how hurt I if in into is it its jump just keep kind know
laugh let light like little live long look made make many may me much must my
myself never new no not now of off old on once one only open or our out over own
pick play please pretty pull put ran read red ride right round run
said saw say see seven shall she show sing sit six sleep small so some soon start stop
take tell ten thank that the their them then there these they think this those three to today
together too try two under up upon us use very walk want warm was wash we well went were what
when where which white who why will wish with work would write yellow yes you your
""".split())


def load_5se_vocab() -> set[str]:
    """CCMC vocab freq>=5 + Dolch 220 + 흔한 보조 단어."""
    vocab: set[str] = set(DOLCH_220)
    for ln in CCMC_VOCAB.read_text().splitlines():
        if "\t" not in ln:
            continue
        w, f = ln.rstrip().split("\t")
        try:
            if int(f) >= CCMC_MIN_FREQ:
                vocab.add(w.lower())
        except ValueError:
            continue
    # 기본 영어 표지/숫자 단어
    vocab.update("zero one two three four five six seven eight nine ten eleven twelve "
                 "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
                 "thirty forty fifty sixty seventy eighty ninety hundred thousand million billion "
                 "monday tuesday wednesday thursday friday saturday sunday "
                 "january february march april may june july august september october november december "
                 "english spanish french german chinese japanese korean".split())
    return vocab


def load_wiki_bodies() -> dict[int, int]:
    """id → body char length."""
    out: dict[int, int] = {}
    with WIKI_BODY.open() as f:
        for i, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            out[i] = len(line)
    return out


STEM_SUFFIXES = ["ing", "ed", "es", "s", "'s", "n't"]


def in_vocab(word: str, vocab: set[str]) -> bool:
    if word in vocab:
        return True
    for suf in STEM_SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            base = word[: -len(suf)]
            if base in vocab:
                return True
            # 더블 자음 단축 (running -> run)
            if base and base[-1] == base[-2] if len(base) >= 2 else False:
                if base[:-1] in vocab:
                    return True
            # ies -> y
            if suf == "es" and base.endswith("i"):
                if base[:-1] + "y" in vocab:
                    return True
    return False


WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")


def validate_one(rec: dict, vocab: set[str], body_chars: int) -> dict:
    text = rec.get("text") or ""
    title = rec.get("title") or ""
    title_lower = title.lower()
    title_words = set(re.findall(r"[a-z]+", title_lower))

    tokens = [t.lower() for t in WORD_RE.findall(text)]
    unique_tokens = set(tokens)
    hard: list[str] = []
    for tok in unique_tokens:
        # title 단어는 면제 (system prompt가 허용)
        if tok in title_words:
            continue
        if in_vocab(tok, vocab):
            continue
        hard.append(tok)
    hard_sorted = sorted(hard)

    # 문장 분리
    sentences = [s.strip() for s in re.split(r"[.!?]+\s+", text) if s.strip()]
    sent_word_counts = [len(re.findall(r"[A-Za-z]+", s)) for s in sentences]
    sent_word_counts = [n for n in sent_word_counts if n > 0]
    avg_sent_words = statistics.mean(sent_word_counts) if sent_word_counts else 0.0

    # sub-clause 분리 (.!?,;: 모두 break로) — list-style "a, b, c, d" 같은 문장이
    # 단순 word count로 outlier로 잡히는 false positive 방지
    clauses = [c.strip() for c in re.split(r"[.!?,;:]+\s*", text) if c.strip()]
    clause_word_counts = [len(re.findall(r"[A-Za-z]+", c)) for c in clauses]
    clause_word_counts = [n for n in clause_word_counts if n > 0]
    max_clause_words = max(clause_word_counts) if clause_word_counts else 0

    issues: list[str] = []
    first = sentences[0].lower() if sentences else ""
    has_title_in_first = bool(title_words & set(re.findall(r"[a-z]+", first)))
    # 'title_missing_in_first', 'longer_than_source' 는 의미 있는 위반이 아니라 제외
    if "?" in text:
        issues.append("has_question")
    if re.search(r"(^|\n)\s*[#*\-]\s+", text) or "**" in text:
        issues.append("has_markdown")
    if max_clause_words > 20:
        issues.append("sentence_length_outlier")

    return {
        "id": rec["id"],
        "hard_count": len(hard_sorted),
        "hard_words": hard_sorted[:5],
        "unique_tokens": len(unique_tokens),
        "hard_ratio": round(len(hard_sorted) / max(1, len(unique_tokens)) * 100, 1),
        "sentence_count": len(sentences),
        "avg_sent_words": round(avg_sent_words, 1),
        "max_clause_words": max_clause_words,
        "has_title_in_first": has_title_in_first,
        "issues": issues,
    }


def main() -> None:
    vocab = load_5se_vocab()
    bodies = load_wiki_bodies()
    print(f"vocab size (5세 proxy): {len(vocab):,}")

    rows = [json.loads(ln) for ln in RAW.read_text().splitlines() if ln.strip() and json.loads(ln).get("ok")]
    print(f"validating {len(rows):,} ok records...")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fout:
        results: list[dict] = []
        for r in rows:
            v = validate_one(r, vocab, bodies.get(r["id"], 0))
            results.append(v)
            fout.write(json.dumps(v, ensure_ascii=False) + "\n")

    # 통계
    hard_counts = [v["hard_count"] for v in results]
    hard_ratios = [v["hard_ratio"] for v in results]
    sent_counts = [v["sentence_count"] for v in results]
    avg_sent = [v["avg_sent_words"] for v in results]
    n_has_issue = sum(1 for v in results if v["issues"])

    issue_counter: dict[str, int] = {}
    for v in results:
        for i in v["issues"]:
            issue_counter[i] = issue_counter.get(i, 0) + 1

    print(f"\n=== 검증 요약 ===")
    print(f"  hard_count   avg={statistics.mean(hard_counts):.1f} p50={int(statistics.median(hard_counts))} p90={sorted(hard_counts)[len(hard_counts)*9//10]} max={max(hard_counts)}")
    print(f"  hard_ratio%  avg={statistics.mean(hard_ratios):.1f} p50={statistics.median(hard_ratios):.1f} p90={sorted(hard_ratios)[len(hard_ratios)*9//10]:.1f} max={max(hard_ratios):.1f}")
    print(f"  sentences    avg={statistics.mean(sent_counts):.1f} p50={int(statistics.median(sent_counts))}")
    print(f"  sent_words   avg={statistics.mean(avg_sent):.1f}")
    print(f"  records with any issue: {n_has_issue} / {len(results)} ({n_has_issue/len(results)*100:.1f}%)")
    print(f"\n  issues:")
    for k, n in sorted(issue_counter.items(), key=lambda x: -x[1]):
        print(f"    {k:<28} {n:>5}")

    # hard_count 상위 10 노출
    top_hard = sorted(results, key=lambda v: -v["hard_count"])[:10]
    print(f"\n  hard_count top 10:")
    for v in top_hard:
        print(f"    id={v['id']:>5}  hard={v['hard_count']:>3}  ratio={v['hard_ratio']:>4.1f}%  sample={v['hard_words']}")


if __name__ == "__main__":
    main()
