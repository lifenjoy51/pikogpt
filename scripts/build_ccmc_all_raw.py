#!/usr/bin/env python3
"""CCMC v2/v3/v4/v5 raw 합본 — 토큰화 전 텍스트만, **내용(form) 기준으로 그룹화**.

`data/external-all-raw/{dict,wiki,conv}.txt`와 동일한 카테고리 형식.

출력: data/ccmc-all-raw/
- lemma_sentences.txt — v2_pro(multi_role+sensory+category+contrast) + v3_extra(narrative+scenario+action_sequence)
                       1 line = 1 lemma의 짧은 문장 묶음 (lemma 어휘 grounding)
- stories.txt        — v4_tinystories
                       1 line = 1 short story (5-10 문장 narrative)
- dialogues.txt      — v2_pro(qa축, pair별로 분리) + v5_qa_dialogues
                       1 line = 1 Q/A 또는 다턴 대화. turn 경계는 `<|turn|>`.

통일 정규화:
- `<|bos|>`, `<|eos|>`, `<|sep|>` 제거
- 리터럴 `\\n` / 실제 newline → 공백 (CCMC raw에는 의미적 구조 없음)
- `# Title` 마커 제거 (CCMC raw에는 없지만 안전망)
- `<|turn|>` 양쪽 공백 ` <|turn|> ` 통일
"""
import json
import re
from pathlib import Path

PG = Path("/Users/joey51/works/llm-playground/data/processed")
OUT = Path("data/ccmc-all-raw")
OUT.mkdir(parents=True, exist_ok=True)

_LITERAL_NL = re.compile(r"\\n|\r|\n")
_MD_HEADER = re.compile(r"(^|\s)#\s+")
_STRIP_TOKENS = re.compile(r"<\|bos\|>|<\|eos\|>|<\|sep\|>")
_Q_PREFIX = re.compile(r"(^|\s)Q:\s*")
_A_PREFIX = re.compile(r"(^|\s)A:\s*")


def normalize(s: str) -> str:
    """plain prose 정규화 — 모든 마커/리터럴 \\n 제거 → 단일 공백 join."""
    s = _STRIP_TOKENS.sub(" ", s)
    s = _LITERAL_NL.sub(" ", s)
    s = _MD_HEADER.sub(r"\1", s)
    return " ".join(s.split())


def split_qa_pair(s: str) -> tuple[str, str] | None:
    """v2_pro qa축 한 item: 'Q: q\\nA: a' → (q, a)."""
    s = _STRIP_TOKENS.sub(" ", s)
    s = _LITERAL_NL.sub(" ", s)
    # Q: ... A: ... 분할
    m = re.match(r"\s*Q:\s*(.*?)\s+A:\s*(.*?)\s*$", s, re.DOTALL)
    if not m:
        return None
    q = " ".join(m.group(1).split())
    a = " ".join(m.group(2).split())
    if not q or not a:
        return None
    return q, a


def build_lemma_sentences():
    """v2_pro 4축 + v3_extra 3축 → 1 line = 1 lemma."""
    out = OUT / "lemma_sentences.txt"
    n_records = n_sents = 0
    n_v2 = n_v3 = 0
    with out.open("w") as fout:
        # v2_pro
        with (PG / "ccmc_v2_pro/raw.jsonl").open() as fin:
            for line in fin:
                r = json.loads(line)
                if not r.get("ok"):
                    continue
                d = r.get("data", {})
                sents = []
                for axis in ("multi_role", "sensory", "category", "contrast"):
                    sents.extend(normalize(s) for s in d.get(axis, []) if s)
                sents = [s for s in sents if s]
                if sents:
                    fout.write(" ".join(sents) + "\n")
                    n_records += 1
                    n_sents += len(sents)
                    n_v2 += 1
        # v3_extra
        with (PG / "ccmc_v3_extra/raw.jsonl").open() as fin:
            for line in fin:
                r = json.loads(line)
                if not r.get("ok"):
                    continue
                d = r.get("data", {})
                texts = []
                for axis in ("narrative", "scenario", "action_sequence"):
                    texts.extend(normalize(t) for t in d.get(axis, []) if t)
                texts = [t for t in texts if t]
                if texts:
                    fout.write(" ".join(texts) + "\n")
                    n_records += 1
                    n_sents += len(texts)
                    n_v3 += 1
    return n_records, n_sents, n_v2, n_v3


def build_stories():
    """v4_tinystories 9개 raw.jsonl 합본 + dedup."""
    out = OUT / "stories.txt"
    files = ["flash", "flash_e2", "flash_e3", "flash_e4", "flash_e5", "flash_e6",
             "pro", "pro_e2", "pro_e3"]
    seen = set()
    n_in = n_unique = 0
    per_file = {}
    with out.open("w") as fout:
        for f in files:
            p = PG / f"ccmc_v4_tinystories/{f}/raw.jsonl"
            if not p.exists():
                continue
            file_count = 0
            with p.open() as fin:
                for line in fin:
                    n_in += 1
                    r = json.loads(line)
                    text = r.get("text")
                    if not text:
                        continue
                    norm = normalize(text)
                    if not norm or norm in seen:
                        continue
                    seen.add(norm)
                    fout.write(norm + "\n")
                    n_unique += 1
                    file_count += 1
            per_file[f] = file_count
    return n_in, n_unique, per_file


def build_dialogues():
    """v2_pro qa축 (pair별 분리) + v5_qa_dialogues."""
    out = OUT / "dialogues.txt"
    n_v2_pairs = n_v5 = 0
    seen = set()
    n_v5_in = n_v5_ok = 0
    with out.open("w") as fout:
        # v2_pro qa축 — 한 lemma의 5~6 Q/A pair를 각각 별도 line
        with (PG / "ccmc_v2_pro/raw.jsonl").open() as fin:
            for line in fin:
                r = json.loads(line)
                if not r.get("ok"):
                    continue
                qa_items = r.get("data", {}).get("qa", []) or []
                for item in qa_items:
                    pair = split_qa_pair(item)
                    if pair is None:
                        continue
                    q, a = pair
                    out_line = f"{q} <|turn|> {a}"
                    if out_line in seen:
                        continue
                    seen.add(out_line)
                    fout.write(out_line + "\n")
                    n_v2_pairs += 1
        # v5_qa_dialogues — 1 dialogue per line, turns 사이 <|turn|>
        with (PG / "ccmc_v5_qa_dialogues/flash/raw.jsonl").open() as fin:
            for line in fin:
                n_v5_in += 1
                r = json.loads(line)
                if not r.get("ok"):
                    continue
                n_v5_ok += 1
                d = r.get("dialogue") or {}
                turns = d.get("turns") or []
                turns = [normalize(t) for t in turns if t]
                if not turns:
                    continue
                joined = " <|turn|> ".join(turns)
                if joined in seen:
                    continue
                seen.add(joined)
                fout.write(joined + "\n")
                n_v5 += 1
    return n_v2_pairs, n_v5, n_v5_in, n_v5_ok


def main():
    print("[lemma_sentences]")
    nrec_ls, nsent_ls, nv2, nv3 = build_lemma_sentences()
    print(f"  records={nrec_ls} (v2_pro={nv2}, v3_extra={nv3}), sentences={nsent_ls}")

    print("[stories]")
    n_in, n_u, per = build_stories()
    print(f"  total_in={n_in}, unique_out={n_u}")
    for f, c in per.items():
        print(f"    {f}: {c}")

    print("[dialogues]")
    n_v2p, n_v5d, n5_in, n5_ok = build_dialogues()
    print(f"  v2_pro pairs={n_v2p}, v5_dialogues={n_v5d} (in={n5_in}, ok={n5_ok})")
    print(f"  total dialogue lines={n_v2p + n_v5d}")

    readme = OUT / "README.md"
    readme.write_text(f"""# CCMC all raw — v2/v3/v4/v5 합본 (텍스트 전용, 내용 기준 그룹화)

생성: `scripts/build_ccmc_all_raw.py`
원천: `/Users/joey51/works/llm-playground/data/processed/ccmc_v*/`

`data/external-all-raw/{{dict,wiki,conv}}.txt`와 동일하게 **content type별로 그룹화**.

## 파일

| 파일 | records | 내용 | 출처 |
|---|---:|---|---|
| `lemma_sentences.txt` | {nrec_ls} | 1 line = 1 lemma의 짧은 문장 묶음 (lemma 어휘 grounding). v2_pro 4축 + v3_extra 3축. | v2_pro: multi_role+sensory+category+contrast ({nv2} lemma). v3_extra: narrative+scenario+action_sequence ({nv3} lemma) |
| `stories.txt` | {n_u} | 1 line = 1 short story (5-10 문장). DeepSeek v4 Pro+Flash 합성. dedup 적용. | v4_tinystories 9개 raw.jsonl ({n_in} 입력) |
| `dialogues.txt` | {n_v2p + n_v5d} | 1 line = 1 Q/A 또는 다턴 대화. turn 경계는 ` <\|turn\|> `. | v2_pro qa축 pair별 분리 ({n_v2p}) + v5_qa_dialogues ({n_v5d}) |

## 정규화 공통

- `<\|bos\|>`, `<\|eos\|>`, `<\|sep\|>` → 제거
- 리터럴 `\\n` / 실제 newline → 공백 (CCMC raw에는 paragraph 구조 없음)
- `# Title` 마커 → 제거 (CCMC에는 없지만 안전망)
- `<\|turn\|>` → ` <\|turn\|> ` 양쪽 공백 통일

## CCMC 합성 출처 (참고)

- v2_pro: 1,826 lemma × 5축. DeepSeek v2/Pro.
- v3_extra: 1,826 lemma × 3축. DeepSeek (v3 보강).
- v4_tinystories: 5,932 (verb,noun,adj) tuple × 9 generation. DeepSeek v4 Pro+Flash.
- v5_qa_dialogues: 14,780 spec → 4-7 turn QA. DeepSeek v4 Flash 5:Pro 2.
""")
    print(f"\nREADME: {readme}")


if __name__ == "__main__":
    main()
