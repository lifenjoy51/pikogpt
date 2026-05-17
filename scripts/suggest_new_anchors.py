#!/usr/bin/env python3
"""현재 anchor set 3,047에 없는 5세 적합 새 anchor LLM 추천 받기.

입력:
  - llm-playground/data/processed/ccmc_v{2_pro,3_extra}/raw.jsonl (v2/v3 lemma 1,826)
  - data/lemma-anchor-filter/kept_lemmas_merged.txt (신규 lemma 1,221)
출력:
  - data/lemma-anchor-filter/suggestions.jsonl (raw LLM 추천, dedup 전)
  - data/lemma-anchor-filter/suggestions_unique.jsonl (dedup, freq 추정 포함)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib import error, request

EXISTING_KEPT = Path("data/lemma-anchor-filter/kept_lemmas_merged.txt")
import os as _os
MODEL_TAG = _os.environ.get("MODEL_TAG", "flash")  # "flash" or "pro"
OUT = Path(f"data/lemma-anchor-filter/suggestions_{MODEL_TAG}.jsonl")
OUT_UNIQUE = Path(f"data/lemma-anchor-filter/suggestions_unique_{MODEL_TAG}.jsonl")
FAILED = Path(f"data/lemma-anchor-filter/suggestions_failed_{MODEL_TAG}.jsonl")

MODEL = "deepseek/deepseek-v4-pro" if MODEL_TAG == "pro" else "deepseek/deepseek-v4-flash"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ENV_PATHS = [Path(".env"), Path("../llm-playground/.env")]
N_BATCH = 1
N_PER_BATCH = 1000

CATEGORY_HINTS = [
    "Cover ALL 26 categories evenly: animal, plant, body, nature, weather, object, food, "
    "place, family, feeling, action, property, clothing, tool, toy, sound, manner, "
    "quantity, time, job, concept, culture, religion, name, material, other. "
    "Aim for ~30-50 lemmas per category.",
]

SYSTEM_PROMPT_TEMPLATE = """You are an expert curriculum curator for preschool / kindergarten level (age 5) English.

Your task: generate EXACTLY 1000 NEW English LEMMA anchor words (base form) suitable for
a 5-year-old vocabulary lesson. These will be used to write 5-7 simple sentences per
anchor.

# Rules
1. Each lemma is a BASE FORM (singular noun, infinitive verb, base adjective/adverb).
2. The lemma must be **NOT** in the EXISTING ANCHORS list below — generate truly NEW words.
3. The lemma must be COMMON to a 5-year-old's spoken/storybook English.
4. {category_hint}
5. Avoid: proper names (people/brands except santa/disney style universally known),
   technical terms, adult topics, obscure words.

# Output
JSONL only, EXACTLY 1000 lines:
{{"lemma": "<base form>", "category": "<one of: animal|plant|body|nature|weather|object|food|place|family|feeling|action|property|clothing|tool|toy|sound|manner|quantity|time|job|concept|culture|religion|name|material|other>", "reason": "<Korean <=30 chars>"}}

No prose before/after. No markdown fences.

# EXISTING ANCHORS (DO NOT REPEAT any of these {n_existing} words)
{existing_list}
"""


def load_dotenv_into_environ() -> str | None:
    if os.environ.get("OPENROUTER_API_KEY"):
        return None
    for p in ENV_PATHS:
        if not p.exists():
            continue
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            if k.strip() == "OPENROUTER_API_KEY":
                os.environ["OPENROUTER_API_KEY"] = v.strip().strip('"').strip("'")
                return str(p)
    return None


def load_existing_anchors() -> set[str]:
    anchors = set()
    # v2_pro + v3_extra
    for path in [
        "/Users/joey51/works/llm-playground/data/processed/ccmc_v2_pro/raw.jsonl",
        "/Users/joey51/works/llm-playground/data/processed/ccmc_v3_extra/raw.jsonl",
    ]:
        try:
            for line in open(path):
                r = json.loads(line)
                l = r.get("lemma") or r.get("word")
                if l: anchors.add(l.lower())
        except FileNotFoundError:
            pass
    # 신규 1221
    for ln in EXISTING_KEPT.read_text().splitlines():
        if ln.startswith("freq\t") or not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) >= 2:
            anchors.add(parts[1].lower())
    return anchors


def call_openrouter(api_key, system_prompt, user_msg, *, temperature=0.8, max_tokens=32000):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False, "exclude": True},
        "provider": {"only": ["DeepSeek"], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    req = request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=300) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    return obj["choices"][0]["message"]["content"], obj.get("usage") or {}


def call_with_retry(api_key, system_prompt, user_msg, *, temperature=0.8, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            return call_openrouter(api_key, system_prompt, user_msg, temperature=temperature)
        except (error.HTTPError, error.URLError, TimeoutError) as e:
            transient = isinstance(e, error.HTTPError) and 500 <= e.code < 600
            transient |= isinstance(e, (error.URLError, TimeoutError))
            if not transient or attempt == max_attempts:
                raise
            wait = attempt * attempt
            print(f"  [retry {attempt}/{max_attempts}] backoff {wait}s")
            time.sleep(wait)


def parse_jsonl(text):
    rows = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("```"):
            continue
        try:
            obj = json.loads(ln)
            if "lemma" in obj:
                rows.append(obj)
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path, rows):
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    env_src = load_dotenv_into_environ()
    if env_src: print(f"loaded OPENROUTER_API_KEY from {env_src}")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: sys.exit("ERROR: OPENROUTER_API_KEY not set")

    existing = load_existing_anchors()
    print(f"Existing anchors: {len(existing):,}")

    existing_list = ", ".join(sorted(existing))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    total_cost = 0.0
    all_suggestions = []

    for bi in range(N_BATCH):
        hint = CATEGORY_HINTS[bi]
        sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            n_existing=len(existing),
            existing_list=existing_list,
            category_hint=hint,
        )
        user_msg = f"Generate exactly 100 NEW anchor words. Focus hint: {hint}"
        print(f"[batch {bi+1}/{N_BATCH}] temp=0.8, hint: {hint[:60]}...")
        try:
            content, usage = call_with_retry(api_key, sys_prompt, user_msg, temperature=0.8)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            append_jsonl(FAILED, [{"batch": bi+1, "error": str(e)}])
            continue
        parsed = parse_jsonl(content)
        # batch 정보 + dedup 표시
        for r in parsed:
            r["batch"] = bi + 1
        append_jsonl(OUT, parsed)
        all_suggestions.extend(parsed)
        cost = usage.get("cost", 0.0) or 0.0
        total_cost += cost
        print(f"  parsed {len(parsed)}/100, cost ${cost:.4f}")

    # Dedup + 기존 anchor와 충돌 제거
    seen = set()
    unique = []
    for r in all_suggestions:
        l = r["lemma"].lower().strip()
        if not l or l in seen:
            continue
        if l in existing:
            continue  # 기존 anchor와 중복
        seen.add(l)
        unique.append({**r, "lemma": l})

    OUT_UNIQUE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in unique) + "\n")
    print(f"\nTotal suggestions: {len(all_suggestions)}")
    print(f"Unique (after dedup + existing-filter): {len(unique)}")
    print(f"Cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
