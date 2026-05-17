#!/usr/bin/env python3
"""Lemma anchor 재판정 v2 — v1에서 drop된 lemma만 v2 프롬프트로 재판정.

입력: data/lemma-anchor-filter/judgments.jsonl (v1 결과, keep=false만 사용)
출력:
  data/lemma-anchor-filter/judgments_v2.jsonl   재판정 결과 (v2 keep/drop)
  data/lemma-anchor-filter/judgments_merged.jsonl  v1(keep) + v2(재판정) 통합 최종
  data/lemma-anchor-filter/kept_lemmas_merged.txt
  data/lemma-anchor-filter/failed_v2.jsonl

설정:
- 모델: deepseek/deepseek-v4-flash
- batch: 133
- shuffle seed=51 (v1과 동일)
- v2 system prompt: 5개 추가 KEEP 카테고리 (manner adverb, quantity, broad social,
  major religion/culture, common abstract)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from urllib import error, request

JUDG_V1 = Path("data/lemma-anchor-filter/judgments.jsonl")
JUDG_V2 = Path("data/lemma-anchor-filter/judgments_v2.jsonl")
JUDG_MERGED = Path("data/lemma-anchor-filter/judgments_merged.jsonl")
KEPT_MERGED = Path("data/lemma-anchor-filter/kept_lemmas_merged.txt")
CATCOUNT_MERGED = Path("data/lemma-anchor-filter/category_counts_merged.txt")
FAILED = Path("data/lemma-anchor-filter/failed_v2.jsonl")

SYSTEM_PROMPT_PATH = Path("prompts/lemma_anchor_filter_system_v2.txt")

MODEL = "deepseek/deepseek-v4-flash"
BATCH_SIZE = 133
SHUFFLE_SEED = 51
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ENV_PATHS = [Path(".env"), Path("../llm-playground/.env")]


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


def load_drops() -> list[dict]:
    """v1 judgments에서 keep=false만 추출 → 새 id 부여 + shuffle."""
    rows = []
    for ln in JUDG_V1.read_text().splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        if obj.get("keep"):
            continue
        rows.append({"lemma": obj["lemma"], "freq": obj.get("freq", 0)})
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(rows)
    for i, r in enumerate(rows, start=1):
        r["id"] = i
    return rows


def load_done_ids() -> set[int]:
    if not JUDG_V2.exists():
        return set()
    done = set()
    for ln in JUDG_V2.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            done.add(json.loads(ln)["id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def call_openrouter(api_key, system_prompt, user_msg):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
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
    with request.urlopen(req, timeout=180) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    return obj["choices"][0]["message"]["content"], obj.get("usage") or {}


def call_with_retry(api_key, system_prompt, user_msg, *, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            return call_openrouter(api_key, system_prompt, user_msg)
        except (error.HTTPError, error.URLError, TimeoutError) as e:
            transient = isinstance(e, error.HTTPError) and 500 <= e.code < 600
            transient |= isinstance(e, (error.URLError, TimeoutError))
            if not transient or attempt == max_attempts:
                raise
            wait = attempt * attempt
            print(f"  [retry {attempt}/{max_attempts}] {type(e).__name__}: backoff {wait}s")
            time.sleep(wait)


def parse_jsonl_response(text, batch_ids):
    expected = set(batch_ids)
    found = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("```"):
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "id" not in obj or "keep" not in obj:
            continue
        found[int(obj["id"])] = obj
    if set(found.keys()) != expected:
        return None
    return [found[i] for i in batch_ids]


def append_jsonl(path, rows):
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_merged_summaries(v2_rows):
    """v1(keep=true) + v2(keep=true) 합쳐 최종 kept 작성."""
    # v1 keep
    v1_kept = {}
    for ln in JUDG_V1.read_text().splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        if obj.get("keep"):
            v1_kept[obj["lemma"]] = obj
    # v2 결과 (재판정된 것만, drop만 입력했으므로 v2 keep = 새로 추가된 것)
    v2_kept = {}
    for r in v2_rows:
        if r.get("keep"):
            v2_kept[r["lemma"]] = r

    # merge — v1 우선, v2가 새로 keep으로 바꾼 것 추가
    final = {**v1_kept, **v2_kept}

    # judgments_merged: v1 keep + v2 결과 전체 (drop 포함)
    merged_lines = []
    seen_lemmas = set()
    # v1 모두 (keep만 다시 + drop은 v2에서 재판정된 것으로 대체)
    for ln in JUDG_V1.read_text().splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        if obj.get("keep"):
            merged_lines.append(obj)
            seen_lemmas.add(obj["lemma"])
    # v2 재판정 결과 추가 (v1에서 drop이었던 것의 새 판정)
    for r in v2_rows:
        if r["lemma"] not in seen_lemmas:
            merged_lines.append(r)

    JUDG_MERGED.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged_lines) + "\n")

    # kept_lemmas
    cat_counts = {}
    kept_list = []
    for r in final.values():
        cat = r.get("category") or "other"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        kept_list.append((r["lemma"], r.get("freq", 0), cat))
    kept_list.sort(key=lambda x: -x[1])
    KEPT_MERGED.write_text("freq\tlemma\tcategory\n" +
                           "\n".join(f"{f}\t{l}\t{c}" for l, f, c in kept_list) + "\n")
    lines = [f"{c}\t{n}" for c, n in sorted(cat_counts.items(), key=lambda x: -x[1])]
    CATCOUNT_MERGED.write_text("\n".join(lines) + "\n")
    print(f"  v1 keep: {len(v1_kept)}, v2 new keep: {len(v2_kept)}, merged total: {len(final)}")
    print(f"  merged categories: {len(cat_counts)}")


def main():
    env_src = load_dotenv_into_environ()
    if env_src:
        print(f"loaded OPENROUTER_API_KEY from {env_src}")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY not set")

    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    drops = load_drops()
    done = load_done_ids()
    pending = [r for r in drops if r["id"] not in done]
    print(f"v1 drops: {len(drops)}, done: {len(done)}, pending: {len(pending)}")

    if pending:
        total_cost = 0.0
        n_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
        for bi in range(n_batches):
            batch = pending[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
            batch_ids = [r["id"] for r in batch]
            user_payload = "[" + ",\n ".join(
                json.dumps({"id": r["id"], "lemma": r["lemma"], "freq": r["freq"]}, ensure_ascii=False)
                for r in batch
            ) + "]"
            print(f"[batch {bi + 1}/{n_batches}] {len(batch)} lemmas (ids {batch_ids[0]}~{batch_ids[-1]})")
            try:
                content, usage = call_with_retry(api_key, system_prompt, user_payload)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                append_jsonl(FAILED, [{"batch": bi + 1, "ids": batch_ids, "error": str(e)}])
                continue
            parsed = parse_jsonl_response(content, batch_ids)
            if parsed is None:
                print("  parse mismatch")
                append_jsonl(FAILED, [{"batch": bi + 1, "ids": batch_ids, "raw": content[:2000]}])
                continue
            for r, p in zip(batch, parsed):
                p["freq"] = r["freq"]
            append_jsonl(JUDG_V2, parsed)
            cost = usage.get("cost", 0.0) or 0.0
            total_cost += cost
            kept_count = sum(1 for p in parsed if p.get("keep"))
            print(f"  OK — v2 kept {kept_count}/{len(parsed)}, cost ${cost:.4f}")
        print(f"\nv2 cost: ${total_cost:.4f}")

    # 통합 요약
    v2_rows = [json.loads(ln) for ln in JUDG_V2.read_text().splitlines() if ln.strip()]
    write_merged_summaries(v2_rows)


if __name__ == "__main__":
    main()
