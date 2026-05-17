#!/usr/bin/env python3
"""Lemma anchor 5세 적합성 필터 — DeepSeek v4 Flash (OpenRouter) batch 호출.

입력: /tmp/missing_anchors_10.tsv (freq\tlemma, lemmatize+set diff 결과 1330개)
출력:
  data/lemma-anchor-filter/judgments.jsonl   keep/drop 결정
  data/lemma-anchor-filter/kept_lemmas.txt   keep=true lemma만 (빈도 보존)
  data/lemma-anchor-filter/category_counts.txt
  data/lemma-anchor-filter/failed.jsonl

설정:
- 모델: deepseek/deepseek-v4-flash
- batch: 133 (정확히 10 batches × 133 = 1330)
- shuffle: seed=51 (결정적)
- reasoning: off
- retry: 5xx/timeout 3회 backoff (memory feedback_retry_policy)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from urllib import error, request

INPUT_TSV = Path("/tmp/missing_anchors_10.tsv")
OUT_DIR = Path("data/lemma-anchor-filter")
JUDGMENTS = OUT_DIR / "judgments.jsonl"
KEPT = OUT_DIR / "kept_lemmas.txt"
CATCOUNT = OUT_DIR / "category_counts.txt"
FAILED = OUT_DIR / "failed.jsonl"

SYSTEM_PROMPT_PATH = Path("prompts/lemma_anchor_filter_system.txt")

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
        try:
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if not ln or ln.startswith("#") or "=" not in ln:
                    continue
                k, v = ln.split("=", 1)
                if k.strip() == "OPENROUTER_API_KEY":
                    os.environ["OPENROUTER_API_KEY"] = v.strip().strip('"').strip("'")
                    return str(p)
        except PermissionError:
            continue
    return None


def load_lemmas() -> list[dict]:
    """tsv → [{'id': i, 'lemma': l, 'freq': f}], shuffle(seed=51)."""
    rows = []
    for ln in INPUT_TSV.read_text().splitlines():
        if not ln.strip() or ln.startswith("freq\t"):
            continue
        freq_s, lemma = ln.split("\t", 1)
        rows.append({"lemma": lemma.strip(), "freq": int(freq_s)})
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(rows)
    for i, r in enumerate(rows, start=1):
        r["id"] = i
    return rows


def load_done_ids() -> set[int]:
    if not JUDGMENTS.exists():
        return set()
    done = set()
    for ln in JUDGMENTS.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            done.add(json.loads(ln)["id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def call_openrouter(api_key: str, system_prompt: str, user_msg: str) -> tuple[str, dict]:
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
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=180) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    return obj["choices"][0]["message"]["content"], obj.get("usage") or {}


def call_with_retry(api_key, system_prompt, user_msg, *, max_attempts=3):
    """5xx/timeout 3회 backoff (1s, 4s, 9s)."""
    last_exc = None
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
            last_exc = e
    raise last_exc  # unreachable


def parse_jsonl_response(text: str, batch_ids: list[int]) -> list[dict] | None:
    expected = set(batch_ids)
    found: dict[int, dict] = {}
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


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_summaries(all_lemmas: list[dict]) -> None:
    """judgments → kept_lemmas.txt (freq 보존) + category_counts.txt."""
    if not JUDGMENTS.exists():
        return
    freq_by_lemma = {r["lemma"]: r["freq"] for r in all_lemmas}
    kept: list[tuple[str, int, str]] = []
    cat_counts: dict[str, int] = {}
    for ln in JUDGMENTS.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if obj.get("keep"):
            l = obj["lemma"]
            cat = obj.get("category") or "other"
            kept.append((l, freq_by_lemma.get(l, 0), cat))
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    kept.sort(key=lambda x: -x[1])
    KEPT.write_text("freq\tlemma\tcategory\n" +
                    "\n".join(f"{f}\t{l}\t{c}" for l, f, c in kept) + "\n")
    lines = [f"{c}\t{n}" for c, n in sorted(cat_counts.items(), key=lambda x: -x[1])]
    CATCOUNT.write_text("\n".join(lines) + "\n")
    print(f"  kept={len(kept)}  categories={len(cat_counts)}")


def main() -> None:
    env_src = load_dotenv_into_environ()
    if env_src:
        print(f"loaded OPENROUTER_API_KEY from {env_src}")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: OPENROUTER_API_KEY not set.\n"
            "  export OPENROUTER_API_KEY=sk-or-... 또는 ./.env / ../llm-playground/.env"
        )

    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_lemmas = load_lemmas()
    done = load_done_ids()
    pending = [r for r in all_lemmas if r["id"] not in done]
    print(f"Total: {len(all_lemmas)}, done: {len(done)}, pending: {len(pending)}")

    if not pending:
        print("All done. Writing summaries.")
        write_summaries(all_lemmas)
        return

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
            print("  parse mismatch — saving raw to failed")
            append_jsonl(FAILED, [{"batch": bi + 1, "ids": batch_ids, "raw": content[:2000]}])
            continue

        # 각 record에 freq 보존
        for r, p in zip(batch, parsed):
            p["freq"] = r["freq"]
        append_jsonl(JUDGMENTS, parsed)
        cost = usage.get("cost", 0.0) or 0.0
        total_cost += cost
        kept_count = sum(1 for p in parsed if p.get("keep"))
        print(f"  OK — kept {kept_count}/{len(parsed)}, cost ${cost:.4f}")

    write_summaries(all_lemmas)
    print(f"\nTotal cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
