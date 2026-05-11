#!/usr/bin/env python3
"""Wiki 표제어 필터 — DeepSeek v4 Flash (OpenRouter) batch 호출.

입력:  data/wiki-topic-filter/titles.jsonl  (extract_wiki_titles_with_preview.py 출력)
출력:
  data/wiki-topic-filter/judgments.jsonl   (모든 keep/drop 결정, append)
  data/wiki-topic-filter/kept_titles.txt   (keep=true title만)
  data/wiki-topic-filter/category_counts.txt
  data/wiki-topic-filter/failed.jsonl      (batch 단위 실패 기록, drop 안 함)

설정:
- 모델: deepseek/deepseek-v4-flash
- batch: 100
- reasoning: 끄기 (Flash엔 영향 없지만 일관성)
- 재시도: 없음 (사용자 지시)
- resume: judgments.jsonl의 마지막 id 보고 이어감

키: env OPENROUTER_API_KEY=sk-or-...
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib import error, request

TITLES = Path("data/wiki-topic-filter/titles.jsonl")
OUT_DIR = Path("data/wiki-topic-filter")
JUDGMENTS = OUT_DIR / "judgments.jsonl"
KEPT = OUT_DIR / "kept_titles.txt"
CATCOUNT = OUT_DIR / "category_counts.txt"
FAILED = OUT_DIR / "failed.jsonl"

SYSTEM_PROMPT_PATH = Path("prompts/wiki_topic_filter_system.txt")
USER_TEMPLATE_PATH = Path("prompts/wiki_topic_filter_user_template.txt")

MODEL = "deepseek/deepseek-v4-flash"
BATCH_SIZE = 100
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ENV_PATHS = [Path(".env"), Path("../llm-playground/.env")]  # 첫 번째 존재하는 파일에서 로드


def load_dotenv_into_environ() -> str | None:
    """첫 번째 존재하는 ENV_PATHS에서 OPENROUTER_API_KEY=... 한 줄만 파싱해 environ에 주입.
    이미 환경변수로 잡혀 있으면 그대로 두고, 사용한 경로(있으면)를 반환."""
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


def load_titles() -> list[dict]:
    return [json.loads(ln) for ln in TITLES.read_text().splitlines() if ln.strip()]


def load_done_ids() -> set[int]:
    if not JUDGMENTS.exists():
        return set()
    done: set[int] = set()
    for ln in JUDGMENTS.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            done.add(json.loads(ln)["id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def call_openrouter(api_key: str, system_prompt: str, user_msg: str) -> tuple[str, dict]:
    """returns (content, usage_dict). usage_dict는 prompt_tokens/completion_tokens/cost 등."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "reasoning": {"enabled": False, "exclude": True},
        "provider": {"only": ["DeepSeek"], "allow_fallbacks": False},
        "usage": {"include": True},  # OpenRouter: 응답에 usage.cost(USD) 포함시키기
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
    with request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    return obj["choices"][0]["message"]["content"], obj.get("usage") or {}


def parse_jsonl_response(text: str, batch_ids: list[int]) -> list[dict] | None:
    """응답에서 JSONL 추출. id 누락/포맷 불일치면 None."""
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


def write_summaries() -> None:
    """judgments.jsonl을 읽어 kept_titles.txt + category_counts.txt 생성."""
    if not JUDGMENTS.exists():
        return
    kept: list[str] = []
    cat_counts: dict[str, int] = {}
    for ln in JUDGMENTS.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if obj.get("keep"):
            kept.append(obj["title"])
            cat = obj.get("category") or "other"
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    KEPT.write_text("\n".join(kept) + "\n")
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
            "  Set via env: export OPENROUTER_API_KEY=sk-or-...\n"
            "  Or place in ./.env or ../llm-playground/.env"
        )

    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    user_template = USER_TEMPLATE_PATH.read_text()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_titles = load_titles()
    # titles.jsonl 자체가 이미 shuffle된 순서로 저장돼 있음 (extract 스크립트가 seed=42).
    done_ids = load_done_ids()
    pending = [t for t in all_titles if t["id"] not in done_ids]
    print(f"titles total={len(all_titles)} done={len(done_ids)} pending={len(pending)}")

    n_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
    max_batches = int(sys.argv[1]) if len(sys.argv) > 1 else n_batches
    run_batches = min(n_batches, max_batches)
    if run_batches < n_batches:
        print(f"  limited to first {run_batches} batches (test mode)")
    total_cost = [0.0]  # mutable list for closure-style update inside loop
    for b in range(run_batches):
        batch = pending[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        batch_ids = [t["id"] for t in batch]
        batch_json = json.dumps(batch, ensure_ascii=False)
        user_msg = user_template.replace("{BATCH_JSON}", batch_json)

        t0 = time.time()
        try:
            raw, usage = call_openrouter(api_key, system_prompt, user_msg)
        except (error.URLError, error.HTTPError, TimeoutError) as e:
            print(f"[batch {b + 1}/{n_batches}] HTTP error: {e} — recorded to failed.jsonl, no retry")
            append_jsonl(FAILED, [{"batch_index": b, "ids": batch_ids, "error": str(e)}])
            continue
        except (KeyError, json.JSONDecodeError) as e:
            print(f"[batch {b + 1}/{n_batches}] response parse error: {e} — recorded to failed.jsonl, no retry")
            append_jsonl(FAILED, [{"batch_index": b, "ids": batch_ids, "error": str(e)}])
            continue

        rows = parse_jsonl_response(raw, batch_ids)
        if rows is None:
            print(f"[batch {b + 1}/{n_batches}] JSONL schema mismatch — recorded to failed.jsonl, no retry")
            append_jsonl(FAILED, [{"batch_index": b, "ids": batch_ids, "raw": raw[:2000]}])
            continue

        append_jsonl(JUDGMENTS, rows)
        kept_n = sum(1 for r in rows if r.get("keep"))
        cost = usage.get("cost", 0.0)
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        total_cost[0] += cost
        print(f"[batch {b + 1}/{n_batches}] ok size={len(rows)} kept={kept_n} "
              f"tok={pt}/{ct} cost=${cost:.5f} cum=${total_cost[0]:.5f} elapsed={time.time() - t0:.1f}s")

    print(f"\ntotal cost this run: ${total_cost[0]:.5f}")
    print("\nfinalize summaries:")
    write_summaries()


if __name__ == "__main__":
    main()
