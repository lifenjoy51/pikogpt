#!/usr/bin/env python3
"""v6 axis 합성 — anchor universe 6,145개에 대해 axis별 DeepSeek Pro 호출.

axis별로 별도 raw_{axis}.jsonl + progress_{axis}.json + failed_{axis}.jsonl을 쓴다.
중간 중단 시 progress_{axis}.json을 보고 done anchor를 건너뛰고 이어감.

입력:
  data/v6-axes/universe.jsonl    (6,145 anchor + article)
  prompts/v6_axis_{a_wiki|b_cause|c_chained|d_counting}.txt

출력 (axis별):
  data/v6-axes/raw_{axis}.jsonl        호출별 record (ok/fail 둘 다, append)
  data/v6-axes/progress_{axis}.json    처리된 anchor set (resume)
  data/v6-axes/failed_{axis}.jsonl     실패만 (재시도 없음 정책)

설정:
- 모델: deepseek/deepseek-v4-pro
- 병렬: ThreadPoolExecutor(--workers, default 16)
- temperature 0.3 / reasoning off / provider DeepSeek 강제 / json_object / usage include
- 재시도: 없음 (사용자 정책)
- cost cap: --cost-cap (default 60 USD, 추정 $12-15의 ~4배 안전마진)
- 테스트: --limit N → 처음 N건만

키: env OPENROUTER_API_KEY 또는 ./.env / ../llm-playground/.env 자동 로드
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import error, request

UNIVERSE = Path("data/v6-axes/universe.jsonl")
OUT_DIR = Path("data/v6-axes")

AXIS_SPEC_PATHS = {
    "a": Path("prompts/v6_axis_a_wiki.txt"),
    "b": Path("prompts/v6_axis_b_cause.txt"),
    "c": Path("prompts/v6_axis_c_chained.txt"),
    "d": Path("prompts/v6_axis_d_counting.txt"),
}

MODEL = "deepseek/deepseek-v4-pro"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ENV_PATHS = [Path(".env"), Path("../llm-playground/.env")]
TEMPERATURE = 0.3
MAX_TOKENS = 1500
HTTP_TIMEOUT = 240


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


def load_universe() -> list[dict]:
    return [json.loads(ln) for ln in UNIVERSE.read_text().splitlines() if ln.strip()]


def load_progress(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()).get("done", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def save_progress(path: Path, done: set[str], lock: threading.Lock) -> None:
    with lock:
        path.write_text(json.dumps({"done": sorted(done)}))


def build_user_msg(axis: str, rec: dict) -> str:
    if axis == "a":
        article = rec.get("article")
        if article:
            body = article.replace("\\n", "\n")
            return f"INPUT title={rec['original_title']} category={rec['category']}\nARTICLE: {body}"
        else:
            return f"INPUT title={rec['original_title']} category={rec['category']}\nARTICLE: (empty)"
    else:
        cat = rec["category"] or "(none)"
        return f"INPUT anchor={rec['anchor']} category={cat}"


def call_one(api_key: str, system_prompt: str, rec: dict, axis: str, model: str) -> dict:
    user_msg = build_user_msg(axis, rec)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "reasoning": {"enabled": False, "exclude": True},
        "provider": {"only": ["DeepSeek"], "allow_fallbacks": False},
        "response_format": {"type": "json_object"},
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
    t0 = time.time()
    try:
        with request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        content = (obj["choices"][0]["message"]["content"] or "").strip()
        usage = obj.get("usage") or {}
    except (error.URLError, error.HTTPError, TimeoutError) as e:
        return _record(rec, ok=False, text=None, reason=None, error=f"http: {e}",
                       meta={"elapsed_s": time.time() - t0, "usage": {}})
    except (KeyError, json.JSONDecodeError) as e:
        return _record(rec, ok=False, text=None, reason=None, error=f"response: {e}",
                       meta={"elapsed_s": time.time() - t0, "usage": {}})

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        return _record(rec, ok=False, text=None, reason=None,
                       error=f"json_decode: {content[:200]}",
                       meta={"elapsed_s": elapsed, "usage": usage})

    text = parsed.get("text")
    reason = parsed.get("reason")
    if text is None and reason is None:
        return _record(rec, ok=False, text=None, reason=None,
                       error="text and reason both null",
                       meta={"elapsed_s": elapsed, "usage": usage})

    return _record(rec, ok=True, text=text, reason=reason, error=None,
                   meta={"elapsed_s": round(elapsed, 2), "usage": usage})


def _record(rec: dict, ok: bool, text: str | None, reason: str | None,
            error: str | None, meta: dict) -> dict:
    return {
        "anchor": rec["anchor"],
        "original_title": rec["original_title"],
        "source": rec["source"],
        "category": rec["category"],
        "article_id": rec.get("article_id"),
        "ok": ok,
        "text": text,
        "reason": reason,
        "error": error,
        "meta": meta,
    }


def append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=list(AXIS_SPEC_PATHS),
                    help="a | b | c | d")
    ap.add_argument("--limit", type=int, default=0, help="처음 N건만 호출 (0=전체)")
    ap.add_argument("--workers", type=int, default=16, help="동시 worker 수")
    ap.add_argument("--cost-cap", type=float, default=60.0,
                    help="누적 cost(USD) 한도 — 도달 시 pending cancel")
    ap.add_argument("--model", default=MODEL,
                    help=f"OpenRouter model id (default: {MODEL})")
    args = ap.parse_args()

    axis = args.axis
    spec_path = AXIS_SPEC_PATHS[axis]
    raw_path = OUT_DIR / f"raw_{axis}.jsonl"
    progress_path = OUT_DIR / f"progress_{axis}.json"
    failed_path = OUT_DIR / f"failed_{axis}.jsonl"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env_src = load_dotenv_into_environ()
    if env_src:
        print(f"loaded OPENROUTER_API_KEY from {env_src}")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY not set")

    system_prompt = spec_path.read_text()
    universe = load_universe()
    done = load_progress(progress_path)
    todo = [r for r in universe if r["anchor"] not in done]
    if args.limit > 0:
        todo = todo[: args.limit]

    print(f"axis={axis}  model={args.model}  spec={spec_path.name}  universe={len(universe)}  done={len(done)}  todo={len(todo)}  workers={args.workers}  cost_cap=${args.cost_cap}")
    if not todo:
        print("nothing to do.")
        return

    lock = threading.Lock()
    state = {"ok": 0, "fail": 0, "skip": 0, "cost": 0.0, "aborted": False}
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(call_one, api_key, system_prompt, r, axis, args.model): r for r in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                rec = fut.result()
            except Exception as e:
                r = futures[fut]
                rec = _record(r, ok=False, text=None, reason=None,
                              error=f"future: {e}",
                              meta={"elapsed_s": 0.0, "usage": {}})

            append_jsonl(raw_path, rec, lock)
            if not rec["ok"]:
                append_jsonl(failed_path, rec, lock)

            cost = (rec.get("meta", {}).get("usage") or {}).get("cost", 0.0) or 0.0
            with lock:
                if rec["ok"]:
                    if rec.get("text") is None:
                        state["skip"] += 1
                    else:
                        state["ok"] += 1
                else:
                    state["fail"] += 1
                state["cost"] += cost
                done.add(rec["anchor"])
                if state["cost"] >= args.cost_cap and not state["aborted"]:
                    state["aborted"] = True
                    print(f"\n!!! cost cap reached ${state['cost']:.4f} >= ${args.cost_cap} — cancelling pending !!!")
                    for f2 in futures:
                        if not f2.done():
                            f2.cancel()

            if i % 50 == 0 or i == len(todo):
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(todo) - i) / rate / 60 if rate > 0 else 0
                with lock:
                    print(f"  [{i}/{len(todo)}] ok={state['ok']} skip={state['skip']} fail={state['fail']} "
                          f"cost=${state['cost']:.4f} rate={rate:.2f}/s ETA={eta:.1f}min")
            if i % 100 == 0:
                save_progress(progress_path, done, lock)
            if state["aborted"]:
                break

    save_progress(progress_path, done, lock)
    print(f"\ndone: ok={state['ok']} skip={state['skip']} fail={state['fail']} "
          f"cost=${state['cost']:.4f} elapsed={(time.time() - t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
