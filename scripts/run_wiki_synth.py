#!/usr/bin/env python3
"""Wiki 본문 합성 — DeepSeek v4 Flash로 5세 explainer 1건씩 병렬 생성.

입력 조인:
  data/wiki-topic-filter/judgments.jsonl   (keep=true만 anchor 사용)
  data/wiki-topic-filter/titles.jsonl      (id → title)
  data/external-all-raw/wiki.txt           (id=line number → full body)

출력:
  data/wiki-synth/raw.jsonl       호출별 record (ok/fail 둘 다, append)
  data/wiki-synth/progress.json   처리된 id set (resume)
  data/wiki-synth/failed.jsonl    실패만 (재시도 없음 정책 — 수동 검수)
  data/wiki-synth/wiki.txt        ok=True만, 1 line=1 doc (외부 wiki.txt 호환)

설정:
- 모델: deepseek/deepseek-v4-flash
- 병렬: ThreadPoolExecutor(max_workers=16)
- temperature 0.3 / reasoning off / provider DeepSeek 강제 / json_object / usage include
- body length cap: BODY_CHAR_CAP (default 8000) — p99 trim해 비용/지연 통제
- 재시도: **없음** (사용자 정책)
- cost cap: --cost-cap (default 1.0 USD)
- 테스트: --limit N → 처음 N건만 호출

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

# ── 경로 ─────────────────────────────────────────────────────────────
TOPIC_FILTER_DIR = Path("data/wiki-topic-filter")
JUDGMENTS_IN = TOPIC_FILTER_DIR / "judgments.jsonl"
TITLES_IN = TOPIC_FILTER_DIR / "titles.jsonl"
WIKI_BODY = Path("data/external-all-raw/wiki.txt")

OUT_DIR = Path("data/wiki-synth")
RAW = OUT_DIR / "raw.jsonl"
PROGRESS = OUT_DIR / "progress.json"
FAILED = OUT_DIR / "failed.jsonl"
WIKI_OUT = OUT_DIR / "wiki.txt"

SYSTEM_PROMPT_PATH = Path("prompts/wiki_synth_system.txt")

# ── 호출 설정 ────────────────────────────────────────────────────────
MODEL = "deepseek/deepseek-v4-flash"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ENV_PATHS = [Path(".env"), Path("../llm-playground/.env")]
WORKERS = 16
TEMPERATURE = 0.3
MAX_TOKENS = 1500          # output cap. 120-200 word는 ~300 token, 5x 여유.
BODY_CHAR_CAP = 8000       # 본문 trim 길이 (p90 ≈ 5872, p99 ≈ 19541 — 상위 10% trim)
HTTP_TIMEOUT = 180


# ── 키 로딩 (run_wiki_topic_filter.py 패턴 재사용) ────────────────
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


# ── 데이터 조인 ─────────────────────────────────────────────────────
def load_anchors() -> list[dict]:
    """judgments(keep) × titles × wiki body → [{id,title,category,body}, ...]"""
    titles_by_id = {}
    for ln in TITLES_IN.read_text().splitlines():
        if ln.strip():
            o = json.loads(ln)
            titles_by_id[o["id"]] = o

    # wiki.txt: line number = id (extract 스크립트와 동일 로직, 빈 줄 없음을 사전 확인했음)
    body_by_id: dict[int, str] = {}
    with WIKI_BODY.open() as f:
        for i, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            body_by_id[i] = line

    anchors: list[dict] = []
    for ln in JUDGMENTS_IN.read_text().splitlines():
        if not ln.strip():
            continue
        j = json.loads(ln)
        if not j.get("keep"):
            continue
        i = j["id"]
        body = body_by_id.get(i, "")
        if len(body) > BODY_CHAR_CAP:
            body = body[:BODY_CHAR_CAP] + "..."
        anchors.append({
            "id": i,
            "title": j.get("title") or titles_by_id.get(i, {}).get("title", ""),
            "category": j.get("category") or "",
            "body": body,
        })
    anchors.sort(key=lambda r: r["id"])
    return anchors


def load_progress() -> set[int]:
    if not PROGRESS.exists():
        return set()
    try:
        return set(json.loads(PROGRESS.read_text()).get("done", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def save_progress(done: set[int], lock: threading.Lock) -> None:
    with lock:
        PROGRESS.write_text(json.dumps({"done": sorted(done)}))


# ── OpenRouter 호출 ────────────────────────────────────────────────
def call_one(api_key: str, system_prompt: str, anchor: dict) -> dict:
    """반환 record: id/title/category/ok/text/error/meta."""
    body = anchor["body"]
    body_in_msg = body.replace("\\n", "\n")  # 외부 wiki.txt의 literal \n을 진짜 줄바꿈으로 풀어 LLM에 보여줌
    user_msg = (
        f"INPUT title={anchor['title']} category={anchor['category']}\n"
        f"ARTICLE: {body_in_msg}"
    )
    payload = {
        "model": MODEL,
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
        return _record(anchor, ok=False, text=None, error=f"http: {e}", meta={"elapsed_s": time.time() - t0, "usage": {}})
    except (KeyError, json.JSONDecodeError) as e:
        return _record(anchor, ok=False, text=None, error=f"response: {e}", meta={"elapsed_s": time.time() - t0, "usage": {}})

    try:
        parsed = json.loads(content)
        text = (parsed.get("text") or "").strip()
    except json.JSONDecodeError:
        return _record(anchor, ok=False, text=None, error=f"json_decode: {content[:200]}",
                       meta={"elapsed_s": elapsed, "usage": usage})

    if not text:
        return _record(anchor, ok=False, text=None, error="empty_text",
                       meta={"elapsed_s": elapsed, "usage": usage})

    return _record(anchor, ok=True, text=text, error=None,
                   meta={"elapsed_s": round(elapsed, 2), "usage": usage})


def _record(anchor: dict, ok: bool, text: str | None, error: str | None, meta: dict) -> dict:
    return {
        "id": anchor["id"],
        "title": anchor["title"],
        "category": anchor["category"],
        "ok": ok,
        "text": text,
        "error": error,
        "meta": meta,
    }


def append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── wiki.txt 빌드 ──────────────────────────────────────────────────
def build_wiki_txt() -> tuple[int, int]:
    """raw.jsonl을 읽어 ok=True record를 외부 wiki.txt와 같은 포맷으로 출력.

    각 라인: f"{title}\\n\\n{text_with_literal_newlines}"
      - text 안의 진짜 줄바꿈은 literal "\\n" 으로 escape (외부 wiki.txt 컨벤션).
    반환: (ok_count, total_count)
    """
    if not RAW.exists():
        return (0, 0)
    ok = 0
    total = 0
    with WIKI_OUT.open("w", encoding="utf-8") as fout:
        # id 순서로 정렬해 결정적 출력
        rows = [json.loads(ln) for ln in RAW.read_text().splitlines() if ln.strip()]
        # 중복 id 발생 시 마지막 record 우선 (재시도 대비)
        latest: dict[int, dict] = {}
        for r in rows:
            latest[r["id"]] = r
        for i in sorted(latest.keys()):
            r = latest[i]
            total += 1
            if not r.get("ok"):
                continue
            text = r["text"].replace("\r\n", "\n")
            text_literal = text.replace("\n", "\\n")
            title = r["title"]
            fout.write(f"{title}\\n\\n{text_literal}\n")
            ok += 1
    return (ok, total)


# ── 메인 ───────────────────────────────────────────────────────────
def main() -> None:
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="처음 N건만 호출 (0=전체)")
    ap.add_argument("--shuffle-seed", type=int, default=None, help="todo 순서 결정적 shuffle (limit과 조합 가능)")
    ap.add_argument("--cost-cap", type=float, default=1.0, help="누적 cost(USD) 한도 — 도달 시 중단")
    ap.add_argument("--model", type=str, default=MODEL, help=f"OpenRouter model id (default: {MODEL})")
    ap.add_argument("--build-only", action="store_true", help="호출 없이 raw.jsonl만 읽어 wiki.txt 빌드")
    args = ap.parse_args()
    MODEL = args.model
    print(f"model: {MODEL}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.build_only:
        ok, total = build_wiki_txt()
        print(f"build only: wiki.txt ok={ok} total={total}")
        return

    env_src = load_dotenv_into_environ()
    if env_src:
        print(f"loaded OPENROUTER_API_KEY from {env_src}")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY not set. Set env or .env / ../llm-playground/.env")

    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    anchors = load_anchors()
    done = load_progress()
    todo = [a for a in anchors if a["id"] not in done]
    if args.shuffle_seed is not None:
        import random as _r
        _r.Random(args.shuffle_seed).shuffle(todo)
    if args.limit > 0:
        todo = todo[: args.limit]
    print(f"anchors total={len(anchors)} done={len(done)} todo={len(todo)} workers={WORKERS} cost_cap=${args.cost_cap}"
          + (f" shuffle_seed={args.shuffle_seed}" if args.shuffle_seed is not None else ""))

    if not todo:
        print("nothing to do.")
        ok, total = build_wiki_txt()
        print(f"wiki.txt ok={ok} total={total}")
        return

    lock = threading.Lock()
    state = {"ok": 0, "fail": 0, "cost": 0.0, "aborted": False}
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(call_one, api_key, system_prompt, a): a for a in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                rec = fut.result()
            except Exception as e:
                a = futures[fut]
                rec = _record(a, ok=False, text=None, error=f"future: {e}", meta={"elapsed_s": 0.0, "usage": {}})

            append_jsonl(RAW, rec, lock)
            if not rec["ok"]:
                append_jsonl(FAILED, rec, lock)

            cost = (rec.get("meta", {}).get("usage") or {}).get("cost", 0.0) or 0.0
            with lock:
                if rec["ok"]:
                    state["ok"] += 1
                else:
                    state["fail"] += 1
                state["cost"] += cost
                done.add(rec["id"])
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
                    print(f"  [{i}/{len(todo)}] ok={state['ok']} fail={state['fail']} "
                          f"cost=${state['cost']:.4f} rate={rate:.2f}/s ETA={eta:.1f}min")
            if i % 100 == 0:
                save_progress(done, lock)
            if state["aborted"]:
                break

    save_progress(done, lock)
    print(f"\ncall done: ok={state['ok']} fail={state['fail']} cost=${state['cost']:.4f} "
          f"elapsed={(time.time() - t_start)/60:.1f}min")

    print("\nbuilding wiki.txt ...")
    ok, total = build_wiki_txt()
    print(f"wiki.txt ok={ok} total={total} -> {WIKI_OUT}")


if __name__ == "__main__":
    main()
