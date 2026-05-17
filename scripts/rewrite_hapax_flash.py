#!/usr/bin/env python3
"""rare_hapax.tsv 4,525 row를 DeepSeek v4 Flash로 sentence rewrite.

각 (word, sentence) → 5세 어휘 strict / 단조로운 표현으로 sentence 재작성.
재작성 불가능하면 빈칸.

입력: data/ccmc-all-raw/_vocab_review/rare_hapax.tsv  (word\tcount\tsentence)
출력:
  data/ccmc-all-raw/_vocab_review/rewrite_raw.jsonl   (batch별 raw 응답)
  data/ccmc-all-raw/_vocab_review/rare_hapax_rewritten.tsv  (word\tcount\tsentence\trewritten)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import error, request

HAPAX_TSV = Path("data/ccmc-all-raw/_vocab_review/rare_hapax.tsv")
OUT_DIR = Path("data/ccmc-all-raw/_vocab_review")
RAW_PATH = OUT_DIR / "rewrite_raw.jsonl"
OUT_TSV = OUT_DIR / "rare_hapax_rewritten.tsv"

MODEL = "deepseek/deepseek-v4-flash"  # CLI --model로 오버라이드 가능
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ENV_PATHS = [Path(".env"), Path("../llm-playground/.env")]
TEMPERATURE = 0.2
MAX_TOKENS = 8000
HTTP_TIMEOUT = 240
BATCH_SIZE = 100
WORKERS = 16

SYSTEM_PROMPT = """You are a strict curriculum editor rewriting sentences for a 5-year-old English learning corpus.

You will receive a JSON list of items, each with an "id", a "word", and a "sentence". The "word" is a rare/unusual word that appeared once in the corpus. The "sentence" contains that word.

# Your job: produce EXACTLY ONE of TWO outputs per item

(A) A SIMPLIFIED rewrite — strictly easier than the input.
(B) An EMPTY STRING "" — only when (A) is impossible.

THERE IS NO THIRD OPTION. **You MUST NOT return the input sentence unchanged.** Echoing the input is forbidden. If you cannot make it simpler, return "".

# What "simpler" means (MUST hold vs. the input)
At least ONE of these must improve:
- Replace the rare/hard word with a more common synonym (e.g., "infrastructure" → "roads and pipes", "discrimination" → "treating people badly").
- Replace any rare side-words too (e.g., "extraordinary" → "very special", "altogether" → "in all", "approximately" → "about").
- Shorten / split long clauses into shorter plain sentences.
- Remove formal or abstract phrasing in favor of concrete 5-year-old words.

Vocabulary: ONLY common 5-year-old English (~2000 words, Dolch + 1st-grade reading). If the rewrite still contains a hard word, simplify that one too.

Style: short, plain, monotone, kindergarten-textbook tone. 5-15 words per sentence. Multiple short sentences are OK if joined with a period.

Format:
- One single line (no newlines inside the rewrite).
- End with a period.
- Preserve the same topic and referent — do not invent new entities.

# When to return "" (the only acceptable refusal)
Return "" if the sentence is already perfectly simple AND replacing the target word would lose essential meaning. Specifically:
- The word is a PROPER NOUN that is essential to the sentence ("Beyoncé", "Popocatépetl", "Nintendo DS", "FC Bayern", "Notre-Dame").
- The word is a FOREIGN word being explicitly taught ("Tschüss is German for goodbye", "olá means hello in Portuguese").
- The word is an ONOMATOPOEIA being demonstrated ("sssssss" for a snake sound, "cu-ckoo" for a cuckoo bird).
- The sentence is an etymology / spelling / pronunciation lesson centered on that exact word.
- A scientific term with no 5-year-old equivalent AND the sentence is essentially about that term.

In ALL OTHER cases, you MUST simplify. The default action is to simplify, not to refuse.

# Self-check before each output
Before writing a rewrite, ask yourself:
1. Is my rewrite STRICTLY different from the input sentence? (If equal → output "")
2. Is my rewrite easier (simpler vocab / shorter / more concrete)? (If not → output "")
3. Is my rewrite still about the same topic? (If not → fix it)

# Few-shot examples

Input: {"id": 0, "word": "infrastructure", "sentence": "The city built new infrastructure for the people."}
Output rewrite: "The city built new roads and pipes for the people."

Input: {"id": 1, "word": "horticulturist", "sentence": "A horticulturist takes care of plants."}
Output rewrite: "A person who takes care of plants is a plant helper."

Input: {"id": 2, "word": "beyoncé", "sentence": "Many famous singers like Beyoncé and Usher sing rhythm and blues."}
Output rewrite: ""   (proper noun subject; cannot simplify)

Input: {"id": 3, "word": "discrimination", "sentence": "Discrimination is when people are not treated fairly because of who they are."}
Output rewrite: "Some people are mean to others just because they look different. That is not fair."

# Output JSON
{"rewrites": [{"id": <int>, "rewritten": "<simpler sentence or empty string>"}]}
- Must include all input ids in order.
- No prose, no markdown fences, JSON only.
- Final reminder: rewritten == sentence is FORBIDDEN. Use "" instead."""


def load_dotenv() -> None:
    if os.environ.get("OPENROUTER_API_KEY"):
        return
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
                    return
        except PermissionError:
            continue


def load_rows() -> list[dict]:
    rows = []
    with HAPAX_TSV.open() as f:
        next(f)
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 3:
                rows.append({"word": parts[0], "count": parts[1], "sentence": parts[2]})
            elif len(parts) == 2:
                rows.append({"word": parts[0], "count": parts[1], "sentence": ""})
    return rows


def call_batch(api_key: str, batch_idx: int, items: list[dict], model: str) -> dict:
    payload_items = [{"id": i, "word": it["word"], "sentence": it["sentence"]}
                     for i, it in enumerate(items)]
    user_msg = "ITEMS:\n" + json.dumps(payload_items, ensure_ascii=False)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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
        data=json.dumps(body).encode("utf-8"),
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
        return {"batch": batch_idx, "items_count": len(items), "ok": False,
                "rewrites": [], "error": f"http: {e}",
                "elapsed_s": round(time.time() - t0, 2), "usage": {}}
    except (KeyError, json.JSONDecodeError) as e:
        return {"batch": batch_idx, "items_count": len(items), "ok": False,
                "rewrites": [], "error": f"response: {e}",
                "elapsed_s": round(time.time() - t0, 2), "usage": {}}

    try:
        parsed = json.loads(content)
        rewrites = parsed.get("rewrites", [])
    except json.JSONDecodeError as e:
        return {"batch": batch_idx, "items_count": len(items), "ok": False,
                "rewrites": [], "error": f"json: {content[:200]}",
                "elapsed_s": round(elapsed, 2), "usage": usage}

    # id별 매핑
    by_id = {}
    for r in rewrites:
        try:
            by_id[int(r["id"])] = r.get("rewritten", "")
        except (KeyError, ValueError, TypeError):
            continue

    out_rewrites = []
    missing = 0
    for i, it in enumerate(items):
        rewritten = by_id.get(i, "")
        if i not in by_id:
            missing += 1
        out_rewrites.append({"word": it["word"], "rewritten": rewritten})

    return {"batch": batch_idx, "items_count": len(items), "ok": True,
            "rewrites": out_rewrites, "missing": missing, "error": None,
            "elapsed_s": round(elapsed, 2), "usage": usage}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL,
                    help=f"OpenRouter model id (default: {MODEL})")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY not set")

    rows = load_rows()
    batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    print(f"model={args.model}  rows={len(rows)}  batches={len(batches)}  batch_size={BATCH_SIZE}  workers={WORKERS}")

    RAW_PATH.unlink(missing_ok=True)
    lock = threading.Lock()
    state = {"ok": 0, "fail": 0, "rewrite_count": 0, "empty_count": 0, "missing_count": 0, "cost": 0.0}
    all_results: dict[str, str] = {}  # word → rewritten
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(call_batch, api_key, i, b, args.model): i for i, b in enumerate(batches)}
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                with RAW_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec["ok"]:
                    state["ok"] += 1
                    state["missing_count"] += rec.get("missing", 0)
                    # input sentence를 빠르게 조회하기 위한 dict
                    items = batches[rec["batch"]]
                    sent_by_word = {it["word"]: it["sentence"] for it in items}
                    for r in rec["rewrites"]:
                        w = r["word"]
                        rw = (r.get("rewritten") or "").strip()
                        # 동일문장 강제 빈칸 변환
                        if rw and rw == sent_by_word.get(w, ""):
                            rw = ""
                            state["echo_filtered"] = state.get("echo_filtered", 0) + 1
                        all_results[w] = rw
                        if rw:
                            state["rewrite_count"] += 1
                        else:
                            state["empty_count"] += 1
                else:
                    state["fail"] += 1
                cost = (rec.get("usage") or {}).get("cost", 0.0) or 0.0
                state["cost"] += cost
                done = state["ok"] + state["fail"]
                print(f"  [{done}/{len(batches)}] batch={rec['batch']:>2} "
                      f"elapsed={rec.get('elapsed_s', 0):>5.1f}s "
                      f"rewrites={state['rewrite_count']} empty={state['empty_count']} "
                      f"miss={state['missing_count']} fail={state['fail']} "
                      f"cost=${state['cost']:.4f}")

    elapsed = time.time() - t0
    print(f"\ndone: ok_batches={state['ok']}/{len(batches)}  "
          f"rewrites={state['rewrite_count']} empty={state['empty_count']} "
          f"echo_filtered={state.get('echo_filtered', 0)} "
          f"missing={state['missing_count']} cost=${state['cost']:.4f} "
          f"elapsed={elapsed:.1f}s")

    # TSV 출력
    out_lines = ["word\tcount\tsentence\trewritten"]
    for r in rows:
        w = r["word"]
        rw = all_results.get(w, "")
        # tab/newline 안전 처리
        rw = rw.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
        out_lines.append(f"{w}\t{r['count']}\t{r['sentence']}\t{rw}")
    OUT_TSV.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {OUT_TSV}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
