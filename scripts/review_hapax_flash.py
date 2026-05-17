#!/usr/bin/env python3
"""rare_hapax.tsv 4,525 단어를 DeepSeek v4 Flash로 검토 — 비정상 어휘 식별.

배치 크기 200, ThreadPoolExecutor 16 worker 병렬.

입력: data/ccmc-all-raw/_vocab_review/rare_hapax.tsv
출력:
  data/ccmc-all-raw/_vocab_review/flash_abnormal.jsonl  (배치별 raw 응답)
  data/ccmc-all-raw/_vocab_review/flash_abnormal_summary.tsv  (word\treason)
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
RAW_PATH = OUT_DIR / "flash_abnormal.jsonl"
SUMMARY_PATH = OUT_DIR / "flash_abnormal_summary.tsv"

MODEL = "deepseek/deepseek-v4-flash"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ENV_PATHS = [Path(".env"), Path("../llm-playground/.env")]
TEMPERATURE = 0.0
MAX_TOKENS = 4000
HTTP_TIMEOUT = 240
BATCH_SIZE = 200
WORKERS = 16

SYSTEM_PROMPT = """You are a corpus quality reviewer for a 5-year-old English learning corpus that includes some foreign loanwords for language learning, proper nouns, scientific terms, and onomatopoeia.

You will be given a LIST OF WORDS (one per line). Identify which are ABNORMAL.

ABNORMAL means clearly broken text:
- English typos (e.g., "tahat" for "that", "thep" for "them")
- Truncated words (e.g., "trdeln" missing characters, missing endings)
- Random garbage / non-word fragments
- Misspellings that look like LLM hallucinations

NOT abnormal (KEEP — do NOT list these):
- Rare but real English words (e.g., "ineffable", "horticulturist")
- Proper nouns: people, places, brands, titles (e.g., "Wollstonecraft", "Schwarzenegger", "Czechoslovakia", "Jayawardenepura", "Constantinople")
- Foreign loanwords / non-English words (e.g., "olá", "danke", "Tschüss", "trdelník", "köszönöm", "dzień", "dobry", "hola", "madre", "mutter")
- Words used in language-learning context (foreign greetings/numbers/thank-yous)
- Place / city / region names from any country
- Technical / scientific / medical terms (e.g., "photosynthesis", "thermodynamics", "cardiovascular")
- Plural forms, possessive forms, hyphenated compounds
- Onomatopoeia / interjections (e.g., "wheee", "pssst", "mmmm", "beeeeh")
- Roman numerals (e.g., "iii")
- Archaic but valid English (e.g., "thee", "thou")

Output JSON: {"abnormal": [{"word": "<word>", "reason": "<≤30 Korean chars>"}]}
- Only list words you are CONFIDENT are abnormal. When in doubt, KEEP (omit from list).
- reason examples: "오타 (that)", "잘린 단어", "의미불명 조각", "단어 아님"
- If no abnormal words in batch: {"abnormal": []}
- No prose. No markdown. JSON only."""


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


def load_words() -> list[str]:
    words = []
    with HAPAX_TSV.open() as f:
        next(f)  # header
        for ln in f:
            w = ln.split("\t", 1)[0].strip()
            if w:
                words.append(w)
    return words


def call_batch(api_key: str, batch_idx: int, words: list[str]) -> dict:
    user_msg = "WORDS:\n" + "\n".join(words)
    payload = {
        "model": MODEL,
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
        return {"batch": batch_idx, "words_count": len(words), "ok": False,
                "abnormal": [], "error": f"http: {e}", "elapsed_s": time.time() - t0,
                "usage": {}}
    except (KeyError, json.JSONDecodeError) as e:
        return {"batch": batch_idx, "words_count": len(words), "ok": False,
                "abnormal": [], "error": f"response: {e}", "elapsed_s": time.time() - t0,
                "usage": {}}

    try:
        parsed = json.loads(content)
        abnormal = parsed.get("abnormal", [])
    except json.JSONDecodeError as e:
        return {"batch": batch_idx, "words_count": len(words), "ok": False,
                "abnormal": [], "error": f"json: {content[:200]}",
                "elapsed_s": elapsed, "usage": usage}

    return {"batch": batch_idx, "words_count": len(words), "ok": True,
            "abnormal": abnormal, "error": None,
            "elapsed_s": round(elapsed, 2), "usage": usage}


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY not set")

    words = load_words()
    batches = [words[i:i + BATCH_SIZE] for i in range(0, len(words), BATCH_SIZE)]
    print(f"words={len(words)}  batches={len(batches)}  batch_size={BATCH_SIZE}  workers={WORKERS}")

    RAW_PATH.unlink(missing_ok=True)
    lock = threading.Lock()
    state = {"ok": 0, "fail": 0, "abnormal_count": 0, "cost": 0.0}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(call_batch, api_key, i, b): i for i, b in enumerate(batches)}
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                with RAW_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec["ok"]:
                    state["ok"] += 1
                    state["abnormal_count"] += len(rec["abnormal"])
                else:
                    state["fail"] += 1
                cost = (rec.get("usage") or {}).get("cost", 0.0) or 0.0
                state["cost"] += cost
                done = state["ok"] + state["fail"]
                print(f"  [{done}/{len(batches)}] batch={rec['batch']} "
                      f"abnormal={len(rec['abnormal'])} "
                      f"elapsed={rec.get('elapsed_s', 0)}s "
                      f"ok_total={state['ok']} abnormal_total={state['abnormal_count']} "
                      f"cost=${state['cost']:.4f}")

    elapsed = time.time() - t0
    print(f"\ndone: ok={state['ok']}/{len(batches)}  abnormal_total={state['abnormal_count']}  "
          f"cost=${state['cost']:.4f}  elapsed={elapsed:.1f}s")

    # summary
    all_abnormal: list[dict] = []
    for ln in RAW_PATH.read_text().splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        if rec.get("ok"):
            all_abnormal.extend(rec["abnormal"])
    seen = {}
    for item in all_abnormal:
        w = item.get("word", "").strip()
        if w and w not in seen:
            seen[w] = item.get("reason", "")
    with SUMMARY_PATH.open("w") as f:
        f.write("word\treason\n")
        for w in sorted(seen):
            f.write(f"{w}\t{seen[w]}\n")
    print(f"summary → {SUMMARY_PATH}  ({len(seen)} unique abnormal words)")


if __name__ == "__main__":
    main()
