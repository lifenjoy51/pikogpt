#!/usr/bin/env python3
"""
vital_titles.json (en.wikipedia 기준 1003개) 의 표제어를 simplewiki에 매칭.

1) simplewiki_clean.jsonl 의 title set과 case-insensitive exact match
2) 매칭 안 된 것은 simple.wikipedia.org MediaWiki API 의 normalized + redirects 따라가서 재매칭
3) (선택) 그래도 안 되면 en.wikipedia.org API로 redirect 따라간 후 그 target을 simplewiki에서 다시 찾기

산출:
  data/external/vital_articles/vital_titles_resolved.json
    각 entry: {"level": N, "simplewiki_title": str|null, "method": str}
    method ∈ {"exact", "simple_redirect", "en_redirect", "missing"}
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

VITAL_DIR = Path("data/external/vital_articles")
CLEAN = Path("data/simplewiki/simplewiki_clean.jsonl")

USER_AGENT = "PikoGPT-vital-resolver/1.0 (joey.51@kakaocorp.com)"


def api_query(host: str, titles: list[str], timeout: int = 30) -> dict:
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "redirects": "1",
        "format": "json",
        "formatversion": "2",
    }
    url = f"https://{host}/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def follow(body: dict, originals: list[str]) -> dict:
    q = body.get("query", {})
    norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
    redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
    pages = {p["title"]: p for p in q.get("pages", [])}

    out = {}
    for orig in originals:
        step1 = norm.get(orig, orig)
        # redirect 체인이 여러 단계일 수 있음 - 최대 5번까지 따라감
        cur = step1
        for _ in range(5):
            nxt = redir.get(cur)
            if nxt is None:
                break
            cur = nxt
        page = pages.get(cur)
        if page is None:
            out[orig] = {"final": None, "missing": True}
        elif page.get("missing"):
            out[orig] = {"final": cur, "missing": True}
        else:
            out[orig] = {"final": page["title"], "missing": False}
    return out


def batch(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    vital = json.load(open(VITAL_DIR / "vital_titles.json"))

    simple_titles = set()
    with open(CLEAN) as f:
        for line in f:
            d = json.loads(line)
            t = d.get("title", "").strip()
            if t:
                simple_titles.add(t)
    simple_lower = {t.lower(): t for t in simple_titles}
    print(f"simplewiki titles: {len(simple_titles):,}", file=sys.stderr)

    final = {}
    unmatched = []
    for t, info in vital.items():
        if t.lower() in simple_lower:
            final[t] = {
                "level": info["level"],
                "category": info.get("category"),
                "simplewiki_title": simple_lower[t.lower()],
                "method": "exact",
            }
        else:
            unmatched.append(t)

    print(f"exact-matched: {len(final)} / {len(vital)}", file=sys.stderr)
    print(f"to resolve via simplewiki API: {len(unmatched)}", file=sys.stderr)

    # Stage 1: simplewiki API
    still_unmatched = []
    for chunk in batch(unmatched, 50):
        try:
            body = api_query("simple.wikipedia.org", chunk)
        except Exception as e:
            print(f"  simple API error: {e}", file=sys.stderr)
            still_unmatched.extend(chunk)
            continue
        result = follow(body, chunk)
        for orig in chunk:
            r = result[orig]
            if r["missing"] or r["final"] is None:
                still_unmatched.append(orig)
                continue
            final_lower = r["final"].lower()
            if final_lower in simple_lower:
                final[orig] = {
                    "level": vital[orig]["level"],
                    "category": vital[orig].get("category"),
                    "simplewiki_title": simple_lower[final_lower],
                    "method": "simple_redirect" if r["final"] != orig else "exact",
                }
            else:
                # API says it exists but it's not in simplewiki_clean.jsonl
                # (cleaner가 stub/짧은 글 컷한 경우 가능)
                still_unmatched.append(orig)
        time.sleep(0.3)

    print(f"after simple API: matched={len(final)}, still unmatched={len(still_unmatched)}",
          file=sys.stderr)

    # Stage 2: en.wikipedia API → redirect target → simplewiki에 그 target이 있는지 확인
    # 환경변수 SKIP_EN=1 로 비활성화 가능 (대규모 L5 처리 시 시간 절약)
    import os as _os
    if _os.environ.get("SKIP_EN") == "1":
        print(f"SKIP_EN=1 — en redirect stage 건너뜀 (still_unmatched={len(still_unmatched)})", file=sys.stderr)
        en_resolved_then_unmatched = list(still_unmatched)
        still_unmatched = []
    en_resolved_then_unmatched = []
    for chunk in batch(still_unmatched, 50):
        try:
            body = api_query("en.wikipedia.org", chunk)
        except Exception as e:
            print(f"  en API error: {e}", file=sys.stderr)
            en_resolved_then_unmatched.extend(chunk)
            continue
        result = follow(body, chunk)

        # en redirect target들을 모아서 simplewiki에 있는지 한 번에 확인
        en_targets = []
        for orig in chunk:
            r = result[orig]
            if r["final"] and not r["missing"]:
                en_targets.append((orig, r["final"]))
        # 1단계: en target이 이미 simple_lower에 있는지 직접 확인
        unresolved_after_en = []
        for orig, en_target in en_targets:
            if en_target.lower() in simple_lower:
                final[orig] = {
                    "level": vital[orig]["level"],
                    "category": vital[orig].get("category"),
                    "simplewiki_title": simple_lower[en_target.lower()],
                    "method": "en_redirect",
                }
            else:
                unresolved_after_en.append((orig, en_target))

        # 2단계: en target을 다시 simplewiki API에 redirect 해소 요청
        if unresolved_after_en:
            target_titles = [tt for _, tt in unresolved_after_en]
            try:
                body2 = api_query("simple.wikipedia.org", target_titles)
            except Exception as e:
                print(f"  simple API (stage2) error: {e}", file=sys.stderr)
                en_resolved_then_unmatched.extend(o for o, _ in unresolved_after_en)
                continue
            result2 = follow(body2, target_titles)
            for orig, en_target in unresolved_after_en:
                r2 = result2.get(en_target, {"final": None, "missing": True})
                if not r2["missing"] and r2["final"] and r2["final"].lower() in simple_lower:
                    final[orig] = {
                        "level": vital[orig]["level"],
                        "category": vital[orig].get("category"),
                        "simplewiki_title": simple_lower[r2["final"].lower()],
                        "method": "en_redirect+simple_redirect",
                    }
                else:
                    en_resolved_then_unmatched.append(orig)

        # en API에 missing인 표제어들도 unmatched로
        for orig in chunk:
            if orig not in final and orig not in en_resolved_then_unmatched:
                en_resolved_then_unmatched.append(orig)
        time.sleep(0.3)

    # 미매칭 최종
    truly_missing = [t for t in vital if t not in final]
    for t in truly_missing:
        final[t] = {
            "level": vital[t]["level"],
            "category": vital[t].get("category"),
            "simplewiki_title": None,
            "method": "missing",
        }

    # 정렬해서 저장
    out = {t: final[t] for t in sorted(final, key=lambda x: (final[x]["level"], x))}
    out_path = VITAL_DIR / "vital_titles_resolved.json"
    with out_path.open("w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 또 단순한 매칭 표제어 텍스트 (코퍼스 빌더가 직접 쓸 수 있는 형태)
    out_txt = VITAL_DIR / "vital_simplewiki_titles.txt"
    with out_txt.open("w") as f:
        for t, info in out.items():
            if info["simplewiki_title"]:
                cat = info.get("category") or "-"
                f.write(f"L{info['level']}\t{cat}\t{info['simplewiki_title']}\t{info['method']}\n")

    # 통계
    c_method = Counter(v["method"] for v in out.values())
    c_lv = Counter((v["level"], v["simplewiki_title"] is not None) for v in out.values())
    print()
    print("=== resolution methods ===")
    for m, n in c_method.most_common():
        print(f"  {m}: {n}")
    print()
    print("=== matched per level ===")
    levels_present = sorted({lv for lv, _ in c_lv})
    for lv in levels_present:
        ok = c_lv[(lv, True)]
        miss = c_lv[(lv, False)]
        total = ok + miss
        print(f"  L{lv}: {ok}/{total}  ({100*ok/total:.1f}%)")
    print()
    print("=== final missing (sample 20) ===")
    miss_titles = [t for t, info in out.items() if info["simplewiki_title"] is None]
    for t in miss_titles[:20]:
        print(f"  L{out[t]['level']}: {t}")
    if len(miss_titles) > 20:
        print(f"  ... and {len(miss_titles) - 20} more")
    print()
    print(f"wrote {out_path}")
    print(f"wrote {out_txt}")


if __name__ == "__main__":
    main()
